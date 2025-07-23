import os
from flask import Flask, jsonify, make_response, request, redirect, url_for, send_from_directory, g
from flask_cors import CORS
from dotenv import load_dotenv
from functools import wraps
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from our new modules
import config
from db_manager import DatabaseManager
from models import load_clip_classification_model, load_gemini_model, load_timesformer_model
from auth_routes import auth_bp
from video_routes import video_bp, configure_app

# Initialize Flask App
app = Flask(__name__, static_folder='frontend/build', static_url_path='/')

# Configure app with middleware and settings from video_routes
configure_app(app)

# Configure CORS for local development
# CORS(app, 
#      resources={
#          r"/api/*": {
#              "origins": ["http://localhost:3000"],
#              "methods": ["GET", "POST", "OPTIONS"],
#              "allow_headers": ["Content-Type", "Authorization"],
#              "supports_credentials": True
#          }
#      })



CORS(app, supports_credentials=True, origins=["https://frontend-rho-lime-29.vercel.app"])



# Add debug logging - MODIFIED to avoid printing binary data
@app.before_request
def log_request_info():
    # Skip logging for multipart/form-data requests to avoid binary data in console
    if request.content_type and 'multipart/form-data' in request.content_type:
        print(f"\n=== New Multipart Request (details omitted) ===")
        print(f"Method: {request.method}")
        print(f"URL: {request.url}")
        print("==================\n")
        return
        
    print(f"\n=== New Request ===")
    print(f"Method: {request.method}")
    print(f"URL: {request.url}")
    print(f"Headers: {dict(request.headers)}")
    
    # Only print data for non-file requests
    if not request.files:
        try:
            # Try to print as text
            data = request.get_data().decode('utf-8', errors='replace')
            # Truncate if too long
            if len(data) > 1000:
                data = data[:1000] + "... [truncated]"
            print(f"Data: {data}")
        except:
            print("Data: [Binary or unprintable content]")
    else:
        print(f"Data: [Contains file upload - content not displayed]")
        for file_key in request.files:
            print(f"File: {file_key}, filename: {request.files[file_key].filename}")
    
    print("==================\n")

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    # Apply CSP headers
    response.headers['Content-Security-Policy'] = (
        "default-src *; "
        "img-src * data: blob:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
    )
    
    # Print response info
    print(f"\n=== Response ===")
    print(f"Status: {response.status}")
    # Optional: Uncomment to print headers if needed
    # print(f"Headers: {dict(response.headers)}")
    print("================\n")
    
    return response

# Apply configurations from config.py
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.secret_key = config.SECRET_KEY

# Initialize Database Manager
# This instance will be used by the auth_required decorator and potentially by blueprints if passed correctly.
db = DatabaseManager()

# Create necessary directories
config.create_directories()

# --- Authentication Decorator ---
def auth_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        session_token = request.cookies.get('session_token')
        if not session_token:
            return jsonify({"success": False, "message": "Authentication required. No session token."}), 401
        
        user_id = db.validate_session(session_token)
        if not user_id:
            response = make_response(jsonify({"success": False, "message": "Invalid or expired session."}), 401)
            # Clear the invalid cookie
            response.delete_cookie('session_token', httponly=True, samesite='Lax')
            return response
        
        user_info = db.get_user_info(user_id)
        if not user_info:
             response = make_response(jsonify({"success": False, "message": "Could not retrieve user details."}), 401)
             response.delete_cookie('session_token', httponly=True, samesite='Lax')
             return response
        
        # Make user info available for the duration of the request
        g.user = user_info # Using Flask's g for request-bound context
        request.user = user_info # Also attaching to request for easier access in route if not using g
        
        return func(*args, **kwargs)
    return wrapper

# Print device info
print(f"Using device: {config.DEVICE}")
if config.DEVICE.type == 'cuda':
    import torch # Import torch here if not already imported, for get_device_name
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Load models during startup
print("\n=== Loading ML Models ===")
from models import load_clip_classification_model, load_gemini_model, load_timesformer_model

print("\nLoading CLIP model...")
load_clip_classification_model()

print("\nLoading Gemini model...")
load_gemini_model()

print("\nLoading Timesformer model...")
load_timesformer_model()
print("=== ML Models Loading Complete ===\n")

# Register Blueprints
app.register_blueprint(auth_bp) # Auth routes like /api/login, /api/register, etc.
app.register_blueprint(video_bp) # Video routes like /api/upload, /api/caption

# Move the before_request decorator here, after blueprint registration
@app.before_request
def before_request():
    # Skip HTTPS redirect in development mode
    if request.endpoint and 'static' not in request.endpoint:
        if not request.is_secure and app.config.get('ENV') == 'production':
            url = request.url.replace('http://', 'https://', 1)
            return redirect(url, code=301)

# Apply auth_required to protected routes in blueprints
# This is a way to apply it if not done within the blueprint. 
# For this setup, it's assumed that relevant routes in auth_routes.py and video_routes.py
# will be decorated or the blueprint itself can have a before_request hook.
# Example of explicit decoration if not done in blueprint:
# app.view_functions['auth_bp.current_user_info'] = auth_required(app.view_functions['auth_bp.current_user_info'])
# app.view_functions['video_bp.upload_video'] = auth_required(app.view_functions['video_bp.upload_video'])
# app.view_functions['video_bp.caption_video'] = auth_required(app.view_functions['video_bp.caption_video'])
# A more common way is to apply decorator directly in blueprint routes, or use `Blueprint.before_request`.
# The current auth_routes.py and video_routes.py have placeholders for where `auth_required` would be applied.
# Let's assume the `auth_required` decorator will be added to the individual routes within the blueprint files for clarity.
# Re-checking `auth_routes.py` and `video_routes.py` to ensure they are expecting this setup.
# For the `/api/me` route in `auth_routes.py`, it should use the `g.user` or `request.user`.
# For `/api/upload` and `/api/caption` in `video_routes.py`, they are protected as well.

# Decorate specific blueprint routes after registration:
# This ensures the app context and `db` instance are available to the decorator.
# Note: The view function name is `blueprint_name.function_name`
if 'auth_bp.current_user_info' in app.view_functions:
    app.view_functions['auth_bp.current_user_info'] = auth_required(app.view_functions['auth_bp.current_user_info'])
if 'video_bp.upload_video' in app.view_functions:
    app.view_functions['video_bp.upload_video'] = auth_required(app.view_functions['video_bp.upload_video'])
if 'video_bp.caption_video' in app.view_functions:
    app.view_functions['video_bp.caption_video'] = auth_required(app.view_functions['video_bp.caption_video'])


# --- Static File Serving and Index Route ---
@app.route('/')
def index():
    # This route can serve as a basic landing page or redirect logic if needed.
    # However, the catch-all `serve_react` handles serving index.html for the React app.
    # If direct access to '/' should also serve React app, this is okay.
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

# Main execution
if __name__ == '__main__':
    print("\n=== Starting Flask Server ===")
    print("Server will be available at: http://localhost:5000")
    print("===========================\n")
    app.run(
        debug=True,
        use_reloader=True,
        host='0.0.0.0',  # Allow external connections
        port=5000,
        reloader_type='stat'
    ) 