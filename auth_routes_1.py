from flask import Blueprint, request, jsonify, make_response, redirect, url_for
from db_manager import DatabaseManager # Assuming db_manager.py is in the same directory or accessible in PYTHONPATH

auth_bp = Blueprint('auth_bp', __name__)

# It's generally better to pass the db instance or have a way to get it,
# rather than creating a new one here if it's meant to be a shared instance.
# For now, assuming DatabaseManager handles its state or you instantiate it in app.py and pass it around.
# If db is a global in app.py, this will be more complex. Consider application context or passing db.
db = DatabaseManager() # This might need adjustment based on how db is managed in your main app.

@auth_bp.route('/api/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    success, user_id, message = db.authenticate_user(username, password)
    
    if success:
        session_token = db.create_session(user_id)
        user_info = db.get_user_info(user_id)
        
        response = make_response(jsonify({"success": True, "message": "Login successful", "user": user_info}))
        # response.set_cookie('session_token', session_token, httponly=True, samesite='Lax', max_age=86400) # 24 hours
        response.set_cookie(
            'session_token',
            session_token,
            httponly=True,
            samesite='None',
            max_age=86400,
            secure=True,  # ngrok is HTTPS
            
        )
        print(response.headers)

        return response
    else:
        return jsonify({"success": False, "message": message}), 401

@auth_bp.route('/api/register', methods=['POST'])
def register():
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    confirm_password = request.form.get('confirm_password')
    
    if password != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match"}), 400
    
    success, message = db.register_user(username, email, password)
    
    if success:
        return jsonify({"success": True, "message": "Registration successful. Please log in."})
    else:
        return jsonify({"success": False, "message": message}), 400

@auth_bp.route('/api/logout', methods=['GET']) # GET is fine for logout if it just clears cookie
def logout():
    session_token = request.cookies.get('session_token')
    if session_token:
        db.delete_session(session_token)
    
    response = make_response(jsonify({"success": True, "message": "Logout successful"}))
    response.delete_cookie('session_token', httponly=True, samesite='Lax')
    return response

# This decorator will be defined in the main app.py or a shared utility, 
# as it needs access to the `db` instance and potentially `request` from Flask.
# For now, we assume it will be available in the context where this blueprint is used.
# If you create a central `auth.py` for the decorator, import it here.
# For simplicity, we will replicate a basic version of it in app.py when integrating.

# A placeholder for the auth_required decorator if it were self-contained here
# from functools import wraps
# def auth_required_local(func):
#     @wraps(func)
#     def wrapper(*args, **kwargs):
#         session_token = request.cookies.get('session_token')
#         if not session_token:
#             return jsonify({"success": False, "message": "Authentication required."}), 401
#         user_id = db.validate_session(session_token)
#         if not user_id:
#             return jsonify({"success": False, "message": "Invalid or expired session."}), 401
#         g.user = db.get_user_info(user_id) # Use Flask's g for request-bound context
#         if not g.user:
#             return jsonify({"success": False, "message": "User not found."}), 401
#         return func(*args, **kwargs)
#     return wrapper

@auth_bp.route('/api/me', methods=['GET'])
# @auth_required # This will be applied in app.py when registering the blueprint, or the decorator needs to be accessible here
def current_user_info():
    # This route will be protected by auth_required in the main app
    # The user object will be attached to request or g by the auth_required decorator
    # For now, we assume `request.user` is populated by the decorator
    if hasattr(request, 'user') and request.user:
        return jsonify({"success": True, "user": request.user})
    else:
        # This case should ideally be caught by auth_required
        return jsonify({"success": False, "message": "Authentication required or user not found."}), 401 