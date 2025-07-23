from flask import Flask, request, render_template, jsonify, session
import os
import torch
from transformers import TimesformerConfig, TimesformerForVideoClassification
from preprocessing import preprocess_video
from glob import glob
import torch.nn as nn
from transformers import BlipProcessor, BlipForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from torchvision import transforms
import torch
import cv2
import os
import html
import time
from PIL import Image
import json
from flask import Flask, render_template, request, redirect, url_for, make_response, session, flash
import os
from werkzeug.utils import secure_filename

from db_manager import DatabaseManager

app = Flask(__name__)


app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max upload size = 50 MB
UPLOAD_FOLDER = 'uploads'
app.secret_key = os.urandom(24)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
db = DatabaseManager()


# Define class labels
class_labels = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 'Fighting', 'normal'
]

def allowed_file(filename):
    allowed_extensions = {'mp4', 'avi', 'mov', 'mkv'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_latest_uploaded_video():
    files = glob(os.path.join(UPLOAD_FOLDER, '*'))
    video_files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    if not video_files:
        return None
    return max(video_files, key=os.path.getmtime)

def auth_required(func):
    """Decorator to require authentication for routes."""
    def wrapper(*args, **kwargs):
        # Check if user has a session cookie
        session_token = request.cookies.get('session_token')
        if not session_token:
            return redirect(url_for('login', next=request.url))
        
        # Validate session
        user_id = db.validate_session(session_token)
        if not user_id:
            response = make_response(redirect(url_for('login', next=request.url)))
            response.delete_cookie('session_token')
            return response
        
        # Add user info to the request context
        request.user = db.get_user_info(user_id)
        return func(*args, **kwargs)
    
    # Rename the function to preserve Flask's route mapping
    wrapper.__name__ = func.__name__
    return wrapper
'''
@app.route('/')
def index():
    return render_template('a1.html')'''

@app.route('/')
def index():
    """Redirect to main application or login page."""
    session_token = request.cookies.get('session_token')
    if session_token and db.validate_session(session_token):
        return redirect(url_for('main_app'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Authenticate user
        success, user_id, message = db.authenticate_user(username, password)
        
        if success:
            # Create session
            session_token = db.create_session(user_id)
            
            # Get next page from query params or default to main app
            next_page = request.args.get('next', url_for('main_app'))
            
            # Set session cookie and redirect
            response = make_response(redirect(next_page))
            response.set_cookie('session_token', session_token, httponly=True, max_age=86400)  # 24 hours
            
            return response
        else:
            # Show error message
            return render_template('login.html', message=message, error=True)
    
    # For GET requests, show login form
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Check if passwords match
        if password != confirm_password:
            return render_template('register.html', message="Passwords do not match", error=True)
        
        # Register user
        success, message = db.register_user(username, email, password)
        
        if success:
            # Show success message and redirect to login
            return render_template('login.html', message="Registration successful. Please log in.", error=False)
        else:
            # Show error message
            return render_template('register.html', message=message, error=True)
    
    # For GET requests, show registration form
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    session_token = request.cookies.get('session_token')
    if session_token:
        db.delete_session(session_token)
    
    # Clear session cookie and redirect to login
    response = make_response(redirect(url_for('login')))
    response.delete_cookie('session_token')
    return response

@app.route('/app')
@auth_required
def main_app():
    """Main application page (protected by authentication)."""
    # This is your original main application route
    # Now it's protected by authentication
    return render_template('a1.html', username=request.user['username'])

@app.route('/process', methods=['POST'])
@auth_required
def process_request():
    """Process request from the frontend and send to model."""
    # Your existing processing logic
    # ...
    
    # You can now access the user information from request.user
    user_id = request.user['id']
    
    # Existing process logic would go here
    # ...
    
    return {"result": "Your processing result"}

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return 'No video part', 400

    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400
    
    if not allowed_file(file.filename):
        return 'File type not allowed', 400
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            print(f"Video saved to {filepath}")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model_name = "facebook/timesformer-base-finetuned-k400"
        config = TimesformerConfig.from_pretrained(model_name)
        
        num_classes = len(class_labels)
        config.num_labels = num_classes
        print(f"Using {num_classes} classes: {class_labels}")
        
        model = TimesformerForVideoClassification.from_pretrained(
            model_name, config=config, ignore_mismatched_sizes=True
        )
        
        model.classifier = nn.Linear(config.hidden_size, num_classes)
        
        checkpoint_path = "timesformer_epoch5.pth"
        try:
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model from checkpoint's model_state_dict")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model from checkpoint dictionary")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model from direct state dict")
                
        except Exception as e:
            print(f"Error loading model weights: {str(e)}")
            try:
                print("Attempting to load with strict=False...")
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)
                print("Loaded model with strict=False (some weights may be missing)")
            except Exception as e2:
                print(f"Emergency loading also failed: {str(e2)}")
                raise e

        model.to(device)
        model.eval()

        print("Preprocessing video...")
        video_tensor = preprocess_video(filepath)
        video_tensor = video_tensor.to(device)
        print(f"Video tensor shape: {video_tensor.shape}")

                with torch.no_grad():
            try:
                print("Running inference with video tensor shape:", video_tensor.shape)
                output = model(pixel_values=video_tensor)
                
                predicted_class_idx = torch.argmax(output.logits, dim=1).item()
                predicted_class = class_labels[predicted_class_idx]
                confidence = torch.softmax(output.logits, dim=1)[0][predicted_class_idx].item() * 100
                
                print(f"Predicted Class: {predicted_class}")
                print(f"Confidence: {confidence:.2f}%")
            except Exception as e:
                print(f"Error during inference: {str(e)}")
                
                try:
                    print("Trying alternative tensor format...")
                    alt_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # converting - [B, C, T, H, W] to [B, T, C, H, W]
                    print("Alternative tensor shape:", alt_tensor.shape)
                    
                    output = model(pixel_values=alt_tensor)
                    
                    predicted_class_idx = torch.argmax(output.logits, dim=1).item()
                    predicted_class = class_labels[predicted_class_idx]
                    confidence = torch.softmax(output.logits, dim=1)[0][predicted_class_idx].item() * 100
                    
                    print(f"Predicted Class (alternative format): {predicted_class}")
                    print(f"Confidence: {confidence:.2f}%")
                except Exception as e2:
                    print(f"Alternative format also failed: {str(e2)}")
                    raise e

            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")

        predicted_class_safe = html.escape(predicted_class)
        confidence_safe = f"{confidence:.2f}"
        text_color = "green" if predicted_class == "normal" else "red"

        return f"""
            <script>
            window.parent.document.getElementById('result').innerHTML = 
                '<div style="margin-top:20px; padding:15px; background-color:#f8f9fa; border-radius:5px;">' +
                '<h3>Predicted: <span style="color:{text_color};">{predicted_class_safe}</span></h3>' +
                '<p>Confidence: <strong>{confidence_safe}%</strong></p>' +
                '</div>';
            </script>
            <html>
                <head><title>Prediction Result</title></head>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h2>Predicted Scene:</h2>
                    <p style="font-size: 24px; color: #333;">
                        <strong>{predicted_class_safe}</strong> with confidence <strong>{confidence_safe}%</strong>
                    </p>
                    <a href="/">⬅ Back to Home</a>
                </body>
            </html>
        """

        except Exception as e:
        error_message = html.escape(str(e))
        return f"""
            <script>
            window.parent.document.getElementById('result').innerHTML = 
                '<div style="margin-top:20px; padding:15px; background-color:#ffebee; border-radius:5px;">' +
                '<h3>Error Processing Video</h3>' +
                '<p style="color:red;">{error_message}</p>' +
                '</div>';
            </script>
            <html>
                <head><title>Error</title></head>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h2>Error Processing Video</h2>
                    <p style="color: red;">{error_message}</p>
                    <a href="/">⬅ Back to Home</a>
                </body>
            </html>
        """

@app.route('/caption_progress', methods=['GET'])
def caption_progress():
    progress_file = os.path.join(UPLOAD_FOLDER, 'caption_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        return jsonify(progress_data)
    return jsonify({"status": "unknown", "progress": 0, "message": "Processing not started"})

@app.route('/caption', methods=['POST'])
def caption_video():
    if 'video' not in request.files:
        return 'No video part', 400

    file = request.files['video']
    if file.filename == '':
        return 'No selected file', 400

    if not allowed_file(file.filename):
        return 'File type not allowed', 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    print("Checkpoint 1: Video saved at", filepath)

    progress_file = os.path.join(UPLOAD_FOLDER, 'caption_progress.json')
    with open(progress_file, 'w') as f:
        json.dump({"status": "started", "progress": 0, "message": "Starting video processing..."}, f)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Checkpoint 2: Using device:", device)

        with open(progress_file, 'w') as f:
            json.dump({"status": "loading", "progress": 10, "message": "Loading BLIP model..."}, f)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print("Checkpoint 3: BLIP model loaded.")

        with open(progress_file, 'w') as f:
            json.dump({"status": "loading", "progress": 20, "message": "Loading T5 model..."}, f)
        t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
        print("Checkpoint 4: T5 model loaded.")

        with open(progress_file, 'w') as f:
            json.dump({"status": "processing", "progress": 30, "message": "Extracting video frames..."}, f)

        cap = cv2.VideoCapture(filepath)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(frame_count // 16, 1)
        frames = []
        for i in range(16):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
                frames.append(pil_img)

            progress = 30 + int((i / 16) * 30)
            with open(progress_file, 'w') as f:
                json.dump({"status": "processing", "progress": progress,
                           "message": f"Extracting frames... ({i+1}/16)"}, f)
        cap.release()
        print(f"Checkpoint 5: Extracted {len(frames)} frames.")

        with open(progress_file, 'w') as f:
            json.dump({"status": "processing", "progress": 60, "message": "Generating captions for frames..."}, f)

        captions = []
        for i, img in enumerate(frames):
            inputs = blip_processor(images=img, return_tensors="pt").to(device)
            output = blip_model.generate(**inputs)
            caption = blip_processor.decode(output[0], skip_special_tokens=True)
            captions.append(caption)

            progress = 60 + int((i / len(frames)) * 20)
            with open(progress_file, 'w') as f:
                json.dump({"status": "processing", "progress": progress,
                           "message": f"Captioning frame {i+1}/{len(frames)}..."}, f)
        print("Checkpoint 6: Captions generated for all frames.")

        combined_caption = ". ".join(captions) + "."

        print("Combined Captions:\n", combined_caption)

        with open(progress_file, 'w') as f:
            json.dump({"status": "finalizing", "progress": 80, "message": "Summarizing captions..."}, f)

        input_ids = t5_tokenizer.encode("summarize: " + combined_caption, return_tensors="pt", truncation=True, max_length=512).to(device)
        summary_ids = t5_model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
        final_caption = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print("Checkpoint 7: Final Caption:", final_caption)

        with open(progress_file, 'w') as f:
            json.dump({"status": "complete", "progress": 100, "message": "Caption generation complete!"}, f)

        final_caption_safe = json.dumps(final_caption)

        return f"""
            <script>
            window.parent.document.getElementById('result').innerHTML = 
                '<div style="margin-top:20px; padding:15px; background-color:#f8f9fa; border-radius:5px;">' +
                '<h3>Predicted:'+ {final_caption_safe} +'</h3>' +
                '</div>';
            </script>
            <html>
                <head><title>Prediction Result</title></head>
                <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                    <h2>Predicted Scene:</h2>
                    <p style="font-size: 24px; color: #333;">
                        <strong>{final_caption_safe}</strong> with confidence
                    </p>
                    <a href="/">⬅ Back to Home</a>
                </body>
            </html>
        """

    except Exception as e:
        error_message = html.escape(str(e))
        print("❌ Error:", error_message)
        with open(progress_file, 'w') as f:
            json.dump({"status": "error", "progress": 0, "message": f"Error: {error_message}"}, f)

        return f"""
        <script>
        window.parent.document.getElementById('result').innerHTML = 
            '<div style="margin-top:20px; padding:15px; background-color:#ffebee; border-radius:5px;">' +
            '<h3>Error Generating Caption</h3>' +
            '<p style="color:red;">{error_message}</p>' +
            '</div>';
        </script>
        <html>
            <head><title>Error</title></head>
            <body style="font-family: Arial; text-align: center; padding-top: 50px;">
                <h2>Error Generating Caption</h2>
                <p style="color: red;">{error_message}</p>
                <a href="/">⬅ Back to Home</a>
            </body>
        </html>
        """

if __name__ == '__main__':
    app.run(debug=True, port=5000)