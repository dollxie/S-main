import os
from glob import glob
from config import ALLOWED_EXTENSIONS, UPLOAD_FOLDER, INFERRED_FRAMES_OUTPUT_DIR
import shutil
from datetime import datetime, timedelta
import json

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_latest_uploaded_video():
    # Consider all allowed extensions, case-insensitively
    patterns = [os.path.join(UPLOAD_FOLDER, f'*.{ext}') for ext in ALLOWED_EXTENSIONS]
    files = []
    for pattern in patterns:
        # Use glob to find files matching each extension pattern
        # This approach is more robust if ALLOWED_EXTENSIONS changes
        files.extend(glob(pattern, recursive=False)) # Add case_insensitive=True if Python 3.11+ and needed
        # For older Python or stricter case matching, you might need to glob for both lower and upper case if OS is case-sensitive.
        # Example for case-insensitivity on case-sensitive systems (more complex):
        # files.extend([f for f_pattern in [f'*.{e.lower()}', f'*.{e.upper()}'] for f in glob(os.path.join(UPLOAD_FOLDER, f_pattern))]) 

    video_files = [f for f in files if os.path.isfile(f)]
    if not video_files:
        return None
    return max(video_files, key=os.path.getmtime) 

def cleanup_old_files(max_age_hours=24):
    """
    Clean up old video files and their associated results.
    Files older than max_age_hours will be deleted.
    """
    current_time = datetime.now()
    max_age = timedelta(hours=max_age_hours)
    
    # Clean up video files
    for filename in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getctime(filepath))
            if current_time - file_time > max_age:
                try:
                    os.remove(filepath)
                    print(f"Deleted old video file: {filename}")
                    
                    # Also clean up associated frames
                    video_name = os.path.splitext(filename)[0]
                    frames_dir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, video_name)
                    if os.path.exists(frames_dir):
                        shutil.rmtree(frames_dir)
                        print(f"Deleted frames directory for: {video_name}")
                except Exception as e:
                    print(f"Error cleaning up {filename}: {str(e)}")

def cleanup_after_processing(video_filename):
    """
    Clean up a specific video file and its results after processing is complete.
    """
    try:
        # Remove the video file
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"Deleted processed video: {video_filename}")
        
        # Remove associated frames
        video_name = os.path.splitext(video_filename)[0]
        frames_dir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, video_name)
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
            print(f"Deleted frames directory for: {video_name}")
            
    except Exception as e:
        print(f"Error cleaning up {video_filename}: {str(e)}")

def ensure_clean_workspace():
    """
    Ensure the workspace is clean before processing new videos.
    """
    # Clean up any files older than 24 hours
    cleanup_old_files(max_age_hours=24)
    
    # Create necessary directories if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(INFERRED_FRAMES_OUTPUT_DIR, exist_ok=True)

def store_suspicious_detections(video_name, detections):
    """
    Store suspicious object detections with timestamps for a video.
    detections should be a list of dicts with format:
    {
        'timestamp': 'MM:SS.ms',
        'objects': ['object1', 'object2'],
        'frame_idx': 123
    }
    """
    try:
        # Create a directory for video metadata if it doesn't exist
        metadata_dir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Create/update the detections file for this video
        detections_file = os.path.join(metadata_dir, f'{video_name}_detections.json')
        
        # Load existing detections if any
        existing_detections = []
        if os.path.exists(detections_file):
            with open(detections_file, 'r') as f:
                existing_detections = json.load(f)
        
        # Add new detections
        existing_detections.extend(detections)
        
        # Sort by timestamp
        existing_detections.sort(key=lambda x: x['timestamp'])
        
        # Save updated detections
        with open(detections_file, 'w') as f:
            json.dump(existing_detections, f, indent=2)
            
        print(f"Stored {len(detections)} suspicious detections for {video_name}")
        
    except Exception as e:
        print(f"Error storing suspicious detections for {video_name}: {str(e)}")

def get_suspicious_detections(video_name):
    """
    Retrieve stored suspicious object detections for a video.
    """
    try:
        detections_file = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, 'metadata', f'{video_name}_detections.json')
        if os.path.exists(detections_file):
            with open(detections_file, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        print(f"Error retrieving suspicious detections for {video_name}: {str(e)}")
        return [] 