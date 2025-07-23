import os
import torch

# Flask App Config
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # Max upload size = 200 MB
UPLOAD_FOLDER = 'uploads'
SECRET_KEY = os.urandom(24) # For Flask app.secret_key

# --- Configuration for CLIP Model ---
MAIN_PROJECT_ROOT = "." 
MODEL_CHECKPOINT_DIR = os.path.join(MAIN_PROJECT_ROOT, "model_checkpoints")  # Updated to use model_checkpoints directory
MODEL_TO_LOAD_FILENAME = "best_model_epoch_18.pth"
FULL_CHECKPOINT_PATH = os.path.join(MODEL_CHECKPOINT_DIR, MODEL_TO_LOAD_FILENAME)

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Inference Parameters for CLIP ---
FRAME_INTERVAL_FOR_INFERENCE = 10

# --- Display/Save Frame Parameters ---
SAVE_INFERRED_FRAMES = True 
INFERRED_FRAMES_OUTPUT_DIR = os.path.join(MAIN_PROJECT_ROOT, "inferred_video_frames_output")
MIN_CONFIDENCE_THRESHOLD = 0.35 

# --- CLIP Model Architecture Setup ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# --- Label Setup for CLIP ---
# These are hardcoded based on the training of 'best_model_epoch_18.pth'
LABELS_MAP = {
    'Abuse': 0, 'Arrest': 1, 'Arson': 2, 'Assault': 3, 'Burglary': 4, 
    'Explosion': 5, 'Fighting': 6, 'RoadAccidents': 7, 'Robbery': 8, 'Shooting': 9,
    'Shoplifting': 10, 'Stealing': 11, 'Vandalism': 12, 'normal': 13
}
IDX_TO_LABEL_MAP = {v: k for k, v in LABELS_MAP.items()}
NUM_CLASSES = len(LABELS_MAP)

# --- Paths for DummyDatasetForLabels (if ever needed, though labels are hardcoded now) ---
EXTRACTED_FRAMES_DIR_FOR_LABELS = os.path.join(MAIN_PROJECT_ROOT, "extracted_dataset_frames_for_labels_temp")
FRAMES_TRAIN_DIR_FOR_LABELS = os.path.join(EXTRACTED_FRAMES_DIR_FOR_LABELS, 'train_frames')
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# --- Gemini Model --- 
# GOOGLE_API_KEY is loaded from .env in the main app.py, so not stored here directly.
GEMINI_MODEL_NAME = "gemini-2.0-flash" # Corrected model name from "gemini-pro-vision"

# --- Video Processing --- 
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
NUM_FRAMES_FOR_CAPTIONING = 5
N_REPRESENTATIVE_FRAMES_PER_SEGMENT = 5 # New config for segment frame display

# --- New Generalized Event Segment Detection ---
# Maximum time in seconds allowed between two consecutive significant events 
# for them to be considered part of the same "chain" or "burst" of activity.
SEGMENT_MAX_GAP_SECONDS_BETWEEN_SIG_EVENTS_IN_CHAIN = 10.0 
# The minimum number of significant events a chain must have to form the basis of a segment.
SEGMENT_MIN_SIG_EVENTS_IN_CHAIN = 2 
# The maximum allowed percentage of normal (non-significant) frames within the 
# full time span of a potential segment (defined by a chain's first and last significant event).
# If a segment span exceeds this percentage of normal frames, it won't be created.
SEGMENT_MAX_NORMAL_FRAMES_PERCENTAGE_IN_SPAN = 0.75 # (e.g., 0.75 means max 75% normal frames)

# --- Timesformer Model Specific Config ---
TIMESFORMER_MODEL_FILENAME = "timesformer_epoch5.pth"
TIMESFORMER_LABELS_MAP = {
    'Abuse': 0, 'Arrest': 1, 'Arson': 2, 'Assault': 3,
    'Burglary': 4, 'Explosion': 5, 'Fighting': 6, 'normal': 7
}
TIMESFORMER_IDX_TO_LABEL_MAP = {v: k for k, v in TIMESFORMER_LABELS_MAP.items()}
TIMESFORMER_INPUT_NUM_FRAMES = 16

# Ensure necessary directories exist
def create_directories():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(INFERRED_FRAMES_OUTPUT_DIR, exist_ok=True)
    # If EXTRACTED_FRAMES_DIR_FOR_LABELS was actually used for dynamic label generation:
    # os.makedirs(EXTRACTED_FRAMES_DIR_FOR_LABELS, exist_ok=True)
    # os.makedirs(FRAMES_TRAIN_DIR_FOR_LABELS, exist_ok=True) 