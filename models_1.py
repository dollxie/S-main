import os
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
import google.generativeai as genai
import traceback

# Import from config.py
from config import (
    DEVICE, CLIP_MODEL_NAME, LABELS_MAP, IDX_TO_LABEL_MAP, NUM_CLASSES,
    FULL_CHECKPOINT_PATH, MODEL_TO_LOAD_FILENAME, MODEL_CHECKPOINT_DIR,
    GEMINI_MODEL_NAME, IMAGE_EXTENSIONS, FRAMES_TRAIN_DIR_FOR_LABELS,
    # Timesformer specific imports from config
    TIMESFORMER_MODEL_FILENAME, TIMESFORMER_LABELS_MAP,
    TIMESFORMER_IDX_TO_LABEL_MAP, TIMESFORMER_INPUT_NUM_FRAMES
)

# For Timesformer
from transformers import TimesformerConfig, TimesformerForVideoClassification
from torchvision import transforms as T # Alias to avoid conflict if other 'transforms' is used
from PIL import Image
import numpy as np

# --- Global Model Variables (initialized by loading functions) ---
clip_processor_global = None
clip_model_global = None  # Base CLIP model
custom_clip_classifier_model_global = None # Our CLIPClassifier instance
gemini_model = None
timesformer_model_global = None # For Timesformer

class DummyDatasetForLabels(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_extensions):
        self.root_dir = root_dir
        self.image_extensions = image_extensions
        self.label_map = {}
        self.idx_to_label = {}
        
        if not os.path.exists(root_dir):
            print(f"Warning: Label source directory '{root_dir}' not found. Using predefined or empty label map.")
            # Rely on LABELS_MAP from config if dir doesn't exist
            self.label_map = LABELS_MAP
            self.idx_to_label = IDX_TO_LABEL_MAP
            return
            
        for label_name in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_name)
            if os.path.isdir(label_path):
                if any(f.lower().endswith(self.image_extensions) for f in os.listdir(label_path)):
                    if label_name not in self.label_map:
                        new_idx = len(self.label_map)
                        self.label_map[label_name] = new_idx
                        self.idx_to_label[new_idx] = label_name
        
        # If after scanning, label_map is still empty, use config (safety)
        if not self.label_map:
            print(f"Warning: No labels derived from directory '{root_dir}'. Using predefined LABELS_MAP.")
            self.label_map = LABELS_MAP
            self.idx_to_label = IDX_TO_LABEL_MAP

    def __len__(self): return 0
    def __getitem__(self, idx): pass

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes, label_map_ref, clip_base_model):
        super().__init__()
        self.clip = clip_base_model
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(self.clip.config.projection_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        ).to(DEVICE)
        
        self.label_map = label_map_ref

    def get_label_name_from_idx(self, idx_tensor):
        idx = idx_tensor.item() if torch.is_tensor(idx_tensor) else idx_tensor
        return IDX_TO_LABEL_MAP.get(idx, "Unknown") # Use global IDX_TO_LABEL_MAP from config

    def forward(self, pixel_values):
        pixel_values = pixel_values.float().to(DEVICE)
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = self.classifier(image_features)
        return logits

def load_clip_classification_model():
    global clip_processor_global, clip_model_global, custom_clip_classifier_model_global
    try:
        print(f"\n=== Loading CLIP Model ===")
        print(f"Device: {DEVICE}")
        print(f"Checkpoint path: {FULL_CHECKPOINT_PATH}")
        print(f"Checkpoint exists: {os.path.exists(FULL_CHECKPOINT_PATH)}")
        
        print("Loading CLIP processor and base model...")
        clip_processor_global = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        clip_model_global = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        clip_model_global.eval()

        print(f"Initializing classifier for {NUM_CLASSES} classes")
        custom_clip_classifier_model_global = CLIPClassifier(NUM_CLASSES, LABELS_MAP, clip_model_global)

        print("Loading checkpoint...")
        checkpoint = torch.load(FULL_CHECKPOINT_PATH, map_location=DEVICE)
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            custom_clip_classifier_model_global.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded from model_state_dict")
        elif isinstance(checkpoint, dict) and 'classifier_state_dict' in checkpoint:
            custom_clip_classifier_model_global.classifier.load_state_dict(checkpoint['classifier_state_dict'])
            print("Loaded from classifier_state_dict")
        else:
            custom_clip_classifier_model_global.load_state_dict(checkpoint)
            print("Loaded from raw checkpoint")

        custom_clip_classifier_model_global.eval()
        print("CLIP model loaded successfully!")
        print("========================\n")

    except Exception as e:
        print(f"\n!!! ERROR loading CLIP model !!!")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("========================\n")
        custom_clip_classifier_model_global = None

def load_gemini_model():
    global gemini_model
    try:
        print("Initializing Gemini model...")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        if not google_api_key:
            print("ERROR: GOOGLE_API_KEY environment variable not set for Gemini.")
            raise ValueError("Google API Key not configured for Gemini")
        genai.configure(api_key=google_api_key)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print(f"Gemini model ({GEMINI_MODEL_NAME}) loaded successfully.")
    except Exception as e:
        print(f"CRITICAL: Failed to load Gemini model ({GEMINI_MODEL_NAME}): {str(e)}")
        traceback.print_exc()
        gemini_model = None

# Functions to get model instances (used by routes)
def get_clip_processor():
    return clip_processor_global

def get_custom_clip_classifier_model():
    return custom_clip_classifier_model_global

def get_gemini_model():
    return gemini_model

# --- Initialization call (optional, can be called from app.py) ---
# It might be better to call these explicitly from app.py after config is set up
# and before the app runs, to control the loading sequence.
# load_clip_classification_model()
# load_gemini_model()

# --- Timesformer Model Loading and Preprocessing ---

def load_timesformer_model():
    global timesformer_model_global
    try:
        print(f"\n=== Loading Timesformer Model ===")
        print(f"Device: {DEVICE}")
        
        timesformer_checkpoint_path = os.path.join(MODEL_CHECKPOINT_DIR, TIMESFORMER_MODEL_FILENAME)
        print(f"Checkpoint path: {timesformer_checkpoint_path}")
        print(f"Checkpoint exists: {os.path.exists(timesformer_checkpoint_path)}")
        
        print("Loading base Timesformer model...")
        base_timesformer_model_name = "facebook/timesformer-base-finetuned-k400"
        config = TimesformerConfig.from_pretrained(base_timesformer_model_name)
        
        num_timesformer_classes = len(TIMESFORMER_LABELS_MAP)
        config.num_labels = num_timesformer_classes
        print(f"Configured for {num_timesformer_classes} classes")
        
        timesformer_model_global = TimesformerForVideoClassification.from_pretrained(
            base_timesformer_model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
        
        print("Loading checkpoint...")
        checkpoint = torch.load(timesformer_checkpoint_path, map_location=DEVICE)
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print("Loading from model_state_dict...")
            timesformer_model_global.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Loading from raw checkpoint...")
            timesformer_model_global.load_state_dict(checkpoint)
            
        timesformer_model_global.eval()
        print("Timesformer model loaded successfully!")
        print("========================\n")

    except Exception as e:
        print(f"\n!!! ERROR loading Timesformer model !!!")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        print("========================\n")
        timesformer_model_global = None

def get_timesformer_model():
    return timesformer_model_global

def preprocess_frames_for_timesformer(pil_image_list: list, target_num_frames: int = TIMESFORMER_INPUT_NUM_FRAMES):
    """
    Preprocesses a list of PIL Image objects for the Timesformer model.
    Pads or samples frames to meet target_num_frames.
    Resizes, converts to tensor, and normalizes.
    Output shape: [1, C, T, H, W]
    """
    if not pil_image_list:
        raise ValueError("Input frame list is empty.")

    # 1. Pad or Sample frames
    num_available_frames = len(pil_image_list)
    processed_pil_images = []

    if num_available_frames == 0:
        # This case should ideally be handled before calling, but as a safeguard:
        # Create a black frame if no frames are available, to prevent crashes downstream.
        # This is a fallback and indicates an issue in the calling logic if reached.
        print("Warning: preprocess_frames_for_timesformer received an empty list. Using a black frame.")
        black_frame = Image.new('RGB', (224, 224), (0, 0, 0)) 
        processed_pil_images = [black_frame] * target_num_frames

    elif num_available_frames < target_num_frames:
        # Pad: repeat each frame, then fill with the last if needed
        base_repeats = target_num_frames // num_available_frames
        remainder = target_num_frames % num_available_frames
        
        for img in pil_image_list:
            processed_pil_images.extend([img] * base_repeats)
        
        # Add remaining frames by repeating the last frame from the original list
        if remainder > 0 and pil_image_list:
            last_frame = pil_image_list[-1]
            processed_pil_images.extend([last_frame] * remainder)
        # Ensure we don't exceed target_num_frames due to rounding/logic issues
        processed_pil_images = processed_pil_images[:target_num_frames]


    elif num_available_frames > target_num_frames:
        # Sample: uniformly select frames
        indices = np.linspace(0, num_available_frames - 1, target_num_frames).astype(int)
        processed_pil_images = [pil_image_list[i] for i in indices]
    else:
        # Exactly target_num_frames available
        processed_pil_images = pil_image_list

    # 2. Apply transformations (Resize, ToTensor)
    # Note: Normalization will be applied to the stacked tensor later
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor() # Converts to [C, H, W] and scales to [0, 1]
    ])
    
    frame_tensors = [transform(img) for img in processed_pil_images]
    
    # 3. Stack into a single tensor [T, C, H, W]
    video_tensor = torch.stack(frame_tensors) 
    
    # 4. Permute to [C, T, H, W] - Timesformer expects this channel-first for video
    video_tensor = video_tensor.permute(1, 0, 2, 3) 
    
    # 5. Normalize the entire video tensor (mean/std over all pixels in the video clip)
    # This matches the normalization from the user's provided working f1.py/preprocessing.py
    video_tensor = (video_tensor - video_tensor.mean()) / (video_tensor.std() + 1e-8)
    
    # 6. Add batch dimension: [1, C, T, H, W]
    video_tensor = video_tensor.unsqueeze(0)
    
    print(f"DEBUG: Timesformer preprocessed tensor final shape: {video_tensor.shape}") # DEBUG
    return video_tensor.to(DEVICE) 