import torch
from transformers import TimesformerForVideoClassification, TimesformerConfig

try:
    print(f"Torch version: {torch.__version__}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu") # Force CPU for simplicity in test
    print(f"Using device: {device}")

    model_name = "facebook/timesformer-base-finetuned-k400"
    
    # For the most basic test, let's use the *exact* config from the pretrained model first.
    # The k400 model was pretrained with num_frames=8.
    config = TimesformerConfig.from_pretrained(model_name)
    
    print(f"Initializing model: {model_name} with its original config.")
    print(f"Original config - num_channels: {config.num_channels}, num_frames: {config.num_frames}, patch_size: {config.patch_size}, image_size: {config.image_size}")

    model = TimesformerForVideoClassification.from_pretrained(
        model_name,
        config=config, # Use the original config for this test
        ignore_mismatched_sizes=False # Set to False for stricter checking with original config
    ).to(device)
    model.eval()
    print("Model loaded successfully.")

    # Create a dummy input tensor
    # Standard input: (batch_size, num_channels, num_frames, height, width)
    batch_size = 1
    num_channels = 3 # Should be config.num_channels
    num_frames_test = config.num_frames # Use original config.num_frames (e.g., 8 for k400)
    height = config.image_size # Use original config.image_size (e.g., 224)
    width = config.image_size  # Use original config.image_size

    dummy_input = torch.randn(batch_size, num_channels, num_frames_test, height, width).to(device)
    print(f"Dummy input tensor shape: {dummy_input.shape}")

    with torch.no_grad():
        outputs = model(pixel_values=dummy_input)
        logits = outputs.logits
    print(f"Model output logits shape: {logits.shape}") # Should be [batch_size, 400] for k400
    print("Minimal test script completed successfully!")

except Exception as e:
    print(f"Error in minimal test script: {e}")
    import traceback
    traceback.print_exc() 