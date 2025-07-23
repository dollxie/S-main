import torch
import numpy as np
from decord import VideoReader, cpu
from torchvision import transforms

def preprocess_video(video_path, num_frames=16):
    """
    Preprocess video for TimesformerForVideoClassification model
    
    Args:
        video_path: Path to the video file
        num_frames: Number of frames to sample from the video
        
    Returns:
        Tensor ready for input to Timesformer model
    """
    try:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            raise ValueError(f"Video file {video_path} contains no frames or could not be read properly.")
        
        if total_frames < num_frames:
            print(f"Warning: Video has only {total_frames} frames, padding will be applied")
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        
        frames = []
        for i in indices:
            frame = vr[i].asnumpy()
            frame_tensor = transform(frame)  
            frames.append(frame_tensor)
        
        video_tensor = torch.stack(frames)  # Shape: [T, C, H, W]
        
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
        
        video_tensor = (video_tensor - video_tensor.mean()) / (video_tensor.std() + 1e-8)
        
        video_tensor = video_tensor.unsqueeze(0)  # Shape: [1, C, T, H, W]
        
        print(f"Final tensor shape: {video_tensor.shape}")
        return video_tensor
        
    except Exception as e:
        print(f"Error preprocessing video: {str(e)}")
        raise