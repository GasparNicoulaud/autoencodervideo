import torch
import numpy as np
from typing import Union, Optional, Tuple, List
import imageio
from einops import rearrange

try:
    import cv2
except ImportError:
    cv2 = None


def load_video(path: str, 
               num_frames: Optional[int] = None,
               size: Optional[Tuple[int, int]] = None,
               normalize: bool = True) -> torch.Tensor:
    """
    Load a video and convert to tensor
    
    Args:
        path: Path to video file
        num_frames: Number of frames to load (None for all)
        size: (height, width) to resize frames
        normalize: Whether to normalize to [-1, 1]
        
    Returns:
        Video tensor of shape (C, T, H, W)
    """
    reader = imageio.get_reader(path)
    frames = []
    
    for i, frame in enumerate(reader):
        if num_frames and i >= num_frames:
            break
            
        if size:
            if cv2 is not None:
                frame = cv2.resize(frame, (size[1], size[0]))
            else:
                # Simple resize using PIL if cv2 not available
                from PIL import Image
                img = Image.fromarray(frame)
                img = img.resize((size[1], size[0]))
                frame = np.array(img)
            
        frames.append(frame)
        
    reader.close()
    
    video = np.array(frames)
    video = rearrange(video, 't h w c -> c t h w')
    video = torch.from_numpy(video).float()
    
    if normalize:
        video = video / 127.5 - 1.0
        
    return video


def save_video(tensor: torch.Tensor, 
               path: str,
               fps: int = 30,
               denormalize: bool = True):
    """
    Save a tensor as video
    
    Args:
        tensor: Video tensor of shape (C, T, H, W) or (B, C, T, H, W)
        path: Output path
        fps: Frames per second
        denormalize: Whether to denormalize from [-1, 1] to [0, 255]
    """
    if tensor.dim() == 5:
        tensor = tensor[0]
        
    if denormalize:
        tensor = (tensor + 1.0) * 127.5
        
    tensor = tensor.clamp(0, 255).byte()
    video = rearrange(tensor, 'c t h w -> t h w c').cpu().numpy()
    
    writer = imageio.get_writer(path, fps=fps)
    for frame in video:
        writer.append_data(frame)
    writer.close()


def create_video_grid(videos: List[torch.Tensor], 
                     grid_size: Optional[Tuple[int, int]] = None,
                     padding: int = 2) -> torch.Tensor:
    """
    Create a grid of videos
    
    Args:
        videos: List of video tensors
        grid_size: (rows, cols) for the grid
        padding: Padding between videos
        
    Returns:
        Grid tensor
    """
    n_videos = len(videos)
    
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_videos)))
        rows = int(np.ceil(n_videos / cols))
    else:
        rows, cols = grid_size
        
    c, t, h, w = videos[0].shape
    
    grid = torch.zeros(c, t, 
                      rows * h + (rows - 1) * padding,
                      cols * w + (cols - 1) * padding)
    
    for idx, video in enumerate(videos):
        if idx >= rows * cols:
            break
            
        row = idx // cols
        col = idx % cols
        
        y_start = row * (h + padding)
        y_end = y_start + h
        x_start = col * (w + padding)
        x_end = x_start + w
        
        grid[:, :, y_start:y_end, x_start:x_end] = video
        
    return grid


def frames_to_video(frames: torch.Tensor) -> torch.Tensor:
    """Convert from (T, C, H, W) to (C, T, H, W)"""
    return rearrange(frames, 't c h w -> c t h w')


def video_to_frames(video: torch.Tensor) -> torch.Tensor:
    """Convert from (C, T, H, W) to (T, C, H, W)"""
    return rearrange(video, 'c t h w -> t c h w')