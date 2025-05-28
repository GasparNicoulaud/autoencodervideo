#!/usr/bin/env python3
"""
Example: Process iPhone video with AnimateDiff support
"""
import torch
import numpy as np
import imageio
from PIL import Image
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


def prepare_video_for_vae(video_path, max_frames=16, target_size=512):
    """
    Prepare any video (including iPhone) for VAE processing
    
    Handles:
    - Different resolutions (1080p, 4K, etc.)
    - Different aspect ratios (16:9, 9:16, etc.)
    - Frame rate adjustment
    """
    print(f"\n1. Loading video: {video_path}")
    
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    
    print(f"   Original resolution: {meta.get('size', 'unknown')}")
    print(f"   FPS: {meta.get('fps', 30)}")
    
    # Sample frames evenly
    frames = []
    for i, frame in enumerate(reader):
        if i % 3 == 0:  # Take every 3rd frame to reduce data
            frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()
    
    print(f"   Sampled {len(frames)} frames")
    
    # Resize frames
    print(f"\n2. Resizing to {target_size}x{target_size}...")
    resized_frames = []
    
    for frame in frames:
        img = Image.fromarray(frame)
        
        # Center crop to square
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize to target
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        resized_frames.append(np.array(img))
    
    return np.array(resized_frames)


def main():
    # Example usage
    print("iPhone Video Processing Example")
    print("="*50)
    
    print("\n📱 iPhone Video Specs:")
    print("- Standard: 1920x1080 (1080p)")
    print("- 4K: 3840x2160")
    print("- Vertical: 1080x1920")
    
    print("\n🤖 Model Capabilities:")
    print("- Stable Diffusion VAE: Best at 512x512")
    print("- Can handle 256x256 to 768x768")
    print("- Larger = slower & more memory")
    
    print("\n💡 Tips for iPhone videos:")
    print("1. The script will automatically resize")
    print("2. Use 512x512 for best quality/speed")
    print("3. Reduce frames if running out of memory")
    
    print("\n🚀 To process your iPhone video:")
    print("python experiments/process_iphone_video.py --video your_video.mp4")
    
    print("\n📊 Resolution vs Performance:")
    print("256x256: Fast, low memory, ok quality")
    print("512x512: Balanced (recommended)")
    print("768x768: Slow, high memory, best quality")
    
    print("\n🎯 Advanced Models You Can Try:")
    print("\n1. AnimateDiff (better for motion):")
    print("   pip install animatediff")
    print("   - Handles temporal consistency better")
    print("   - ~4GB memory requirement")
    
    print("\n2. VideoMAE (lightweight):")
    print("   from transformers import VideoMAEModel")
    print("   - Smaller model, faster processing")
    print("   - Good for understanding content")
    
    print("\n3. FILM (frame interpolation):")
    print("   - Create smooth slow-motion")
    print("   - Works with any resolution")


if __name__ == '__main__':
    main()