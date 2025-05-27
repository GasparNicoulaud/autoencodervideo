#!/usr/bin/env python3
"""
Debug why videos appear black
"""
import torch
import numpy as np
import imageio
import cv2

# Create a simple test pattern
print("Creating test pattern...")
frames = []
for t in range(8):
    frame = np.zeros((256, 256, 3), dtype=np.float32)
    
    # Create colorful pattern
    for i in range(256):
        for j in range(256):
            frame[i, j, 0] = (i / 255.0)  # Red gradient
            frame[j, i, 1] = (j / 255.0)  # Green gradient  
            frame[i, j, 2] = 0.5 + 0.5 * np.sin(t * np.pi / 4)  # Blue pulse
    
    # Add white square
    frame[100:150, 100:150, :] = 1.0
    
    frames.append(frame)

print(f"Frame value range: [{frames[0].min():.2f}, {frames[0].max():.2f}]")

# Test 1: Save with imageio directly (0-255 range)
print("\nTest 1: imageio with uint8...")
frames_uint8 = [(frame * 255).astype(np.uint8) for frame in frames]
imageio.mimsave('output/test_imageio_uint8.mp4', frames_uint8, fps=4)

# Test 2: Save with imageio float (0-1 range)
print("Test 2: imageio with float...")
imageio.mimsave('output/test_imageio_float.mp4', frames, fps=4)

# Test 3: Save individual frames as PNG
print("Test 3: Saving individual frames...")
for i, frame in enumerate(frames[:3]):
    frame_uint8 = (frame * 255).astype(np.uint8)
    imageio.imwrite(f'output/test_frame_{i}.png', frame_uint8)

# Test 4: Using our save_video function
print("\nTest 4: Using save_video function...")
from src.utils import save_video

# Convert to torch tensor in correct format
video_tensor = torch.from_numpy(np.array(frames)).permute(3, 0, 1, 2).float()
print(f"Tensor shape: {video_tensor.shape}")
print(f"Tensor range: [{video_tensor.min():.2f}, {video_tensor.max():.2f}]")

# Convert to [-1, 1] range
video_tensor_normalized = video_tensor * 2 - 1
save_video(video_tensor_normalized.unsqueeze(0), 'output/test_save_video.mp4', denormalize=True)

print("\nCreated test videos:")
print("- output/test_imageio_uint8.mp4")
print("- output/test_imageio_float.mp4") 
print("- output/test_save_video.mp4")
print("- output/test_frame_*.png")
print("\nCheck which ones display correctly!")

# Also create a GIF for comparison
print("\nBonus: Creating GIF...")
imageio.mimsave('output/test_pattern.gif', frames_uint8, fps=4)
print("- output/test_pattern.gif")