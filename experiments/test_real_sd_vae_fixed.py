#!/usr/bin/env python3
"""
Test with real Stable Diffusion VAE - Fixed version
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL
from src.utils import save_video, create_video_grid


def test_sd_vae_fixed():
    """Test Stable Diffusion VAE with proper input normalization"""
    
    print("Loading Stable Diffusion VAE...")
    
    # Load the VAE
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    print(f"✓ VAE loaded on {device}")
    
    # Create a simple test video with PROPER RANGE
    print("\nCreating test video with proper normalization...")
    frames = 8
    height = width = 256
    
    video_frames = []
    for t in range(frames):
        frame = torch.zeros(3, height, width)
        
        # Create animated pattern
        phase = t / frames * 2 * np.pi
        
        # Create a colorful animated pattern
        for i in range(height):
            for j in range(width):
                # Rainbow effect
                r = 0.5 + 0.5 * np.sin(2 * np.pi * (i / height + phase))
                g = 0.5 + 0.5 * np.sin(2 * np.pi * (j / width + phase + np.pi/3))
                b = 0.5 + 0.5 * np.sin(2 * np.pi * ((i+j) / (height+width) + phase + 2*np.pi/3))
                
                frame[0, i, j] = r
                frame[1, i, j] = g
                frame[2, i, j] = b
        
        # Add a white moving square
        square_size = 30
        center_x = width // 2 + int(50 * np.cos(phase))
        center_y = height // 2 + int(50 * np.sin(phase))
        
        x_start = max(0, center_x - square_size // 2)
        x_end = min(width, center_x + square_size // 2)
        y_start = max(0, center_y - square_size // 2)
        y_end = min(height, center_y + square_size // 2)
        
        frame[:, y_start:y_end, x_start:x_end] = 1.0  # White square
        
        # CRITICAL: Convert from [0, 1] to [-1, 1] for VAE
        frame = (frame * 2.0) - 1.0
        
        video_frames.append(frame)
    
    # Stack into video tensor
    video = torch.stack(video_frames, dim=0)  # (T, C, H, W)
    
    print(f"Video shape: {video.shape}")
    print(f"Value range: [{video.min():.2f}, {video.max():.2f}] (should be ~[-1, 1])")
    
    # Process each frame through VAE
    print("\nEncoding frames with VAE...")
    encoded_frames = []
    reconstructed_frames = []
    
    with torch.no_grad():
        for i, frame in enumerate(video):
            # Add batch dimension and move to device
            frame_batch = frame.unsqueeze(0).to(device)
            
            # Encode
            latent = vae.encode(frame_batch).latent_dist.sample()
            encoded_frames.append(latent)
            
            # Decode
            recon = vae.decode(latent).sample
            reconstructed_frames.append(recon[0].cpu())
            
            if i == 0:
                print(f"  First frame latent shape: {latent.shape}")
                print(f"  Latent value range: [{latent.min():.2f}, {latent.max():.2f}]")
                print(f"  Reconstructed range: [{recon.min():.2f}, {recon.max():.2f}]")
    
    # Stack reconstructions
    recon_video = torch.stack(reconstructed_frames, dim=0)
    
    # Convert to (C, T, H, W) format for saving
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    recon_video = recon_video.permute(1, 0, 2, 3).unsqueeze(0)
    
    # Save results - the save_video function expects [-1, 1] range
    save_video(video, "output/sd_vae_original_fixed.mp4", denormalize=True)
    save_video(recon_video, "output/sd_vae_reconstruction_fixed.mp4", denormalize=True)
    
    print("\n✅ Fixed version complete!")
    print("\nGenerated:")
    print("- output/sd_vae_original_fixed.mp4")
    print("- output/sd_vae_reconstruction_fixed.mp4")
    
    # Test latent space manipulation with visible results
    print("\n\nTesting latent manipulations with visible effects...")
    test_visible_manipulations(vae, encoded_frames, device)
    
    return vae


def test_visible_manipulations(vae, encoded_frames, device):
    """Test manipulations that produce visible results"""
    
    # Use the middle frame as base
    base_latent = encoded_frames[4]
    
    manipulations = []
    labels = []
    
    # Original
    with torch.no_grad():
        original = vae.decode(base_latent).sample[0].cpu()
        manipulations.append(original)
        labels.append("Original")
    
    # 1. Interpolate between two different frames
    print("1. Frame interpolation...")
    other_latent = encoded_frames[0]
    for i, alpha in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        latent_interp = (1 - alpha) * base_latent + alpha * other_latent
        decoded = vae.decode(latent_interp).sample[0].cpu()
        manipulations.append(decoded)
        labels.append(f"Interp {alpha:.2f}")
    
    # 2. Add structured noise
    print("2. Adding structured patterns to latent...")
    for pattern in ['gradient', 'checkerboard', 'radial']:
        latent_mod = base_latent.clone()
        h, w = latent_mod.shape[2], latent_mod.shape[3]
        
        if pattern == 'gradient':
            # Add gradient to latent
            grad = torch.linspace(-0.5, 0.5, h).unsqueeze(1).repeat(1, w)
            latent_mod[:, 0, :, :] += grad.to(device)
        elif pattern == 'checkerboard':
            # Add checkerboard pattern
            checker = torch.zeros(h, w)
            for i in range(0, h, 4):
                for j in range(0, w, 4):
                    if ((i//4) + (j//4)) % 2 == 0:
                        checker[i:i+4, j:j+4] = 0.3
            latent_mod[:, 1, :, :] += checker.to(device)
        elif pattern == 'radial':
            # Add radial pattern
            center_h, center_w = h // 2, w // 2
            for i in range(h):
                for j in range(w):
                    dist = ((i - center_h)**2 + (j - center_w)**2)**0.5
                    latent_mod[:, 2, i, j] += 0.1 * np.sin(dist)
        
        decoded = vae.decode(latent_mod).sample[0].cpu()
        manipulations.append(decoded)
        labels.append(f"Add {pattern}")
    
    # 3. Extreme manipulations
    print("3. Extreme manipulations...")
    
    # Zero out different channels
    for channel in range(4):
        latent_zero = base_latent.clone()
        latent_zero[:, channel, :, :] = 0
        decoded = vae.decode(latent_zero).sample[0].cpu()
        manipulations.append(decoded)
        labels.append(f"Zero ch{channel}")
    
    # Create a grid with first 15 manipulations
    print("\nCreating visualization grid...")
    
    # Convert to video format (add time dimension)
    grid_frames = []
    for img in manipulations[:15]:
        # Repeat frame for 8 timesteps
        img_video = img.unsqueeze(1).repeat(1, 8, 1, 1)
        grid_frames.append(img_video)
    
    grid = create_video_grid(grid_frames, grid_size=(3, 5))
    save_video(grid.unsqueeze(0), "output/sd_vae_manipulations_fixed.mp4", denormalize=True)
    
    # Also save as individual images to debug
    print("\nSaving first few as images for debugging...")
    import torchvision
    for i in range(min(5, len(manipulations))):
        # Convert from [-1, 1] to [0, 1] for saving
        img = (manipulations[i] + 1.0) / 2.0
        img = torch.clamp(img, 0, 1)
        torchvision.utils.save_image(
            img, 
            f"output/debug_manipulation_{i}_{labels[i].replace(' ', '_')}.png"
        )
    
    print("\n✅ Manipulations saved!")
    print("- output/sd_vae_manipulations_fixed.mp4")
    print("- output/debug_manipulation_*.png (for debugging)")


if __name__ == '__main__':
    vae = test_sd_vae_fixed()
    
    print("\n" + "="*60)
    print("DEBUGGING TIPS:")
    print("="*60)
    print("\nIf videos are still black, check:")
    print("1. The PNG images in output/ - are they black too?")
    print("2. Value ranges printed above - should be ~[-1, 1]")
    print("3. Try opening in different video players")
    print("\nThe VAE is working if:")
    print("- Latent shapes are correct (1, 4, 32, 32)")
    print("- Value ranges are reasonable")
    print("- PNG images show content")
    
    # Try to open results
    import subprocess
    try:
        subprocess.run(["open", "output/sd_vae_original_fixed.mp4"])
        subprocess.run(["open", "output/debug_manipulation_0_Original.png"])
    except:
        pass