#!/usr/bin/env python3
"""
Test with real Stable Diffusion VAE
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL
from src.utils import save_video, create_video_grid


def test_sd_vae():
    """Test Stable Diffusion VAE on video frames"""
    
    print("Loading Stable Diffusion VAE...")
    print("This will download ~335MB on first run\n")
    
    # Load the VAE
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32  # Use float32 for CPU/MPS
    ).to(device)
    vae.eval()
    
    print(f"✓ VAE loaded on {device}")
    print(f"  Latent channels: {vae.config.latent_channels}")
    print(f"  Downscale factor: {2 ** (len(vae.config.block_out_channels) - 1)}")
    
    # Create a simple test video
    print("\nCreating test video...")
    frames = 8
    height = width = 256  # Smaller for faster processing
    
    video_frames = []
    for t in range(frames):
        frame = torch.zeros(3, height, width)
        
        # Create animated pattern
        phase = t / frames * 2 * np.pi
        
        # Gradient background
        x = torch.linspace(-1, 1, width)
        y = torch.linspace(-1, 1, height)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        frame[0] = torch.sin(xx * 5 + phase) * 0.5
        frame[1] = torch.sin(yy * 5 + phase + np.pi/3) * 0.5
        frame[2] = torch.sin((xx + yy) * 3 + phase + 2*np.pi/3) * 0.5
        
        # Add moving circle
        center_x = width // 2 + int(30 * np.cos(phase))
        center_y = height // 2 + int(30 * np.sin(phase))
        
        for i in range(height):
            for j in range(width):
                dist = ((i - center_y)**2 + (j - center_x)**2)**0.5
                if dist < 20:
                    frame[:, i, j] = torch.tensor([1, 0.5, 0])
                    
        video_frames.append(frame)
    
    # Stack into video tensor
    video = torch.stack(video_frames, dim=0)  # (T, C, H, W)
    
    print(f"Video shape: {video.shape}")
    
    # Process each frame through VAE
    print("\nEncoding frames with real VAE...")
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
            
            print(f"  Frame {i+1}/{frames} - Latent shape: {latent.shape}")
    
    # Stack reconstructions
    recon_video = torch.stack(reconstructed_frames, dim=0)
    
    # Convert to (C, T, H, W) format for saving
    video = video.permute(1, 0, 2, 3).unsqueeze(0)
    recon_video = recon_video.permute(1, 0, 2, 3).unsqueeze(0)
    
    # Save results
    save_video(video, "output/sd_vae_original.mp4", denormalize=False)
    save_video(recon_video, "output/sd_vae_reconstruction.mp4", denormalize=False)
    
    print("\n✅ Success! Real VAE test complete.")
    print("\nGenerated:")
    print("- output/sd_vae_original.mp4")
    print("- output/sd_vae_reconstruction.mp4")
    
    # Test latent space manipulation
    print("\n\nTesting latent space manipulation...")
    test_latent_manipulation(vae, encoded_frames[0], device)
    
    return vae


def test_latent_manipulation(vae, base_latent, device):
    """Test various latent manipulations with real VAE"""
    
    manipulations = []
    
    # 1. Scale latent
    print("1. Testing latent scaling...")
    for scale in [0.0, 0.5, 1.0, 1.5, 2.0]:
        latent_scaled = base_latent * scale
        decoded = vae.decode(latent_scaled).sample[0].cpu()
        manipulations.append(decoded)
    
    # 2. Add noise
    print("2. Testing noise addition...")
    for noise_level in [0.1, 0.3, 0.5]:
        noise = torch.randn_like(base_latent) * noise_level
        latent_noisy = base_latent + noise.to(device)
        decoded = vae.decode(latent_noisy).sample[0].cpu()
        manipulations.append(decoded)
    
    # 3. Interpolate with random
    print("3. Testing interpolation...")
    random_latent = torch.randn_like(base_latent).to(device)
    for alpha in [0.25, 0.5, 0.75]:
        latent_interp = (1 - alpha) * base_latent + alpha * random_latent
        decoded = vae.decode(latent_interp).sample[0].cpu()
        manipulations.append(decoded)
    
    # 4. Channel manipulation
    print("4. Testing channel-specific manipulation...")
    for channel in range(4):  # VAE has 4 latent channels
        latent_mod = base_latent.clone()
        latent_mod[:, channel, :, :] *= 2.0
        decoded = vae.decode(latent_mod).sample[0].cpu()
        manipulations.append(decoded)
    
    # Create grid (take first 15 for 3x5 grid)
    grid_frames = []
    for img in manipulations[:15]:
        # Add time dimension
        img_video = img.unsqueeze(1).repeat(1, 8, 1, 1)  # Repeat for 8 frames
        grid_frames.append(img_video)
    
    grid = create_video_grid(grid_frames, grid_size=(3, 5))
    save_video(grid.unsqueeze(0), "output/sd_vae_manipulations.mp4", denormalize=False)
    
    print("\n✅ Latent manipulations saved to: output/sd_vae_manipulations.mp4")
    print("\nShows effects of:")
    print("- Scaling (0x to 2x)")
    print("- Adding noise")
    print("- Interpolation with random latent")
    print("- Channel-specific amplification")


if __name__ == '__main__':
    vae = test_sd_vae()
    
    print("\n" + "="*60)
    print("REAL VAE SUCCESSFULLY TESTED!")
    print("="*60)
    print("\nKey differences from our toy model:")
    print("- Real VAE compresses 256x256 → 32x32 latents (8x reduction)")
    print("- 4 latent channels (not arbitrary latent_dim)")
    print("- Trained on millions of real images")
    print("- Much better reconstruction quality")
    print("\nYou can now:")
    print("1. Use this VAE on real videos")
    print("2. Apply all the latent manipulation techniques")
    print("3. Integrate with video generation models")
    
    # Open the results
    import subprocess
    try:
        subprocess.run(["open", "output/sd_vae_reconstruction.mp4"])
        subprocess.run(["open", "output/sd_vae_manipulations.mp4"])
    except:
        pass