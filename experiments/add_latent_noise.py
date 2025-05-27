#!/usr/bin/env python3
"""
Simple experiment: Add noise to latent space
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL


def create_test_video(frames=16, height=256, width=256):
    """Create a simple test video with a moving circle"""
    video = np.zeros((frames, height, width, 3), dtype=np.float32)
    
    for t in range(frames):
        # Background gradient
        for i in range(height):
            for j in range(width):
                video[t, i, j, 0] = i / height * 0.3  # Red gradient
                video[t, i, j, 1] = j / width * 0.3   # Green gradient
                video[t, i, j, 2] = 0.2               # Blue constant
        
        # Moving circle
        center_x = width // 2 + int(60 * np.cos(t * 2 * np.pi / frames))
        center_y = height // 2 + int(60 * np.sin(t * 2 * np.pi / frames))
        radius = 40
        
        for i in range(max(0, center_y - radius), min(height, center_y + radius)):
            for j in range(max(0, center_x - radius), min(width, center_x + radius)):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < radius:
                    # White circle
                    video[t, i, j, :] = 1.0
    
    return video


def main():
    print("1. Creating test video...")
    video_np = create_test_video(frames=16, height=256, width=256)
    
    # Save original video
    Path("output").mkdir(exist_ok=True)
    imageio.mimsave('output/original.mp4', (video_np * 255).astype(np.uint8), fps=8)
    print("   Saved: output/original.mp4")
    
    # Convert to tensor format
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
    video_tensor = video_tensor * 2.0 - 1.0  # Convert to [-1, 1]
    
    print("\n2. Loading VAE...")
    device = "cpu"  # Use CPU for stability
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    print("\n3. Encoding video...")
    latents = []
    with torch.no_grad():
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t, :, :].unsqueeze(0)
            latent = vae.encode(frame).latent_dist.sample()
            latents.append(latent)
    
    print("\n4. Decoding without noise...")
    reconstructed_frames = []
    with torch.no_grad():
        for latent in latents:
            decoded = vae.decode(latent).sample[0]
            reconstructed_frames.append(decoded)
    
    # Save reconstructed video
    recon_video = torch.stack(reconstructed_frames, dim=1)
    recon_video = recon_video.permute(1, 2, 3, 0)  # (C,T,H,W) -> (T,H,W,C)
    recon_video = ((recon_video + 1.0) / 2.0).clamp(0, 1)
    recon_video_np = (recon_video * 255).byte().cpu().numpy()
    imageio.mimsave('output/reconstructed.mp4', recon_video_np, fps=8)
    print("   Saved: output/reconstructed.mp4")
    
    print("\n5. Adding noise to latent space and decoding...")
    noise_level = 5.0  # Adjust this to control noise amount
    noisy_frames = []
    
    with torch.no_grad():
        for latent in latents:
            # Add Gaussian noise to latent
            noise = torch.randn_like(latent) * noise_level
            noisy_latent = latent + noise
            
            # Decode noisy latent
            decoded = vae.decode(noisy_latent).sample[0]
            noisy_frames.append(decoded)
    
    # Save noisy video
    noisy_video = torch.stack(noisy_frames, dim=1)
    noisy_video = noisy_video.permute(1, 2, 3, 0)  # (C,T,H,W) -> (T,H,W,C)
    noisy_video = ((noisy_video + 1.0) / 2.0).clamp(0, 1)
    noisy_video_np = (noisy_video * 255).byte().cpu().numpy()
    imageio.mimsave('output/reconstructed_with_noise.mp4', noisy_video_np, fps=8)
    print("   Saved: output/reconstructed_with_noise.mp4")
    
    print("\n✅ Done! Check the output folder for:")
    print("   - original.mp4              (input video)")
    print("   - reconstructed.mp4         (VAE reconstruction)")
    print("   - reconstructed_with_noise.mp4 (VAE + latent noise)")
    print(f"\n   Noise level used: {noise_level}")
    print("   Latent shape per frame:", latents[0].shape)
    
    # Open the videos
    import subprocess
    try:
        subprocess.run(["open", "output/original.mp4"])
        subprocess.run(["open", "output/reconstructed.mp4"])
        subprocess.run(["open", "output/reconstructed_with_noise.mp4"])
    except:
        pass


if __name__ == '__main__':
    main()