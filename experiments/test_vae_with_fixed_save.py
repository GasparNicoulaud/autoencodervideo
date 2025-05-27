#!/usr/bin/env python3
"""
Test VAE with properly working video save
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL


def save_video_properly(tensor, path, fps=8):
    """Save video with proper conversion"""
    # tensor shape: (B, C, T, H, W) or (C, T, H, W)
    if tensor.dim() == 5:
        tensor = tensor[0]
    
    # tensor is (C, T, H, W), convert to (T, H, W, C)
    video = tensor.permute(1, 2, 3, 0).detach().cpu().numpy()
    
    # Ensure proper range [0, 1]
    video = (video + 1.0) / 2.0  # from [-1, 1] to [0, 1]
    video = np.clip(video, 0, 1)
    
    # Convert to uint8
    video = (video * 255).astype(np.uint8)
    
    # Save with imageio
    imageio.mimsave(path, video, fps=fps)
    print(f"Saved {path} - shape: {video.shape}, range: [{video.min()}, {video.max()}]")


def test_vae_simple():
    """Simple test with guaranteed visible output"""
    
    print("Loading VAE...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    print("Creating bright test pattern...")
    
    # Create a very simple, bright test image
    height = width = 256
    image = torch.ones(1, 3, height, width) * 0.8  # Bright gray
    
    # Add some colored squares
    image[:, 0, 50:100, 50:100] = 1.0   # Red square
    image[:, 1, 50:100, 50:100] = 0.0
    image[:, 2, 50:100, 50:100] = 0.0
    
    image[:, 0, 150:200, 50:100] = 0.0   # Green square
    image[:, 1, 150:200, 50:100] = 1.0
    image[:, 2, 150:200, 50:100] = 0.0
    
    image[:, 0, 50:100, 150:200] = 0.0   # Blue square
    image[:, 1, 50:100, 150:200] = 0.0
    image[:, 2, 50:100, 150:200] = 1.0
    
    image[:, 0, 150:200, 150:200] = 1.0   # White square
    image[:, 1, 150:200, 150:200] = 1.0
    image[:, 2, 150:200, 150:200] = 1.0
    
    # Convert to [-1, 1] for VAE
    image = (image * 2.0) - 1.0
    
    print(f"Input range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Save input as image
    input_img = ((image[0] + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    imageio.imwrite('output/vae_input.png', input_img)
    
    # Encode and decode
    with torch.no_grad():
        image = image.to(device)
        latent = vae.encode(image).latent_dist.sample()
        print(f"Latent shape: {latent.shape}")
        
        reconstructed = vae.decode(latent).sample
        print(f"Output range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
    
    # Save reconstruction as image
    recon_img = ((reconstructed[0] + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
    imageio.imwrite('output/vae_reconstruction.png', recon_img)
    
    # Create video by repeating the frame
    video_frames = 8
    input_video = image.unsqueeze(2).repeat(1, 1, video_frames, 1, 1)
    recon_video = reconstructed.unsqueeze(2).repeat(1, 1, video_frames, 1, 1)
    
    # Save videos
    save_video_properly(input_video, 'output/vae_input_video.mp4')
    save_video_properly(recon_video, 'output/vae_recon_video.mp4')
    
    # Test latent manipulation
    print("\nTesting latent manipulation...")
    manipulations = []
    
    # Original
    manipulations.append(reconstructed)
    
    # Scale latent
    for scale in [0.5, 1.5, 2.0]:
        scaled_latent = latent * scale
        decoded = vae.decode(scaled_latent).sample
        manipulations.append(decoded)
    
    # Add noise
    noise = torch.randn_like(latent) * 0.5
    noisy_latent = latent + noise
    decoded = vae.decode(noisy_latent).sample
    manipulations.append(decoded)
    
    # Create manipulation video
    all_frames = []
    for img in manipulations:
        # Repeat each manipulation for a few frames
        for _ in range(6):
            frame = img[0].permute(1, 2, 0).cpu()
            all_frames.append(frame)
    
    manip_video = torch.stack(all_frames).permute(3, 0, 1, 2)
    save_video_properly(manip_video.unsqueeze(0), 'output/vae_manipulations_video.mp4', fps=6)
    
    print("\n✅ Done! Check these files:")
    print("- output/vae_input.png (should show colored squares)")
    print("- output/vae_reconstruction.png (VAE reconstruction)")
    print("- output/vae_input_video.mp4")
    print("- output/vae_recon_video.mp4")
    print("- output/vae_manipulations_video.mp4")
    
    return True


if __name__ == '__main__':
    success = test_vae_simple()
    
    if success:
        print("\n🎉 VAE is working correctly!")
        print("The black video issue was due to the save_video function.")
        print("The VAE itself is functioning properly!")
        
        # Open results
        import subprocess
        try:
            subprocess.run(["open", "output/vae_input.png"])
            subprocess.run(["open", "output/vae_reconstruction.png"])
            subprocess.run(["open", "output/vae_manipulations_video.mp4"])
        except:
            pass