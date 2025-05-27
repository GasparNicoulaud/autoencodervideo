#!/usr/bin/env python3
"""
Generate colorful checkerboard video samples
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import VideoAutoencoder
from src.utils import save_video, create_video_grid


def generate_colorful_checkerboard_video(frames=16, height=64, width=64, block_size=8):
    """Generate a colorful animated checkerboard video"""
    video = np.zeros((3, frames, height, width), dtype=np.float32)
    
    for t in range(frames):
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # Create animated colors
                r = np.sin(2 * np.pi * (i / height + t / frames)) * 0.5 + 0.5
                g = np.sin(2 * np.pi * (j / width + t / frames + 0.33)) * 0.5 + 0.5
                b = np.sin(2 * np.pi * ((i + j) / (height + width) + t / frames + 0.66)) * 0.5 + 0.5
                
                # Apply checkerboard pattern
                if ((i // block_size) + (j // block_size)) % 2 == 0:
                    video[0, t, i:i+block_size, j:j+block_size] = r
                    video[1, t, i:i+block_size, j:j+block_size] = g
                    video[2, t, i:i+block_size, j:j+block_size] = b
                else:
                    # Complementary colors
                    video[0, t, i:i+block_size, j:j+block_size] = 1 - r
                    video[1, t, i:i+block_size, j:j+block_size] = 1 - g
                    video[2, t, i:i+block_size, j:j+block_size] = 1 - b
    
    # Convert to [-1, 1] range
    video = video * 2 - 1
    return torch.from_numpy(video)


def generate_rainbow_noise_video(frames=16, height=64, width=64):
    """Generate colorful noise with structure"""
    video = torch.zeros(3, frames, height, width)
    
    for t in range(frames):
        # Create structured noise with color gradients
        x_grad = torch.linspace(0, 1, width).unsqueeze(0).repeat(height, 1)
        y_grad = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, width)
        
        # Animate the gradients
        phase = t / frames * 2 * np.pi
        
        # Red channel - horizontal gradient with noise
        video[0, t] = torch.sin(x_grad * 10 + phase) * 0.3 + torch.randn(height, width) * 0.1
        
        # Green channel - vertical gradient with noise
        video[1, t] = torch.sin(y_grad * 10 + phase + np.pi/3) * 0.3 + torch.randn(height, width) * 0.1
        
        # Blue channel - diagonal gradient with noise
        video[2, t] = torch.sin((x_grad + y_grad) * 7 + phase + 2*np.pi/3) * 0.3 + torch.randn(height, width) * 0.1
    
    return torch.tanh(video)  # Keep in [-1, 1] range


def main():
    parser = argparse.ArgumentParser(description='Generate colorful video samples')
    parser.add_argument('--output-dir', type=str, default='output/colorful_samples',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=9,
                       help='Number of samples to generate')
    parser.add_argument('--frames', type=int, default=16,
                       help='Number of frames')
    parser.add_argument('--size', type=int, default=64,
                       help='Frame size (height and width)')
    parser.add_argument('--pattern', type=str, default='checkerboard',
                       choices=['checkerboard', 'rainbow', 'mixed'],
                       help='Pattern type')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_samples} colorful {args.pattern} samples...")
    
    videos = []
    for i in range(args.num_samples):
        if args.pattern == 'checkerboard':
            # Vary block size for each sample
            block_size = 4 + (i % 4) * 4  # 4, 8, 12, 16
            video = generate_colorful_checkerboard_video(
                frames=args.frames, 
                height=args.size, 
                width=args.size,
                block_size=block_size
            )
        elif args.pattern == 'rainbow':
            video = generate_rainbow_noise_video(
                frames=args.frames,
                height=args.size,
                width=args.size
            )
        else:  # mixed
            if i % 2 == 0:
                block_size = 4 + (i % 4) * 4
                video = generate_colorful_checkerboard_video(
                    frames=args.frames, 
                    height=args.size, 
                    width=args.size,
                    block_size=block_size
                )
            else:
                video = generate_rainbow_noise_video(
                    frames=args.frames,
                    height=args.size,
                    width=args.size
                )
        
        videos.append(video)
        save_video(video.unsqueeze(0), f"{args.output_dir}/sample_{i:03d}.mp4")
    
    # Create grid
    grid = create_video_grid(videos[:9], grid_size=(3, 3))
    save_video(grid.unsqueeze(0), f"{args.output_dir}/colorful_grid.mp4")
    
    print(f"\nColorful samples saved to: {args.output_dir}")
    print(f"View the grid: open {args.output_dir}/colorful_grid.mp4")
    
    # Also generate with model to show latent space control
    print("\nGenerating model-based variations...")
    model = VideoAutoencoder(latent_dim=128, base_channels=32)
    model.eval()
    
    # Create a base colorful pattern
    base_video = generate_colorful_checkerboard_video(args.frames, args.size, args.size, 8)
    
    # Encode it
    with torch.no_grad():
        z, _, _ = model.encode(base_video.unsqueeze(0))
        
        # Generate variations
        variations = []
        for i in range(9):
            # Apply different modifications to latent
            z_mod = z.clone()
            
            if i == 0:
                pass  # Original
            elif i == 1:
                z_mod = z_mod * 1.5  # Scale up
            elif i == 2:
                z_mod = z_mod * 0.5  # Scale down
            elif i == 3:
                z_mod = z_mod + torch.randn_like(z) * 0.3  # Add noise
            elif i == 4:
                z_mod[:, :64] = -z_mod[:, :64]  # Invert first half
            elif i == 5:
                z_mod = torch.roll(z_mod, shifts=32, dims=1)  # Shift dimensions
            elif i == 6:
                z_mod = z_mod * torch.sin(torch.linspace(0, np.pi, z.shape[1]))  # Sine modulation
            elif i == 7:
                z_mod = torch.where(z_mod > 0, z_mod * 2, z_mod * 0.5)  # Nonlinear
            elif i == 8:
                z_mod = z_mod * (torch.rand_like(z) > 0.3).float()  # Sparse
            
            decoded = model.decode(z_mod)
            variations.append(decoded[0])
        
        # Save variations grid
        var_grid = create_video_grid(variations, grid_size=(3, 3))
        save_video(var_grid.unsqueeze(0), f"{args.output_dir}/latent_variations_grid.mp4")
    
    print(f"View latent variations: open {args.output_dir}/latent_variations_grid.mp4")


if __name__ == '__main__':
    main()