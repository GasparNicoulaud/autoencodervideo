#!/usr/bin/env python3
"""
Simple script to generate random samples without requiring video files
"""
import torch
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import VideoAutoencoder
from src.latent import LatentManipulator
from src.utils import save_video, create_video_grid


def generate_random_samples(model, num_samples=4, device='cpu'):
    """Generate random video samples"""
    # Random latent codes
    z = torch.randn(num_samples, model.latent_dim).to(device)
    
    with torch.no_grad():
        videos = model.decode(z)
    
    return videos


def main():
    parser = argparse.ArgumentParser(description='Generate random video samples')
    parser.add_argument('--output-dir', type=str, default='output/samples',
                       help='Output directory')
    parser.add_argument('--num-samples', type=int, default=9,
                       help='Number of samples to generate')
    parser.add_argument('--latent-dim', type=int, default=256,
                       help='Latent dimension')
    parser.add_argument('--frames', type=int, default=16,
                       help='Number of frames')
    parser.add_argument('--size', type=int, default=64,
                       help='Frame size (height and width)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Initializing model...")
    model = VideoAutoencoder(
        latent_dim=args.latent_dim, 
        base_channels=32  # Smaller for faster generation
    )
    model.to(args.device)
    model.eval()
    
    print(f"Generating {args.num_samples} random samples...")
    videos = generate_random_samples(model, args.num_samples, args.device)
    
    # Save individual videos
    for i in range(args.num_samples):
        save_video(videos[i:i+1], f"{args.output_dir}/sample_{i:03d}.mp4")
    
    # Create grid
    grid_videos = [videos[i] for i in range(min(9, args.num_samples))]
    grid = create_video_grid(grid_videos, grid_size=(3, 3))
    save_video(grid.unsqueeze(0), f"{args.output_dir}/samples_grid.mp4")
    
    print("\nExperimenting with latent manipulations...")
    manipulator = LatentManipulator(model)
    
    # Generate base sample
    z_base = torch.randn(1, model.latent_dim).to(args.device)
    
    # 1. Scaling experiment
    scales = [0.5, 1.0, 1.5, 2.0]
    scale_videos = []
    for scale in scales:
        z_scaled = z_base * scale
        with torch.no_grad():
            video = model.decode(z_scaled)
        scale_videos.append(video[0])
        save_video(video, f"{args.output_dir}/scale_{scale}.mp4")
    
    # 2. Dimension manipulation
    dim_videos = []
    for dim in [0, 10, 50, 100]:
        if dim < model.latent_dim:
            z_dim = z_base.clone()
            z_dim[:, dim] = z_dim[:, dim] * 3.0
            with torch.no_grad():
                video = model.decode(z_dim)
            dim_videos.append(video[0])
            save_video(video, f"{args.output_dir}/dimension_{dim}.mp4")
    
    # 3. Pattern application
    patterns = {
        'sine': lambda z: z * torch.sin(torch.linspace(0, 2*3.14159, z.shape[1]).to(z.device)),
        'sparse': lambda z: z * (torch.rand_like(z) > 0.7).float(),
        'threshold': lambda z: torch.where(torch.abs(z) > 1.0, z, torch.zeros_like(z))
    }
    
    for name, pattern_fn in patterns.items():
        z_pattern = pattern_fn(z_base)
        with torch.no_grad():
            video = model.decode(z_pattern)
        save_video(video, f"{args.output_dir}/pattern_{name}.mp4")
    
    print(f"\nAll samples saved to: {args.output_dir}")
    print("\nYou can view the results with:")
    print(f"  open {args.output_dir}/samples_grid.mp4")


if __name__ == '__main__':
    main()