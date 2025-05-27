#!/usr/bin/env python3
"""
Demo script for latent space interpolation
"""
import torch
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_pretrained_model
from src.latent import interpolate_latents, interpolate_along_path, radial_interpolation
from src.utils import load_video, save_video, create_video_grid


def main():
    parser = argparse.ArgumentParser(description='Latent space interpolation demo')
    parser.add_argument('--model', type=str, default='vae_ucf101', 
                       help='Pretrained model name or path')
    parser.add_argument('--video1', type=str, required=True,
                       help='First input video path')
    parser.add_argument('--video2', type=str, required=True,
                       help='Second input video path')
    parser.add_argument('--output-dir', type=str, default='output/interpolations',
                       help='Output directory')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of interpolation steps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    model = load_pretrained_model(args.model, device=args.device)
    model.eval()
    
    print(f"Loading videos...")
    video1 = load_video(args.video1, num_frames=32, size=(128, 128))
    video2 = load_video(args.video2, num_frames=32, size=(128, 128))
    
    video1 = video1.unsqueeze(0).to(args.device)
    video2 = video2.unsqueeze(0).to(args.device)
    
    print("Encoding videos...")
    with torch.no_grad():
        z1, _, _ = model.encode(video1)
        z2, _, _ = model.encode(video2)
    
    print("\n1. Linear interpolation...")
    linear_interp = interpolate_latents(z1[0], z2[0], steps=args.steps, method='linear')
    
    linear_videos = []
    with torch.no_grad():
        for z in linear_interp:
            recon = model.decode(z.unsqueeze(0))
            linear_videos.append(recon[0])
    
    linear_grid = create_video_grid(linear_videos[:10])
    save_video(linear_grid, f"{args.output_dir}/linear_interpolation_grid.mp4")
    
    combined_linear = torch.stack(linear_videos, dim=1).squeeze(0)
    save_video(combined_linear, f"{args.output_dir}/linear_interpolation_sequence.mp4")
    
    print("\n2. Spherical interpolation...")
    spherical_interp = interpolate_latents(z1[0], z2[0], steps=args.steps, method='spherical')
    
    spherical_videos = []
    with torch.no_grad():
        for z in spherical_interp:
            recon = model.decode(z.unsqueeze(0))
            spherical_videos.append(recon[0])
    
    spherical_grid = create_video_grid(spherical_videos[:10])
    save_video(spherical_grid, f"{args.output_dir}/spherical_interpolation_grid.mp4")
    
    combined_spherical = torch.stack(spherical_videos, dim=1).squeeze(0)
    save_video(combined_spherical, f"{args.output_dir}/spherical_interpolation_sequence.mp4")
    
    print("\n3. Circular path interpolation...")
    center = (z1[0] + z2[0]) / 2
    radius = torch.norm(z1[0] - z2[0]) / 4
    
    circular_path = radial_interpolation(center, radius, num_points=args.steps)
    
    circular_videos = []
    with torch.no_grad():
        for z in circular_path:
            recon = model.decode(z.unsqueeze(0))
            circular_videos.append(recon[0])
    
    combined_circular = torch.stack(circular_videos, dim=1).squeeze(0)
    save_video(combined_circular, f"{args.output_dir}/circular_interpolation.mp4")
    
    print("\n4. Multi-point path interpolation...")
    z3 = z1[0] + torch.randn_like(z1[0]) * 0.3
    z4 = z2[0] + torch.randn_like(z2[0]) * 0.3
    
    path_interp = interpolate_along_path([z1[0], z3, z4, z2[0]], 
                                        steps_per_segment=10, 
                                        loop=True)
    
    path_videos = []
    with torch.no_grad():
        for z in path_interp:
            recon = model.decode(z.unsqueeze(0))
            path_videos.append(recon[0])
    
    combined_path = torch.stack(path_videos, dim=1).squeeze(0)
    save_video(combined_path, f"{args.output_dir}/path_interpolation.mp4")
    
    print(f"\nAll interpolations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()