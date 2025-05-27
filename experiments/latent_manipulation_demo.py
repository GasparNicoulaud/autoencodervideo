#!/usr/bin/env python3
"""
Demo script for latent space manipulation
"""
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_pretrained_model, VideoAutoencoder
from src.latent import LatentManipulator
from src.utils import load_video, save_video


def main():
    parser = argparse.ArgumentParser(description='Latent space manipulation demo')
    parser.add_argument('--model', type=str, default='vae_ucf101', 
                       help='Pretrained model name or path')
    parser.add_argument('--video', type=str, required=True,
                       help='Input video path')
    parser.add_argument('--output-dir', type=str, default='output/manipulations',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    if args.model.endswith('.pth'):
        model = VideoAutoencoder()
        model.load_state_dict(torch.load(args.model))
        model.to(args.device)
    else:
        model = load_pretrained_model(args.model, device=args.device)
    
    model.eval()
    
    print(f"Loading video: {args.video}")
    video = load_video(args.video, num_frames=32, size=(128, 128))
    video = video.unsqueeze(0).to(args.device)
    
    print("Encoding video...")
    with torch.no_grad():
        z, mu, log_var = model.encode(video)
        original_recon = model.decode(z)
    
    save_video(original_recon, f"{args.output_dir}/original_reconstruction.mp4")
    
    manipulator = LatentManipulator(model)
    manipulator.save_original_params()
    
    print("\n1. Scaling latent weights...")
    manipulator.scale_weights('decoder_fc', scale_factor=1.5)
    with torch.no_grad():
        scaled_recon = model.decode(z)
    save_video(scaled_recon, f"{args.output_dir}/scaled_weights.mp4")
    
    manipulator.restore_original_params()
    
    print("\n2. Adding noise to weights...")
    manipulator.add_noise('decoder_fc', noise_std=0.1)
    with torch.no_grad():
        noisy_recon = model.decode(z)
    save_video(noisy_recon, f"{args.output_dir}/noisy_weights.mp4")
    
    manipulator.restore_original_params()
    
    print("\n3. Quantizing weights...")
    manipulator.quantize_weights('decoder_fc', num_levels=16)
    with torch.no_grad():
        quantized_recon = model.decode(z)
    save_video(quantized_recon, f"{args.output_dir}/quantized_weights.mp4")
    
    print("\n4. Direct latent manipulation...")
    z_scaled = z * 2.0
    with torch.no_grad():
        scaled_latent_recon = model.decode(z_scaled)
    save_video(scaled_latent_recon, f"{args.output_dir}/scaled_latent.mp4")
    
    z_shifted = z + torch.randn_like(z) * 0.5
    with torch.no_grad():
        shifted_latent_recon = model.decode(z_shifted)
    save_video(shifted_latent_recon, f"{args.output_dir}/shifted_latent.mp4")
    
    print("\n5. Dimension-specific manipulation...")
    for dim in [0, 10, 50, 100]:
        z_dim_modified = z.clone()
        z_dim_modified[:, dim] = z_dim_modified[:, dim] * 3.0
        
        with torch.no_grad():
            dim_recon = model.decode(z_dim_modified)
        save_video(dim_recon, f"{args.output_dir}/dimension_{dim}_amplified.mp4")
    
    stats = manipulator.compute_weight_statistics()
    print("\n\nWeight statistics after manipulations:")
    for layer, layer_stats in stats.items():
        print(f"\n{layer}:")
        for param_type, param_stats in layer_stats.items():
            print(f"  {param_type}:")
            for stat_name, value in param_stats.items():
                if stat_name != 'shape':
                    print(f"    {stat_name}: {value:.4f}")
                else:
                    print(f"    {stat_name}: {value}")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()