#!/usr/bin/env python3
"""
Advanced programmatic control of latent space
"""
import torch
import numpy as np
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models import VideoAutoencoder
from src.latent import LatentManipulator
from src.utils import save_video


class ProgrammaticLatentController:
    def __init__(self, model):
        self.model = model
        self.manipulator = LatentManipulator(model)
        
    def create_custom_weight_pattern(self, layer_name, pattern_type='gradient'):
        """Create custom weight patterns programmatically"""
        params = self.model.get_latent_params()
        
        if layer_name in params and 'weight' in params[layer_name]:
            weights = params[layer_name]['weight']
            
            if pattern_type == 'gradient':
                rows, cols = weights.shape
                gradient = torch.linspace(0, 1, cols).unsqueeze(0).repeat(rows, 1)
                new_weights = weights * gradient.to(weights.device)
                
            elif pattern_type == 'checkerboard':
                rows, cols = weights.shape
                checkerboard = torch.zeros_like(weights)
                checkerboard[::2, ::2] = 1
                checkerboard[1::2, 1::2] = 1
                new_weights = weights * checkerboard
                
            elif pattern_type == 'radial':
                rows, cols = weights.shape
                center_r, center_c = rows // 2, cols // 2
                r_indices = torch.arange(rows).unsqueeze(1).float()
                c_indices = torch.arange(cols).unsqueeze(0).float()
                distances = torch.sqrt((r_indices - center_r)**2 + (c_indices - center_c)**2)
                radial_mask = 1 - (distances / distances.max())
                new_weights = weights * radial_mask.to(weights.device)
                
            params[layer_name]['weight'] = new_weights
            self.model.set_latent_params(params)
            
    def apply_frequency_filter(self, layer_name, cutoff_freq=0.1, filter_type='low'):
        """Apply frequency filtering to weights"""
        def freq_filter(w):
            w_fft = torch.fft.fft2(w.float())
            rows, cols = w.shape
            
            crow, ccol = rows // 2, cols // 2
            mask = torch.zeros_like(w)
            
            if filter_type == 'low':
                r = int(cutoff_freq * min(rows, cols) / 2)
                center_square = torch.ones(2*r, 2*r)
                mask[crow-r:crow+r, ccol-r:ccol+r] = center_square
            elif filter_type == 'high':
                mask = torch.ones_like(w)
                r = int(cutoff_freq * min(rows, cols) / 2)
                mask[crow-r:crow+r, ccol-r:ccol+r] = 0
                
            w_fft_filtered = w_fft * mask.to(w.device)
            w_filtered = torch.fft.ifft2(w_fft_filtered).real
            
            return w_filtered.to(w.dtype)
            
        self.manipulator.manipulate_weights(layer_name, freq_filter)
        
    def create_activation_patterns(self, z, pattern='sine'):
        """Create specific activation patterns in latent space"""
        if pattern == 'sine':
            t = torch.linspace(0, 2*np.pi, z.shape[1])
            modulation = torch.sin(t).to(z.device)
            return z * modulation
            
        elif pattern == 'pulse':
            mask = torch.zeros_like(z)
            mask[:, ::10] = 2.0
            return z * mask
            
        elif pattern == 'exponential':
            decay = torch.exp(-torch.linspace(0, 5, z.shape[1])).to(z.device)
            return z * decay
            
        elif pattern == 'random_sparse':
            mask = (torch.rand_like(z) > 0.8).float() * 2.0
            return z * mask
            
        return z


def main():
    parser = argparse.ArgumentParser(description='Programmatic latent control demo')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--output-dir', type=str, default='output/programmatic',
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Initializing model...")
    model = VideoAutoencoder(latent_dim=512, base_channels=64)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(args.device)
    model.eval()
    
    controller = ProgrammaticLatentController(model)
    
    print("\n1. Generating with different weight patterns...")
    z_random = torch.randn(1, 512).to(args.device)
    
    patterns = ['gradient', 'checkerboard', 'radial']
    for pattern in patterns:
        controller.manipulator.save_original_params()
        controller.create_custom_weight_pattern('decoder_fc', pattern)
        
        with torch.no_grad():
            video = model.decode(z_random)
        save_video(video, f"{args.output_dir}/weight_pattern_{pattern}.mp4")
        
        controller.manipulator.restore_original_params()
    
    print("\n2. Applying frequency filters...")
    for filter_type in ['low', 'high']:
        for cutoff in [0.05, 0.1, 0.2]:
            controller.manipulator.save_original_params()
            controller.apply_frequency_filter('decoder_fc', cutoff, filter_type)
            
            with torch.no_grad():
                video = model.decode(z_random)
            save_video(video, f"{args.output_dir}/freq_filter_{filter_type}_{cutoff}.mp4")
            
            controller.manipulator.restore_original_params()
    
    print("\n3. Creating activation patterns...")
    activation_patterns = ['sine', 'pulse', 'exponential', 'random_sparse']
    
    for pattern in activation_patterns:
        z_modulated = controller.create_activation_patterns(z_random, pattern)
        
        with torch.no_grad():
            video = model.decode(z_modulated)
        save_video(video, f"{args.output_dir}/activation_pattern_{pattern}.mp4")
    
    print("\n4. Programmatic weight evolution...")
    evolution_steps = 10
    evolution_videos = []
    
    controller.manipulator.save_original_params()
    
    for step in range(evolution_steps):
        alpha = step / (evolution_steps - 1)
        
        controller.manipulator.scale_weights('decoder_fc', 0.5 + alpha * 1.5)
        
        noise_level = 0.2 * (1 - alpha)
        controller.manipulator.add_noise('encoder_fc', noise_level)
        
        with torch.no_grad():
            video = model.decode(z_random)
        evolution_videos.append(video[0])
        
        save_video(video, f"{args.output_dir}/evolution_step_{step:02d}.mp4")
        
        controller.manipulator.restore_original_params()
        controller.manipulator.save_original_params()
    
    print("\n5. Creating custom latent space functions...")
    
    def custom_transform(z):
        z_transformed = z.clone()
        z_transformed[:, :100] = torch.tanh(z[:, :100] * 2)
        z_transformed[:, 100:200] = torch.sigmoid(z[:, 100:200])
        z_transformed[:, 200:300] = torch.relu(z[:, 200:300])
        z_transformed[:, 300:] = z[:, 300:] ** 2
        return z_transformed
    
    z_custom = custom_transform(z_random)
    with torch.no_grad():
        video = model.decode(z_custom)
    save_video(video, f"{args.output_dir}/custom_transform.mp4")
    
    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()