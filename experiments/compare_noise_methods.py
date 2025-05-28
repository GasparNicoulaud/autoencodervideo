#!/usr/bin/env python3
"""
Compare Additive Noise vs Interpolation Methods
Shows the difference between latent + noise vs lerp(latent, noise)
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('.')

def save_video_frames(frames, path, fps=8):
    """Save list of numpy frames as video"""
    imageio.mimsave(path, frames, fps=fps)

def load_video_frames(path, num_frames=16, size=(512, 512)):
    """Load video frames as numpy arrays"""
    reader = imageio.get_reader(path)
    frames = []
    
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
            
        # Process frame with top cropping for portrait videos
        if frame.shape[:2] != size:
            img = Image.fromarray(frame)
            
            # Get current dimensions
            height, width = frame.shape[:2]
            target_height, target_width = size
            
            # For portrait videos, crop from top to get square
            if height > width:
                # Portrait: crop from top, take top square portion
                crop_size = min(width, height)
                left = (width - crop_size) // 2  # Center horizontally
                top = 0  # Start from top
                right = left + crop_size
                bottom = top + crop_size
                img = img.crop((left, top, right, bottom))
            else:
                # Landscape: use center crop
                crop_size = min(width, height)
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                img = img.crop((left, top, right, bottom))
            
            # Now resize to target size
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            frame = np.array(img)
            
        frames.append(frame)
        
    reader.close()
    return frames

def test_noise_methods(video_path, model_type="sd_vae", noise_levels=[1.0, 5.0, 10.0], num_frames=16):
    """Compare additive vs interpolation noise methods"""
    from diffusers import AutoencoderKL
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load VAE
    print(f"\nLoading {model_type} VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    # Load video
    print(f"Loading video: {video_path}")
    frames = load_video_frames(video_path, num_frames=num_frames, size=(512, 512))
    print(f"Loaded {len(frames)} frames")
    
    # Encode to latents
    print("Encoding to latent space...")
    latents = []
    with torch.no_grad():
        for frame in frames:
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            latent = vae.encode(frame_tensor).latent_dist.sample()
            latents.append(latent[0])
    
    latents = torch.stack(latents)
    latent_mean = latents.mean().item()
    latent_std = latents.std().item()
    print(f"Latent stats: mean={latent_mean:.3f}, std={latent_std:.3f}")
    
    results = {
        'original': frames,
        'additive': {},
        'interpolation': {},
        'pure_noise': None
    }
    
    # Test pure noise first
    print("\nGenerating pure noise...")
    pure_noise_frames = []
    with torch.no_grad():
        for i in range(num_frames):
            # Generate noise with same distribution as latents
            noise = torch.randn(1, 4, 64, 64).to(device) * latent_std + latent_mean
            frame = vae.decode(noise).sample[0]
            frame = ((frame + 1) * 127.5).clamp(0, 255)
            frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            pure_noise_frames.append(frame)
    results['pure_noise'] = pure_noise_frames
    
    # Test both methods at each noise level
    for noise_level in noise_levels:
        print(f"\nTesting noise level {noise_level}...")
        
        # Method 1: Additive (original method)
        print(f"  Additive method: latent + noise * {noise_level}")
        noise = torch.randn_like(latents) * noise_level
        noisy_latents = latents + noise
        
        decoded_frames = []
        with torch.no_grad():
            for latent in noisy_latents:
                frame = vae.decode(latent.unsqueeze(0)).sample[0]
                frame = ((frame + 1) * 127.5).clamp(0, 255)
                frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                decoded_frames.append(frame)
        results['additive'][noise_level] = decoded_frames
        
        # Method 2: Interpolation
        alpha = min(1.0, noise_level / 10.0)  # Convert to 0-1 range
        print(f"  Interpolation method: lerp(latent, noise, {alpha:.2f})")
        
        # Generate noise with same distribution as latents
        noise = torch.randn_like(latents) * latent_std + latent_mean
        noisy_latents = (1 - alpha) * latents + alpha * noise
        
        decoded_frames = []
        with torch.no_grad():
            for latent in noisy_latents:
                frame = vae.decode(latent.unsqueeze(0)).sample[0]
                frame = ((frame + 1) * 127.5).clamp(0, 255)
                frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                decoded_frames.append(frame)
        results['interpolation'][noise_level] = decoded_frames
    
    return results

def create_comparison_grid(results, output_path, frame_indices=[0, 4, 8, 12]):
    """Create a visual comparison grid"""
    # Setup grid dimensions
    cell_size = 200
    padding = 5
    label_height = 30
    
    noise_levels = sorted(list(results['additive'].keys()))
    num_cols = len(frame_indices) + 1  # +1 for labels
    num_rows = 2 + len(noise_levels) * 2 + 1  # Original, methods, pure noise
    
    grid_width = num_cols * (cell_size + padding) + padding
    grid_height = num_rows * (cell_size + padding) + padding + label_height
    
    grid = Image.new('RGB', (grid_width, grid_height), color='black')
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Title
    if font:
        draw.text((10, 5), "Additive vs Interpolation Noise Methods", fill='white', font=font)
    
    y = label_height
    
    # Original frames
    x = padding
    if font:
        draw.text((x + 10, y + cell_size//2), "Original", fill='white', font=font)
    x += cell_size + padding
    
    for idx in frame_indices:
        if idx < len(results['original']):
            img = Image.fromarray(results['original'][idx])
            img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            grid.paste(img, (x, y))
        x += cell_size + padding
    y += cell_size + padding
    
    # Noise comparisons
    for noise_level in noise_levels:
        # Additive method
        x = padding
        if font:
            draw.text((x + 10, y + cell_size//2), f"Add {noise_level}", fill='white', font=font)
        x += cell_size + padding
        
        for idx in frame_indices:
            if idx < len(results['additive'][noise_level]):
                img = Image.fromarray(results['additive'][noise_level][idx])
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y))
            x += cell_size + padding
        y += cell_size + padding
        
        # Interpolation method
        x = padding
        alpha = min(1.0, noise_level / 10.0)
        if font:
            draw.text((x + 10, y + cell_size//2), f"Lerp α={alpha:.1f}", fill='white', font=font)
        x += cell_size + padding
        
        for idx in frame_indices:
            if idx < len(results['interpolation'][noise_level]):
                img = Image.fromarray(results['interpolation'][noise_level][idx])
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y))
            x += cell_size + padding
        y += cell_size + padding
    
    # Pure noise
    x = padding
    if font:
        draw.text((x + 10, y + cell_size//2), "Pure Noise", fill='white', font=font)
    x += cell_size + padding
    
    if results['pure_noise']:
        for idx in frame_indices:
            if idx < len(results['pure_noise']):
                img = Image.fromarray(results['pure_noise'][idx])
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y))
            x += cell_size + padding
    
    # Save grid
    grid.save(str(output_path))
    print(f"✅ Saved comparison grid: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--noise', type=str, default='1,5,10', help='Comma-separated noise levels')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--output', type=str, default='output_noise_comparison')
    
    args = parser.parse_args()
    
    noise_levels = [float(x) for x in args.noise.split(',')]
    
    print("="*70)
    print("NOISE METHOD COMPARISON: ADDITIVE vs INTERPOLATION")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Noise levels: {noise_levels}")
    print(f"Frames: {args.frames}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Run comparison
    results = test_noise_methods(args.video, noise_levels=noise_levels, num_frames=args.frames)
    
    # Save all videos
    print("\nSaving videos...")
    
    # Original
    save_video_frames(results['original'], str(output_dir / "original.mp4"))
    
    # Additive method
    for level, frames in results['additive'].items():
        save_video_frames(frames, str(output_dir / f"additive_noise_{level}.mp4"))
    
    # Interpolation method
    for level, frames in results['interpolation'].items():
        save_video_frames(frames, str(output_dir / f"interpolation_noise_{level}.mp4"))
    
    # Pure noise
    save_video_frames(results['pure_noise'], str(output_dir / "pure_noise.mp4"))
    
    # Create comparison grid
    create_comparison_grid(results, output_dir / "method_comparison.png")
    
    print(f"\n{'='*70}")
    print("✅ COMPARISON COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print("\nKey differences:")
    print("- Additive: Creates high-contrast noise, exceeds normal latent range")
    print("- Interpolation: Stays within latent distribution, more natural")
    print("- Pure noise: Shows what the VAE generates from random latents")
    print("\nThe interpolation method should produce more 'semantic' glitches!")
    print("="*70)

if __name__ == "__main__":
    main()