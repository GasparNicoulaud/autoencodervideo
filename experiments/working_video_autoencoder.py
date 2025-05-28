#!/usr/bin/env python3
"""
Working Video Autoencoder with Uniform Noise
Testing the hypothesis that good autoencoders create structured semantic glitches from uniform noise
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image
import sys
sys.path.append('.')
from src.utils.video_io import save_video

def save_video_frames(frames, path, fps=8):
    """Save list of numpy frames as video"""
    imageio.mimsave(path, frames, fps=fps)

def load_video_frames(path, num_frames=16, target_size=None):
    """Load video frames with optimal sizing"""
    reader = imageio.get_reader(path)
    frames = []
    
    # Get first frame to determine input size
    first_frame = reader.get_data(0)
    input_height, input_width = first_frame.shape[:2]
    
    # Use provided target_size or calculate optimal
    if target_size is None:
        # Use the minimum dimension as square size, capped at 1024
        target_size = min(input_width, input_height)
        target_size = min(target_size, 1024)
    
    print(f"Input video size: {input_width}x{input_height}")
    print(f"Processing at: {target_size}x{target_size}")
    
    reader.close()
    reader = imageio.get_reader(path)  # Restart reader
    
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
            
        # Process frame with top cropping for portrait videos
        if frame.shape[:2] != (target_size, target_size):
            img = Image.fromarray(frame)
            
            # Get current dimensions
            width, height = img.size
            
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
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            frame = np.array(img)
            
        frames.append(frame)
        
    reader.close()
    return frames

def test_animatediff_uniform(video_path, output_dir, noise_levels=[1.0, 3.0, 5.0, 10.0]):
    """Test AnimateDiff VAE with uniform noise - should create motion-aware glitches"""
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter
        print("\n" + "="*60)
        print("TESTING: AnimateDiff VAE (Motion-Aware)")
        print("="*60)
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load AnimateDiff
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
        pipe = AnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            motion_adapter=adapter,
            torch_dtype=torch.float32
        )
        vae = pipe.vae.to(device)
        vae.eval()
        
        # Load video
        frames = load_video_frames(video_path, num_frames=16)
        print(f"Loaded {len(frames)} frames")
        
        # Encode
        print("Encoding to latent space...")
        latents = []
        with torch.no_grad():
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
                frame_tensor = frame_tensor.unsqueeze(0).to(device)
                latent = vae.encode(frame_tensor).latent_dist.sample()
                latents.append(latent[0])
        
        latents = torch.stack(latents)
        print(f"Latent shape: {latents.shape}")
        print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
        
        # Test different noise levels
        results = []
        for noise_level in noise_levels:
            print(f"\nApplying uniform noise (strength={noise_level})...")
            
            # Simple uniform noise
            noise = torch.randn_like(latents) * noise_level
            noisy_latents = latents + noise
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    frame = vae.decode(latent.unsqueeze(0)).sample[0]
                    frame = ((frame + 1) * 127.5).clamp(0, 255)
                    frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    decoded_frames.append(frame)
            
            # Save
            save_path = output_dir / f"animatediff_noise_{noise_level}.mp4"
            save_video_frames(decoded_frames, str(save_path), fps=8)
            results.append((noise_level, decoded_frames))
            print(f"✅ Saved: {save_path}")
        
        return results
        
    except Exception as e:
        print(f"AnimateDiff failed: {e}")
        return None

def test_zeroscope_uniform(video_path, output_dir, noise_levels=[1.0, 3.0, 5.0, 10.0]):
    """Test ZeroScope VAE with uniform noise - video-specific model"""
    try:
        from diffusers import DiffusionPipeline
        print("\n" + "="*60)
        print("TESTING: ZeroScope V2 VAE (Video-Optimized)")
        print("="*60)
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Load ZeroScope
        pipe = DiffusionPipeline.from_pretrained(
            "cerspense/zeroscope_v2_576w",
            torch_dtype=torch.float32
        )
        vae = pipe.vae.to(device)
        vae.eval()
        
        # Load video
        frames = load_video_frames(video_path, num_frames=16)
        print(f"Loaded {len(frames)} frames")
        
        # Encode
        print("Encoding to latent space...")
        latents = []
        with torch.no_grad():
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
                frame_tensor = frame_tensor.unsqueeze(0).to(device)
                latent = vae.encode(frame_tensor).latent_dist.sample()
                latents.append(latent[0])
        
        latents = torch.stack(latents)
        print(f"Latent shape: {latents.shape}")
        print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
        
        # Test different noise levels
        results = []
        for noise_level in noise_levels:
            print(f"\nApplying uniform noise (strength={noise_level})...")
            
            # Simple uniform noise
            noise = torch.randn_like(latents) * noise_level
            noisy_latents = latents + noise
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    frame = vae.decode(latent.unsqueeze(0)).sample[0]
                    frame = ((frame + 1) * 127.5).clamp(0, 255)
                    frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    decoded_frames.append(frame)
            
            # Resize back to 512x512 for consistency
            from PIL import Image
            resized_frames = []
            for frame in decoded_frames:
                img = Image.fromarray(frame)
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            
            # Save
            save_path = output_dir / f"zeroscope_noise_{noise_level}.mp4"
            save_video_frames(resized_frames, str(save_path), fps=8)
            results.append((noise_level, resized_frames))
            print(f"✅ Saved: {save_path}")
        
        return results
        
    except Exception as e:
        print(f"ZeroScope failed: {e}")
        return None

def test_sd_vae_uniform(video_path, output_dir, noise_levels=[1.0, 3.0, 5.0, 10.0]):
    """Test standard SD VAE as baseline - not video-aware"""
    from diffusers import AutoencoderKL
    print("\n" + "="*60)
    print("TESTING: SD VAE (Baseline - Not Video-Aware)")
    print("="*60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load SD VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    # Load video
    frames = load_video_frames(video_path, num_frames=16)
    print(f"Loaded {len(frames)} frames")
    
    # Encode
    print("Encoding to latent space...")
    latents = []
    with torch.no_grad():
        for frame in frames:
            # Handle both numpy arrays and tensors
            if isinstance(frame, torch.Tensor):
                frame_tensor = frame.float() / 127.5 - 1
            else:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
            frame_tensor = frame_tensor.unsqueeze(0).to(device)
            latent = vae.encode(frame_tensor).latent_dist.sample()
            latents.append(latent[0])
    
    latents = torch.stack(latents)
    print(f"Latent shape: {latents.shape}")
    print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
    
    # Test different noise levels
    results = []
    for noise_level in noise_levels:
        print(f"\nApplying uniform noise (strength={noise_level})...")
        
        # Simple uniform noise
        noise = torch.randn_like(latents) * noise_level
        noisy_latents = latents + noise
        
        # Decode
        decoded_frames = []
        with torch.no_grad():
            for latent in noisy_latents:
                frame = vae.decode(latent.unsqueeze(0)).sample[0]
                frame = ((frame + 1) * 127.5).clamp(0, 255)
                frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                decoded_frames.append(frame)
        
        # Save
        save_path = output_dir / f"sd_vae_noise_{noise_level}.mp4"
        save_video_frames(decoded_frames, str(save_path), fps=8)
        results.append((noise_level, decoded_frames))
        print(f"✅ Saved: {save_path}")
    
    return results

def create_comparison_grid(all_results, output_dir, frame_idx=8):
    """Create a comparison grid showing different models and noise levels"""
    from PIL import Image, ImageDraw, ImageFont
    
    print("\nCreating comparison grid...")
    
    # Extract frames
    models = list(all_results.keys())
    noise_levels = [1.0, 3.0, 5.0, 10.0]
    
    # Create grid
    cell_size = 256
    padding = 5
    grid_width = (len(noise_levels) + 1) * (cell_size + padding) + padding
    grid_height = len(models) * (cell_size + padding) + padding + 40  # Extra space for labels
    
    grid = Image.new('RGB', (grid_width, grid_height), color='black')
    
    # Add noise level labels
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except:
        font = None
    
    draw = ImageDraw.Draw(grid)
    
    # Column headers
    x = padding + cell_size + padding
    for noise_level in noise_levels:
        if font:
            draw.text((x + cell_size//2 - 20, 10), f"Noise {noise_level}", fill='white', font=font)
        x += cell_size + padding
    
    # Add frames to grid
    y = 40
    for model in models:
        x = padding
        
        # Model label
        if font:
            draw.text((x + 10, y + cell_size//2), model.split('_')[0], fill='white', font=font)
        
        # Original frame
        if all_results[model] and len(all_results[model]) > 0:
            # Get original (first result, before noise)
            original_frames = all_results[model][0][1]
            if frame_idx < len(original_frames):
                img = Image.fromarray(original_frames[frame_idx])
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y))
        
        x += cell_size + padding
        
        # Noisy frames
        for i, (noise_level, frames) in enumerate(all_results[model]):
            if frame_idx < len(frames):
                img = Image.fromarray(frames[frame_idx])
                img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                grid.paste(img, (x, y))
            x += cell_size + padding
        
        y += cell_size + padding
    
    # Save grid
    grid_path = output_dir / "model_comparison_grid.png"
    grid.save(str(grid_path))
    print(f"✅ Saved comparison grid: {grid_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='output_working_vae')
    parser.add_argument('--noise', type=str, default='1,3,5,10', help='Comma-separated noise levels')
    
    args = parser.parse_args()
    
    print("="*70)
    print("WORKING VIDEO AUTOENCODER EXPERIMENT")
    print("="*70)
    print("Testing uniform noise with different video autoencoders")
    print("Hypothesis: Better autoencoders create structured semantic glitches")
    print("="*70)
    
    # Parse noise levels
    noise_levels = [float(x) for x in args.noise.split(',')]
    print(f"Noise levels to test: {noise_levels}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Test each model
    all_results = {}
    
    # 1. SD VAE (baseline)
    sd_results = test_sd_vae_uniform(args.video, output_dir, noise_levels)
    if sd_results:
        all_results['sd_vae'] = sd_results
    
    # 2. AnimateDiff VAE (motion-aware)
    ad_results = test_animatediff_uniform(args.video, output_dir, noise_levels)
    if ad_results:
        all_results['animatediff'] = ad_results
    
    # 3. ZeroScope VAE (video-optimized)
    zs_results = test_zeroscope_uniform(args.video, output_dir, noise_levels)
    if zs_results:
        all_results['zeroscope'] = zs_results
    
    # Create comparison grid
    if all_results:
        create_comparison_grid(all_results, output_dir)
    
    # Save original for reference
    frames = load_video_frames(args.video, num_frames=16)
    save_video_frames(frames, str(output_dir / "original.mp4"), fps=8)
    
    print(f"\n{'='*70}")
    print("✅ EXPERIMENT COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print("\nKey insights to look for:")
    print("1. SD VAE: Should show pixel-level noise, less structure")
    print("2. AnimateDiff: Should show motion-coherent glitches")
    print("3. ZeroScope: Should show video-aware semantic distortions")
    print("\nThe better the autoencoder, the more structured the glitches!")
    print("="*70)

if __name__ == "__main__":
    main()