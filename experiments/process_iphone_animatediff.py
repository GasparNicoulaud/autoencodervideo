#!/usr/bin/env python3
"""
Process iPhone videos (.mov) with AnimateDiff for temporal consistency
"""
import torch
import numpy as np
import imageio
from PIL import Image
from pathlib import Path
import sys
import argparse
sys.path.append(str(Path(__file__).parent.parent))


def prepare_video_for_animatediff(video_path, target_frames=16, target_size=512):
    """
    Prepare iPhone video (including .mov) for AnimateDiff
    
    AnimateDiff requirements:
    - Square videos (256x256, 512x512, or 768x768)
    - Fixed frame counts (8, 16, 24, 32)
    - Consistent frame timing
    """
    print(f"\n1. Loading video: {video_path}")
    
    # imageio-ffmpeg handles .mov files automatically
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        
        print(f"   Format: {Path(video_path).suffix}")
        print(f"   Original resolution: {meta.get('size', 'unknown')}")
        print(f"   FPS: {meta.get('fps', 30)}")
        print(f"   Duration: {meta.get('duration', 'unknown')}s")
        
    except Exception as e:
        print(f"   Error reading video: {e}")
        print("   Make sure imageio-ffmpeg is installed: pip install imageio-ffmpeg")
        return None
    
    # Sample frames evenly for AnimateDiff
    frames = []
    total_frames = meta.get('nframes', 100)
    
    # Calculate which frames to sample
    if total_frames > target_frames:
        # Sample evenly across the video
        frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    else:
        # Use all frames if video is short
        frame_indices = range(total_frames)
    
    print(f"\n2. Sampling {len(frame_indices)} frames from {total_frames} total...")
    
    current_idx = 0
    for i, frame in enumerate(reader):
        if current_idx < len(frame_indices) and i == frame_indices[current_idx]:
            frames.append(frame)
            current_idx += 1
            print(f"   Frame {current_idx}/{len(frame_indices)}", end='\r')
        
        if len(frames) >= target_frames:
            break
    
    reader.close()
    print(f"\n   Collected {len(frames)} frames")
    
    # Resize for AnimateDiff
    print(f"\n3. Resizing to {target_size}x{target_size} (AnimateDiff requirement)...")
    resized_frames = []
    
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        
        # Get dimensions
        width, height = img.size
        
        # Center crop to square (AnimateDiff needs square videos)
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize to target
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        resized_frames.append(np.array(img))
    
    video_array = np.array(resized_frames)
    print(f"   Final shape: {video_array.shape}")
    
    return video_array


def process_with_animatediff_vae(video_array, noise_level=0.3):
    """
    Process video with AnimateDiff's VAE for temporal consistency
    """
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        from diffusers.utils import export_to_video
    except ImportError:
        print("\n❌ AnimateDiff not installed!")
        print("Please run: pip install diffusers transformers accelerate")
        return None
    
    print("\n4. Loading AnimateDiff...")
    
    # For VAE-only usage, we'll use the motion module approach
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Note: Full AnimateDiff pipeline includes text-to-video
    # For VAE-only, we need a different approach
    print("   Note: AnimateDiff is primarily designed for text-to-video generation")
    print("   For VAE-only usage, we'll use the motion-aware encoding approach")
    
    # Convert video to tensor
    video_tensor = torch.from_numpy(video_array).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    video_tensor = video_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    
    print(f"   Video tensor shape: {video_tensor.shape}")
    
    # For now, return the tensor for processing
    # Full AnimateDiff integration would require more setup
    return video_tensor


def process_with_simple_temporal_vae(video_array, noise_level=0.3):
    """
    Alternative: Use SD VAE with temporal consistency tricks
    """
    from diffusers import AutoencoderKL
    
    print("\n4. Using SD VAE with temporal consistency...")
    device = "cpu"  # Use CPU for stability
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    # Convert to tensor
    video_tensor = torch.from_numpy(video_array).float() / 255.0
    video_tensor = video_tensor.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
    video_tensor = video_tensor * 2.0 - 1.0
    
    frames_count = video_tensor.shape[0]
    
    # Encode all frames
    print("\n5. Encoding with temporal consistency...")
    latents = []
    
    with torch.no_grad():
        for i in range(frames_count):
            frame = video_tensor[i:i+1]
            latent = vae.encode(frame).latent_dist.sample()
            latents.append(latent)
    
    # Apply temporally consistent noise
    print("\n6. Adding temporally consistent noise...")
    
    # Create smooth noise that changes gradually over time
    base_noise = torch.randn_like(latents[0])
    noisy_latents = []
    
    for i in range(frames_count):
        # Interpolate noise over time for consistency
        t = i / (frames_count - 1)
        
        # Create time-varying noise
        if i == 0:
            frame_noise = base_noise
        else:
            # Blend with previous frame's noise for consistency
            new_noise = torch.randn_like(base_noise)
            frame_noise = (1 - 0.3) * prev_noise + 0.3 * new_noise
        
        prev_noise = frame_noise
        
        # Apply noise
        noisy_latent = latents[i] + frame_noise * noise_level
        noisy_latents.append(noisy_latent)
    
    # Decode all frames
    print("\n7. Decoding...")
    clean_frames = []
    noisy_frames = []
    
    with torch.no_grad():
        for i in range(frames_count):
            # Clean reconstruction
            clean = vae.decode(latents[i]).sample[0]
            clean_frames.append(clean)
            
            # Noisy reconstruction
            noisy = vae.decode(noisy_latents[i]).sample[0]
            noisy_frames.append(noisy)
    
    return clean_frames, noisy_frames


def main():
    parser = argparse.ArgumentParser(description='Process iPhone video with temporal consistency')
    parser.add_argument('--video', type=str, required=True, help='Path to video file (.mov, .mp4, etc)')
    parser.add_argument('--frames', type=int, default=16, choices=[8, 16, 24, 32],
                       help='Number of frames (AnimateDiff supports 8,16,24,32)')
    parser.add_argument('--size', type=int, default=512, choices=[256, 512, 768],
                       help='Resolution (AnimateDiff supports 256,512,768)')
    parser.add_argument('--noise', type=float, default=0.3, help='Noise level')
    parser.add_argument('--method', type=str, default='temporal-vae', 
                       choices=['temporal-vae', 'animatediff'],
                       help='Processing method')
    args = parser.parse_args()
    
    # Prepare video
    video_array = prepare_video_for_animatediff(args.video, args.frames, args.size)
    
    if video_array is None:
        return
    
    # Save original
    Path("output").mkdir(exist_ok=True)
    imageio.mimsave('output/iphone_original_ad.mp4', video_array, fps=8)
    print(f"\nSaved: output/iphone_original_ad.mp4")
    
    if args.method == 'animatediff':
        print("\n⚠️  Full AnimateDiff integration requires additional setup")
        print("Using temporal VAE approach instead...")
    
    # Process with temporal consistency
    clean_frames, noisy_frames = process_with_simple_temporal_vae(video_array, args.noise)
    
    # Convert and save
    print("\n8. Saving results...")
    
    # Clean reconstruction
    clean_video = torch.stack(clean_frames)
    clean_video = clean_video.permute(0, 2, 3, 1)  # (T,C,H,W) -> (T,H,W,C)
    clean_video = ((clean_video + 1.0) / 2.0).clamp(0, 1)
    clean_video_np = (clean_video * 255).byte().cpu().numpy()
    imageio.mimsave('output/iphone_clean_ad.mp4', clean_video_np, fps=8)
    
    # Noisy reconstruction
    noisy_video = torch.stack(noisy_frames)
    noisy_video = noisy_video.permute(0, 2, 3, 1)
    noisy_video = ((noisy_video + 1.0) / 2.0).clamp(0, 1)
    noisy_video_np = (noisy_video * 255).byte().cpu().numpy()
    imageio.mimsave('output/iphone_noisy_ad.mp4', noisy_video_np, fps=8)
    
    print("\n✅ Done! Results with temporal consistency:")
    print("  - output/iphone_original_ad.mp4")
    print("  - output/iphone_clean_ad.mp4")
    print("  - output/iphone_noisy_ad.mp4")
    print(f"\nProcessed: {args.frames} frames at {args.size}x{args.size}")
    print("Note: Used temporally consistent noise for smoother results")


if __name__ == '__main__':
    main()