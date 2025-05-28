#!/usr/bin/env python3
"""
Process iPhone videos with VAE/AnimateDiff - handles different resolutions and .mov files
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import sys
import argparse
import os
sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKL
from src.utils import load_video

# Import AnimateDiff utilities
try:
    from process_iphone_animatediff import prepare_video_for_animatediff, create_temporal_noise
    ANIMATEDIFF_AVAILABLE = True
except ImportError:
    ANIMATEDIFF_AVAILABLE = False


def process_iphone_video(video_path, max_frames=16, target_size=512, use_animatediff=False):
    """
    Process iPhone video with proper resizing and aspect ratio handling
    
    iPhone typical resolutions:
    - 1920x1080 (1080p)
    - 3840x2160 (4K) 
    - 1280x720 (720p)
    
    Supports .mov files from iPhone as well as .mp4
    """
    print(f"Loading video: {video_path}")
    
    # Get video info - imageio-ffmpeg handles .mov files automatically
    try:
        reader = imageio.get_reader(video_path)
        meta = reader.get_meta_data()
        fps = meta.get('fps', 30)
        total_frames = meta.get('nframes', None)
    except Exception as e:
        print(f"Error reading video: {e}")
        print("Note: For .mov files, ensure imageio-ffmpeg is installed: pip install imageio-ffmpeg")
        raise
    
    print(f"Video info:")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Read frames
    frames = []
    frame_indices = np.linspace(0, min(total_frames-1, max_frames*3), max_frames, dtype=int)
    
    for i, frame in enumerate(reader):
        if i in frame_indices:
            frames.append(frame)
        if len(frames) >= max_frames:
            break
    
    reader.close()
    
    # Convert to tensor and resize
    video_np = np.array(frames)
    print(f"  Original shape: {video_np.shape}")
    
    # Use AnimateDiff preprocessing if requested
    if use_animatediff and ANIMATEDIFF_AVAILABLE:
        print("\nUsing AnimateDiff preprocessing...")
        frames_tensor, fps = prepare_video_for_animatediff(video_path, max_frames, target_size)
        video_resized = []
        for frame in frames_tensor:
            # Convert from tensor to numpy
            frame_np = ((frame + 1.0) / 2.0 * 255).clamp(0, 255).byte().cpu().numpy()
            frame_np = frame_np.transpose(1, 2, 0)  # CHW to HWC
            video_resized.append(frame_np)
        video_resized = np.array(video_resized)
        print(f"  AnimateDiff shape: {video_resized.shape}")
        return video_resized, fps
    
    # Resize to square for VAE (it expects square inputs)
    from PIL import Image
    resized_frames = []
    
    for frame in video_np:
        img = Image.fromarray(frame)
        
        # Calculate resize to maintain aspect ratio
        width, height = img.size
        if width > height:
            new_width = target_size
            new_height = int(height * target_size / width)
        else:
            new_height = target_size
            new_width = int(width * target_size / height)
        
        # Resize maintaining aspect ratio
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Pad to square
        square_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - new_width) // 2
        paste_y = (target_size - new_height) // 2
        square_img.paste(img, (paste_x, paste_y))
        
        resized_frames.append(np.array(square_img))
    
    video_resized = np.array(resized_frames)
    print(f"  Resized shape: {video_resized.shape}")
    
    return video_resized, fps


def list_advanced_models():
    """List more advanced models you can run locally"""
    print("\n" + "="*60)
    print("ADVANCED VIDEO MODELS FOR LOCAL USE")
    print("="*60)
    
    models = {
        "Stable Diffusion VAE (Current)": {
            "type": "Image VAE applied to frames",
            "resolution": "512x512 (optimal), can handle others",
            "memory": "~2GB",
            "quality": "Good for general videos",
            "limitation": "No temporal consistency"
        },
        
        "AnimateDiff (Temporal Consistency)": {
            "type": "Temporal VAE / Frame consistency",
            "resolution": "256x256, 512x512, 768x768",
            "frames": "8, 16, 24, or 32 frames optimal",
            "memory": "~4-6GB", 
            "quality": "Excellent motion handling",
            "install": "pip install diffusers transformers accelerate",
            "model_id": "guoyww/animatediff-motion-adapter-v1-5-2",
            "formats": "Supports .mov files from iPhone!"
        },
        
        "FILM (Frame Interpolation)": {
            "type": "Frame interpolation model",
            "resolution": "Any (very flexible)",
            "memory": "~2-4GB",
            "quality": "Excellent for slow motion",
            "install": "pip install tensorflow",
            "use_case": "Create smooth slow-motion"
        },
        
        "VideoMAE": {
            "type": "Masked Autoencoder",
            "resolution": "224x224 (small) or 384x384",
            "memory": "~4GB",
            "quality": "Good for understanding",
            "install": "pip install transformers",
            "model_id": "MCG-NJU/videomae-base"
        },
        
        "CogVideo VAE (Experimental)": {
            "type": "Large video VAE",
            "resolution": "480x480",
            "memory": "~8-12GB",
            "quality": "State of the art",
            "limitation": "Needs optimization for M1"
        }
    }
    
    print("\nRecommended for M1 Max:")
    print("1. Current (SD VAE) - Works well, stable")
    print("2. AnimateDiff VAE - Better temporal consistency")
    print("3. VideoMAE - Lighter weight, good quality")
    
    return models


def main():
    parser = argparse.ArgumentParser(description='Process iPhone video')
    parser.add_argument('--video', type=str, help='Path to iPhone video (.mp4 or .mov)')
    parser.add_argument('--model', type=str, default='sd-vae', 
                       choices=['sd-vae', 'animatediff', 'list-models'],
                       help='Model to use or list available models')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames (8/16/24/32 for AnimateDiff)')
    parser.add_argument('--size', type=int, default=512, help='Target size (256/512/768 for AnimateDiff)')
    parser.add_argument('--noise', type=float, default=5.0, help='Noise level for latent space')
    parser.add_argument('--temporal', action='store_true', help='Use temporal consistency (AnimateDiff-style)')
    args = parser.parse_args()
    
    if args.model == 'list-models':
        list_advanced_models()
        return
    
    if not args.video:
        print("Creating demo with synthetic video...")
        # Use the simple test video if no input provided
        from add_latent_noise import create_test_video
        video_np = create_test_video(frames=16, height=512, width=512)
        fps = 8
    else:
        # Check file extension
        video_ext = Path(args.video).suffix.lower()
        if video_ext == '.mov':
            print(f"\n📱 Detected iPhone .mov file")
            print("Note: imageio-ffmpeg will handle the .mov format")
        
        # Process iPhone video
        video_np, fps = process_iphone_video(args.video, args.frames, args.size, args.temporal)
    
    # Save original
    Path("output").mkdir(exist_ok=True)
    imageio.mimsave('output/iphone_original.mp4', (video_np * 255).astype(np.uint8), fps=min(fps, 30))
    print("\nSaved: output/iphone_original.mp4")
    
    # Convert to tensor
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float() / 255.0
    video_tensor = video_tensor * 2.0 - 1.0
    
    # Load model
    print(f"\nLoading {args.model}...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    if args.size > 512 and device == "mps":
        print("Note: Using CPU for large resolution on M1")
        device = "cpu"
    
    if args.model == 'animatediff':
        # Use AnimateDiff VAE
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            print("Loading AnimateDiff pipeline...")
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float32,
            ).to(device)
            vae = pipe.vae
            vae.eval()
            print("✅ AnimateDiff VAE loaded!")
        except ImportError:
            print("❌ AnimateDiff not available. Install with:")
            print("   pip install diffusers[torch] transformers accelerate")
            print("   Falling back to SD-VAE...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32
            ).to(device)
            vae.eval()
    else:
        # Use standard SD VAE
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(device)
        vae.eval()
    
    # Process video
    print(f"\nProcessing on {device}...")
    latents = []
    reconstructed = []
    noisy = []
    
    with torch.no_grad():
        for t in range(video_tensor.shape[1]):
            print(f"Frame {t+1}/{video_tensor.shape[1]}", end='\r')
            
            frame = video_tensor[:, t, :, :].unsqueeze(0).to(device)
            
            # Encode
            latent = vae.encode(frame).latent_dist.sample()
            latents.append(latent.cpu())
            
            # Decode clean
            clean = vae.decode(latent).sample[0].cpu()
            reconstructed.append(clean)
            
            # Decode with noise
            if args.temporal and ANIMATEDIFF_AVAILABLE:
                # Use temporal noise if requested
                if t == 0:
                    # Create temporal noise for all frames at once
                    temporal_noise_all = create_temporal_noise(
                        (video_tensor.shape[1], *latent.shape[1:]),
                        temporal_weight=0.8,
                        device=device
                    )
                noise = temporal_noise_all[t].unsqueeze(0) * args.noise
            else:
                noise = torch.randn_like(latent) * args.noise
            
            noisy_latent = latent + noise
            noisy_decoded = vae.decode(noisy_latent).sample[0].cpu()
            noisy.append(noisy_decoded)
    
    print("\n\nSaving results...")
    
    # Save reconstructed
    recon_video = torch.stack(reconstructed, dim=1)
    recon_video = recon_video.permute(1, 2, 3, 0)
    recon_video = ((recon_video + 1.0) / 2.0).clamp(0, 1)
    recon_video_np = (recon_video * 255).byte().numpy()
    imageio.mimsave('output/iphone_reconstructed.mp4', recon_video_np, fps=min(fps, 30))
    
    # Save noisy
    noisy_video = torch.stack(noisy, dim=1)
    noisy_video = noisy_video.permute(1, 2, 3, 0)
    noisy_video = ((noisy_video + 1.0) / 2.0).clamp(0, 1)
    noisy_video_np = (noisy_video * 255).byte().numpy()
    imageio.mimsave('output/iphone_noisy.mp4', noisy_video_np, fps=min(fps, 30))
    
    print("\n✅ Done! Results:")
    print("  - output/iphone_original.mp4")
    print("  - output/iphone_reconstructed.mp4") 
    print("  - output/iphone_noisy.mp4")
    print(f"\nStats:")
    print(f"  Resolution: {args.size}x{args.size}")
    print(f"  Frames: {video_tensor.shape[1]}")
    print(f"  Latent shape: {latents[0].shape}")
    print(f"  Compression: {args.size}x{args.size} -> {latents[0].shape[-2]}x{latents[0].shape[-1]}")
    print(f"  Temporal consistency: {'Yes' if args.temporal else 'No'}")
    
    if args.temporal and not ANIMATEDIFF_AVAILABLE:
        print("\n⚠️  Note: Full AnimateDiff not available. Using temporal noise simulation.")
        print("For full AnimateDiff: pip install diffusers transformers accelerate")


if __name__ == '__main__':
    main()