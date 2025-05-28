#!/usr/bin/env python3
"""
Use Stable Video Diffusion (SVD) for better motion handling
SVD is specifically designed for video and handles motion much better than image-based VAEs
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image
import sys


def process_video_with_svd(video_path, output_dir="output_svd", num_frames=25, motion_bucket_id=127, noise_aug_strength=0.02):
    """
    Process video with Stable Video Diffusion
    
    SVD advantages:
    - Trained specifically on video data
    - Better temporal consistency
    - Handles camera motion and object motion well
    - Can generate 14-25 frames natively
    """
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video
    except ImportError:
        print("❌ SVD not available. Install with:")
        print("   pip install diffusers transformers accelerate")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load video and get first frame
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        frames.append(frame)
        if i >= 0:  # Just get first frame for SVD
            break
    reader.close()
    
    # Prepare first frame
    first_frame = Image.fromarray(frames[0])
    width, height = first_frame.size
    print(f"Original size: {width}x{height}")
    
    # SVD works best with certain resolutions
    # Native: 576x1024, but can handle others
    if width > height:
        new_width = 1024
        new_height = int(height * 1024 / width)
    else:
        new_height = 1024
        new_width = int(width * 1024 / height)
    
    # Make sure dimensions are divisible by 8
    new_width = (new_width // 8) * 8
    new_height = (new_height // 8) * 8
    
    first_frame = first_frame.resize((new_width, new_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {new_width}x{new_height}")
    
    # Load SVD
    print("\nLoading Stable Video Diffusion...")
    # Use CPU for SVD on M1 Mac to avoid memory issues
    device = "cpu"  # SVD has memory issues on MPS
    print(f"Using device: {device} (SVD works better on CPU for M1 Mac)")
    
    # Use img2vid-xt for better quality
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float32,
    )
    
    # Move to device
    pipe = pipe.to(device)
    
    # Enable memory optimizations
    pipe.enable_attention_slicing()
    
    # Enable CPU offload if available (but not on pure CPU mode)
    if device != "cpu" and hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    
    print("Memory optimizations enabled")
    
    print(f"\nGenerating {num_frames} frames with SVD...")
    print(f"Motion bucket: {motion_bucket_id} (0=still, 255=high motion)")
    print(f"Noise augmentation: {noise_aug_strength}")
    
    # Generate video
    # motion_bucket_id controls amount of motion (0-255)
    # noise_aug_strength adds variation
    generator = torch.manual_seed(42)
    
    print("Generating video (this may take several minutes on CPU)...")
    
    frames_generated = pipe(
        first_frame,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        decode_chunk_size=4,  # Smaller chunks for memory efficiency
        generator=generator,
        num_inference_steps=25,  # Reduce steps for faster generation
    ).frames[0]
    
    # Save as video
    export_to_video(frames_generated, f"{output_dir}/svd_generated.mp4", fps=7)
    print(f"\nSaved generated video: {output_dir}/svd_generated.mp4")
    
    # Also save as GIF
    frames_generated[0].save(
        f"{output_dir}/svd_generated.gif",
        save_all=True,
        append_images=frames_generated[1:],
        duration=1000/7,  # 7 fps
        loop=0
    )
    print(f"Saved as GIF: {output_dir}/svd_generated.gif")
    
    # Save comparison
    first_frame.save(f"{output_dir}/first_frame.png")
    print(f"Saved first frame: {output_dir}/first_frame.png")
    
    print("\n✅ SVD processing complete!")
    print(f"   Generated {len(frames_generated)} frames")
    print(f"   Resolution: {new_width}x{new_height}")
    print("\n💡 Tips:")
    print("   - Adjust motion_bucket_id (0-255) for motion amount")
    print("   - Try different noise_aug_strength (0.0-0.1) for variation")
    print("   - SVD excels at camera motion and object movement")


def process_with_extended_frames(video_path, output_dir="output_svd_extended", target_frames=50):
    """
    Generate longer videos by chaining SVD generations
    """
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video
    except ImportError:
        print("❌ SVD not available")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Generating extended video with {target_frames} frames...")
    print("This will chain multiple SVD generations")
    
    # Load model  
    device = "cpu"  # Use CPU for extended generation too
    print(f"Using device: {device}")
    
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float32,
    ).to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    if device != "cpu" and hasattr(pipe, 'enable_model_cpu_offload'):
        pipe.enable_model_cpu_offload()
    
    # Load first frame
    reader = imageio.get_reader(video_path)
    first_frame = Image.fromarray(reader.get_data(0))
    reader.close()
    
    # Resize
    first_frame = first_frame.resize((1024, 576), Image.Resampling.LANCZOS)
    
    all_frames = []
    current_frame = first_frame
    frames_generated = 0
    
    while frames_generated < target_frames:
        print(f"\nGenerating frames {frames_generated} to {frames_generated + 25}...")
        
        # Generate next batch
        frames = pipe(
            current_frame,
            num_frames=25,
            motion_bucket_id=127,
            decode_chunk_size=8,
        ).frames[0]
        
        # Add frames (skip first if not first batch to avoid duplicates)
        start_idx = 1 if frames_generated > 0 else 0
        all_frames.extend(frames[start_idx:])
        
        # Use last frame as next starting point
        current_frame = frames[-1]
        frames_generated = len(all_frames)
        
        if frames_generated >= target_frames:
            all_frames = all_frames[:target_frames]
            break
    
    # Save extended video
    export_to_video(all_frames, f"{output_dir}/svd_extended_{target_frames}frames.mp4", fps=7)
    print(f"\n✅ Generated {len(all_frames)} frames!")
    print(f"Saved: {output_dir}/svd_extended_{target_frames}frames.mp4")


def main():
    parser = argparse.ArgumentParser(description='Stable Video Diffusion - Better motion handling')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--frames', type=int, default=25, help='Number of frames (14-25 for single gen)')
    parser.add_argument('--motion', type=int, default=127, help='Motion amount (0-255)')
    parser.add_argument('--noise', type=float, default=0.02, help='Noise augmentation (0.0-0.1)')
    parser.add_argument('--extended', type=int, help='Generate extended video with N frames')
    parser.add_argument('--output', type=str, default='output_svd', help='Output directory')
    args = parser.parse_args()
    
    if args.extended:
        process_with_extended_frames(args.video, args.output, args.extended)
    else:
        process_video_with_svd(
            args.video, 
            args.output,
            args.frames,
            args.motion,
            args.noise
        )


if __name__ == '__main__':
    main()