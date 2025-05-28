#!/usr/bin/env python3
"""
Use ZeroScope V2 - More memory efficient video model for M1 Mac
Better than AnimateDiff for motion, lighter than SVD
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image


def process_with_zeroscope(video_path, output_dir="output_zeroscope", num_frames=24):
    """
    Use ZeroScope V2 - efficient video model that works well on M1 Mac
    """
    try:
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        print("❌ Diffusers not available. Install with:")
        print("   pip install diffusers transformers accelerate")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Generating video with ZeroScope V2...")
    print(f"Note: ZeroScope generates new content rather than processing input video")
    
    # Load ZeroScope
    print("\nLoading ZeroScope V2...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    print(f"\nGenerating {num_frames} frames with ZeroScope...")
    
    # Generic prompt for good video quality
    prompt = "high quality video, smooth motion, detailed, cinematic"
    
    # Generate video
    generator = torch.manual_seed(42)
    
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=320,
        width=576,
        num_inference_steps=20,
        generator=generator,
    )
    
    frames = result.frames[0]
    
    # Save video
    export_to_video(frames, f"{output_dir}/zeroscope_generated.mp4", fps=8)
    print(f"\nSaved generated video: {output_dir}/zeroscope_generated.mp4")
    
    # Save as GIF
    frames[0].save(
        f"{output_dir}/zeroscope_generated.gif",
        save_all=True,
        append_images=frames[1:],
        duration=125,  # 8 fps
        loop=0
    )
    print(f"Saved as GIF: {output_dir}/zeroscope_generated.gif")
    
    print("\n✅ ZeroScope processing complete!")
    print(f"   Generated {len(frames)} frames")
    print(f"   Resolution: 576x320")


def process_with_custom_prompt(prompt, output_dir="output_zeroscope", num_frames=24):
    """Generate video from custom text prompt"""
    try:
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        print("❌ Diffusers not available")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Generating video from prompt: '{prompt}'")
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    pipe = DiffusionPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",
        torch_dtype=torch.float32,
    ).to(device)
    
    pipe.enable_attention_slicing()
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    generator = torch.manual_seed(42)
    
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=320,
        width=576,
        num_inference_steps=20,
        generator=generator,
    )
    
    frames = result.frames[0]
    
    # Save outputs
    export_to_video(frames, f"{output_dir}/custom_prompt.mp4", fps=8)
    print(f"Saved: {output_dir}/custom_prompt.mp4")


def main():
    parser = argparse.ArgumentParser(description='ZeroScope V2 - Efficient video generation')
    parser.add_argument('--video', type=str, help='Input video path (for reference)')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--frames', type=int, default=24, help='Number of frames (8-64)')
    parser.add_argument('--output', type=str, default='output_zeroscope', help='Output directory')
    args = parser.parse_args()
    
    if args.prompt:
        process_with_custom_prompt(args.prompt, args.output, args.frames)
    elif args.video:
        process_with_zeroscope(args.video, args.output, args.frames)
    else:
        print("ZeroScope V2 Video Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  From prompt: python use_zeroscope_v2.py --prompt 'A cat playing piano'")
        print("  Quick test:  python use_zeroscope_v2.py --video dummy.mov")
        print("\nFeatures:")
        print("  - Memory efficient (works well on M1 Mac)")
        print("  - Good motion quality")
        print("  - Fast generation")
        print("  - Native resolution: 576x320 (widescreen)")
        print("  - Up to 64 frames")


if __name__ == '__main__':
    main()