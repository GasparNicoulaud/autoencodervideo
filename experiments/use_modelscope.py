#!/usr/bin/env python3
"""
Use ModelScope Text-to-Video - Good balance of quality and efficiency
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image


def generate_with_modelscope(prompt="A beautiful nature scene", num_frames=16, output_dir="output_modelscope"):
    """
    ModelScope Text-to-Video - works well on M1 Mac
    """
    try:
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        print("❌ Diffusers not available")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading ModelScope Text-to-Video...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load ModelScope
    pipe = DiffusionPipeline.from_pretrained(
        "ali-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float32,
    )
    pipe = pipe.to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print(f"\nGenerating {num_frames} frames...")
    print(f"Prompt: {prompt}")
    
    # Generate video
    video_frames = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=256,
        width=256,
        num_inference_steps=25,
    ).frames
    
    # Save video
    export_to_video(video_frames, f"{output_dir}/modelscope_video.mp4", fps=8)
    print(f"Saved video: {output_dir}/modelscope_video.mp4")
    
    # Save as GIF
    video_frames[0].save(
        f"{output_dir}/modelscope_video.gif",
        save_all=True,
        append_images=video_frames[1:],
        duration=125,
        loop=0
    )
    print(f"Saved GIF: {output_dir}/modelscope_video.gif")
    
    print("\n✅ ModelScope generation complete!")
    print(f"   Frames: {len(video_frames)}")
    print(f"   Resolution: 256x256")
    print("   Note: ModelScope works well up to 32-48 frames")


def main():
    parser = argparse.ArgumentParser(description='ModelScope Text-to-Video')
    parser.add_argument('--prompt', type=str, default="A beautiful nature scene", help='Text prompt')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames (8-48)')
    parser.add_argument('--output', type=str, default='output_modelscope', help='Output directory')
    args = parser.parse_args()
    
    generate_with_modelscope(args.prompt, args.frames, args.output)


if __name__ == '__main__':
    main()