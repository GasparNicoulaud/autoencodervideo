#!/usr/bin/env python3
"""
Advanced video models that handle movement better and support more frames
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image


def list_advanced_models():
    """List advanced video models with better motion handling"""
    models = {
        "🎬 AnimateDiff v3 (Motion Director)": {
            "frames": "Up to 64 frames",
            "resolution": "Up to 1024x1024",
            "memory": "~8-16GB",
            "model_id": "guoyww/animatediff-motion-director-v1-5-3",
            "strengths": "Camera control, better motion coherence",
            "install": "pip install diffusers transformers",
        },
        
        "🎥 Stable Video Diffusion (SVD)": {
            "frames": "14-25 frames standard, up to 100+ with extensions",
            "resolution": "576x1024 (native), works with others",
            "memory": "~12-16GB",
            "model_id": "stabilityai/stable-video-diffusion-img2vid-xt",
            "strengths": "Excellent motion quality, temporal consistency",
            "install": "pip install diffusers transformers accelerate",
            "note": "Best for image-to-video generation"
        },
        
        "🚀 ModelScope Text2Video": {
            "frames": "Up to 64 frames",
            "resolution": "256x256 to 1024x1024",
            "memory": "~8-12GB",
            "model_id": "ali-vilab/text-to-video-ms-1.7b",
            "strengths": "Long video generation, good motion",
            "install": "pip install modelscope diffusers",
        },
        
        "🎯 ZeroScope V2": {
            "frames": "Up to 64 frames",
            "resolution": "576x320, 1024x576",
            "memory": "~6-10GB", 
            "model_id": "cerspense/zeroscope_v2_576w",
            "strengths": "Efficient, good quality/speed ratio",
            "install": "pip install diffusers torch",
        },
        
        "🔥 CogVideoX": {
            "frames": "Up to 49 frames (6 seconds)",
            "resolution": "480x720, 720x480",
            "memory": "~20-30GB (5B model)",
            "model_id": "THUDM/CogVideoX-5b",
            "strengths": "State-of-art quality, long coherent videos",
            "install": "pip install diffusers transformers accelerate",
            "note": "Very large model, but excellent results"
        },
        
        "💫 I2VGen-XL": {
            "frames": "Up to 128 frames",
            "resolution": "Multiple resolutions up to 1280x720",
            "memory": "~12-16GB",
            "model_id": "ali-vilab/i2vgen-xl",
            "strengths": "Image-to-video, very long sequences",
            "install": "pip install diffusers",
        },
        
        "🎪 Tune-A-Video": {
            "frames": "24-48 frames typical",
            "resolution": "512x512",
            "memory": "~6-8GB",
            "model_id": "CompVis/stable-diffusion-v1-4",
            "strengths": "One-shot video tuning, style transfer",
            "install": "pip install tune-a-video",
        }
    }
    
    print("\n" + "="*70)
    print("ADVANCED VIDEO MODELS - Better Motion & More Frames")
    print("="*70)
    
    for name, info in models.items():
        print(f"\n{name}")
        print("-" * len(name))
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR USE CASE (M1 Max 32GB)")
    print("="*70)
    
    print("\n1. For BEST motion quality with more frames:")
    print("   → Stable Video Diffusion (SVD)")
    print("   → Excellent temporal consistency")
    print("   → Native 25 frames, extendable to 100+")
    
    print("\n2. For LONGEST videos (64+ frames):")
    print("   → I2VGen-XL (up to 128 frames)")
    print("   → ModelScope (up to 64 frames)")
    
    print("\n3. For HIGHEST quality (if you can handle 20GB+):")
    print("   → CogVideoX-5b")
    print("   → State-of-art results")
    
    print("\n4. For BEST balance on M1 Max:")
    print("   → ZeroScope V2 (efficient)")
    print("   → AnimateDiff v3 (good control)")
    
    return models


def process_with_svd(video_path, output_dir="output_svd"):
    """
    Process video with Stable Video Diffusion
    Better motion handling than AnimateDiff
    """
    try:
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video
    except ImportError:
        print("❌ SVD not available. Install with:")
        print("   pip install diffusers transformers accelerate")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading Stable Video Diffusion...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float32,
        variant="fp16" if device == "cuda" else None
    )
    pipe = pipe.to(device)
    
    # Load first frame from video
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()
    
    # Prepare image
    image = Image.fromarray(first_frame)
    image = image.resize((1024, 576))  # SVD native resolution
    
    print("Generating video with SVD...")
    frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
    
    export_to_video(frames, f"{output_dir}/svd_output.mp4", fps=7)
    print(f"Saved: {output_dir}/svd_output.mp4")


def process_with_cogvideo(text_prompt="A beautiful nature scene", output_dir="output_cogvideo"):
    """
    Use CogVideoX for high-quality video generation
    Note: This is a large model (5B parameters)
    """
    try:
        from diffusers import CogVideoXPipeline
    except ImportError:
        print("❌ CogVideoX not available. Install with:")
        print("   pip install diffusers transformers accelerate")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading CogVideoX (this is a large model)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Use the 2B model for M1 Mac (smaller than 5B)
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float32
    )
    pipe = pipe.to(device)
    
    print(f"Generating video: '{text_prompt}'")
    video = pipe(
        prompt=text_prompt,
        num_frames=49,  # 6 seconds at 8fps
        guidance_scale=6,
        num_inference_steps=50
    ).frames[0]
    
    # Save video
    imageio.mimsave(f"{output_dir}/cogvideo_output.mp4", video, fps=8)
    print(f"Saved: {output_dir}/cogvideo_output.mp4")


def install_model(model_name):
    """Helper to install specific models"""
    install_commands = {
        "svd": "pip install diffusers transformers accelerate",
        "cogvideo": "pip install diffusers transformers accelerate",
        "modelscope": "pip install modelscope diffusers",
        "zeroscope": "pip install diffusers torch",
        "i2vgen": "pip install diffusers transformers",
    }
    
    if model_name in install_commands:
        print(f"\nTo install {model_name}:")
        print(f"   {install_commands[model_name]}")
    else:
        print(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Advanced video models explorer')
    parser.add_argument('--list', action='store_true', help='List all advanced models')
    parser.add_argument('--model', type=str, help='Model to use (svd, cogvideo, etc.)')
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--prompt', type=str, help='Text prompt for generation')
    parser.add_argument('--install', type=str, help='Show install command for model')
    args = parser.parse_args()
    
    if args.list:
        list_advanced_models()
    elif args.install:
        install_model(args.install)
    elif args.model == "svd" and args.video:
        process_with_svd(args.video)
    elif args.model == "cogvideo" and args.prompt:
        process_with_cogvideo(args.prompt)
    else:
        print("Advanced Video Models Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  List models:     python advanced_video_models.py --list")
        print("  Install help:    python advanced_video_models.py --install svd")
        print("  Process (SVD):   python advanced_video_models.py --model svd --video input.mov")
        print("  Generate (Cog):  python advanced_video_models.py --model cogvideo --prompt 'A cat'")
        print("\nFor your use case (movement + more frames), try:")
        print("  1. Stable Video Diffusion (SVD) - Best motion")
        print("  2. I2VGen-XL - Up to 128 frames")
        print("  3. CogVideoX - Highest quality")


if __name__ == '__main__':
    main()