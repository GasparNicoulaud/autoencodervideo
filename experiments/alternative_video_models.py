#!/usr/bin/env python3
"""
Alternative video models with STRONG uniform noise for semantic glitches
Focused experiment using models that work well on M1 Max 32GB
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image
import sys
sys.path.append('.')
from src.utils.video_io import load_video, save_video


def list_alternative_models():
    """List alternative video models optimized for different use cases"""
    
    print("\n" + "="*70)
    print("ALTERNATIVE VIDEO MODELS FOR M1 MAC")
    print("="*70)
    
    models = {
        "🎯 Text2Video-Zero": {
            "type": "Zero-shot text-to-video",
            "frames": "Up to 64 frames",
            "memory": "~4-6GB (very efficient!)",
            "strengths": "No training required, works with any SD model",
            "install": "pip install diffusers transformers",
            "usage": "Can convert ANY Stable Diffusion model to video",
            "code": """
from diffusers import TextToVideoZeroPipeline
pipe = TextToVideoZeroPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32
).to("mps")
result = pipe(prompt="cat walking", num_frames=24)
"""
        },
        
        "🔄 ControlNet Video": {
            "type": "Controlled video generation", 
            "frames": "Flexible, process frame by frame",
            "memory": "~6-8GB",
            "strengths": "Precise control using pose/depth/edge",
            "install": "pip install controlnet-aux diffusers",
            "usage": "Extract poses from your video, generate new content",
        },
        
        "🎬 Hotshot-XL": {
            "type": "Text-to-GIF model",
            "frames": "8 frames (optimized for GIFs)",
            "memory": "~8-10GB",
            "model_id": "hotshotco/Hotshot-XL",
            "strengths": "Fast, good for short loops",
            "install": "pip install diffusers",
        },
        
        "🌊 DynamiCrafter": {
            "type": "Image animation",
            "frames": "16-32 frames",
            "memory": "~10-12GB",
            "strengths": "Animates still images with text prompts",
            "note": "Good for adding motion to photos",
        },
        
        "⚡ LaVie": {
            "type": "Text-to-video",
            "frames": "16-61 frames",
            "memory": "~8GB",
            "strengths": "Good quality/speed ratio",
            "model_id": "Vchitect/LaVie",
        },
        
        "🎪 Show-1": {
            "type": "Text-to-video",
            "frames": "Up to 128 frames",
            "memory": "~12-16GB",
            "strengths": "Long videos, good consistency",
            "model_id": "showlab/show-1",
        },
        
        "🚀 VideoFusion": {
            "type": "Diffusion-based video model",
            "frames": "16-64 frames",
            "memory": "~6-10GB",
            "strengths": "Efficient architecture",
        },
        
        "💡 SEINE": {
            "type": "Short video generation",
            "frames": "16-24 frames",
            "memory": "~6GB",
            "strengths": "Very memory efficient",
            "note": "Good for M1 Mac",
        }
    }
    
    print("\n🌟 RECOMMENDED FOR YOUR USE CASE (M1 Max 32GB):\n")
    
    print("1. **Text2Video-Zero** (Most Flexible)")
    print("   - Works with ANY Stable Diffusion model")
    print("   - Very memory efficient")
    print("   - Can use custom models for specific styles")
    
    print("\n2. **Hotshot-XL** (For Quick Tests)")
    print("   - Optimized for short animations")
    print("   - Fast generation")
    print("   - Good for testing ideas")
    
    print("\n3. **ControlNet Video** (For Your Video)")
    print("   - Can extract poses/edges from your iPhone video")
    print("   - Generate new content following the same motion")
    print("   - Very flexible control")
    
    for name, info in models.items():
        print(f"\n{name}")
        print("-" * len(name))
        for key, value in info.items():
            if key == "code":
                continue
            print(f"  {key}: {value}")
        if "code" in info:
            print(f"  Example code:")
            print(info["code"])
    
    return models


def try_text2video_zero(prompt="A cat walking on the beach", num_frames=24, output_dir="output_t2v_zero"):
    """
    Use Text2Video-Zero - Very efficient, works with any SD model
    """
    try:
        from diffusers import TextToVideoZeroPipeline
    except ImportError:
        print("❌ Text2Video-Zero not available. Install with:")
        print("   pip install diffusers transformers")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading Text2Video-Zero (very memory efficient!)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Can use any SD model!
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = TextToVideoZeroPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)
    
    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    
    print(f"\nGenerating {num_frames} frames...")
    print(f"Prompt: {prompt}")
    
    # Generate video
    result = pipe(
        prompt=prompt,
        num_frames=num_frames,
        height=512,
        width=512,
        num_inference_steps=25,
    ).frames
    
    # Save video
    video_path = f"{output_dir}/text2video_zero.mp4"
    result[0].save(
        f"{output_dir}/text2video_zero.gif",
        save_all=True,
        append_images=result[1:],
        duration=100,
        loop=0
    )
    
    print(f"\n✅ Generated video saved to: {output_dir}/text2video_zero.gif")
    print("Text2Video-Zero advantages:")
    print("  - Works with ANY Stable Diffusion model")
    print("  - Very memory efficient")
    print("  - Can use LoRAs and custom models")


def try_hotshot_xl(prompt="A beautiful sunset", output_dir="output_hotshot"):
    """
    Hotshot-XL - Optimized for GIF generation
    """
    try:
        from diffusers import DiffusionPipeline
    except ImportError:
        print("❌ Diffusers not available")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading Hotshot-XL (optimized for GIFs)...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    pipe = DiffusionPipeline.from_pretrained(
        "hotshotco/Hotshot-XL",
        torch_dtype=torch.float32,
    ).to(device)
    
    pipe.enable_attention_slicing()
    
    # Generate 8-frame GIF
    result = pipe(
        prompt=prompt,
        num_frames=8,
        width=672,
        height=384,
        num_inference_steps=20,
    )
    
    # Save
    result.frames[0].save(
        f"{output_dir}/hotshot.gif",
        save_all=True,
        append_images=result.frames[1:],
        duration=125,
        loop=0
    )
    
    print(f"Saved: {output_dir}/hotshot.gif")


def process_with_controlnet_video(video_path, output_dir="output_controlnet"):
    """
    Use ControlNet to extract motion and generate new content
    """
    try:
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
        from controlnet_aux import OpenposeDetector
    except ImportError:
        print("❌ ControlNet not available. Install with:")
        print("   pip install diffusers controlnet-aux")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Using ControlNet for video processing...")
    print("This extracts poses from your video and generates new content")
    
    # Load video
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= 16:  # Process first 16 frames
            break
        frames.append(frame)
    reader.close()
    
    # Extract poses
    print("Extracting poses from video...")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    
    pose_images = []
    for frame in frames:
        pose = openpose(Image.fromarray(frame))
        pose_images.append(pose)
    
    # Save pose video
    pose_images[0].save(
        f"{output_dir}/extracted_poses.gif",
        save_all=True,
        append_images=pose_images[1:],
        duration=100,
        loop=0
    )
    print(f"Saved poses: {output_dir}/extracted_poses.gif")
    
    # Load ControlNet
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float32,
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float32,
    ).to(device)
    
    pipe.enable_attention_slicing()
    
    # Generate new video following poses
    print("Generating new video following extracted motion...")
    generated_frames = []
    
    prompt = "a robot dancing, high quality, detailed"
    
    for i, pose in enumerate(pose_images):
        print(f"Frame {i+1}/{len(pose_images)}", end='\r')
        
        result = pipe(
            prompt=prompt,
            image=pose,
            num_inference_steps=20,
        ).images[0]
        
        generated_frames.append(result)
    
    # Save generated video
    generated_frames[0].save(
        f"{output_dir}/controlnet_generated.gif",
        save_all=True,
        append_images=generated_frames[1:],
        duration=100,
        loop=0
    )
    
    print(f"\n✅ Generated video: {output_dir}/controlnet_generated.gif")
    print("ControlNet allows you to:")
    print("  - Extract motion from any video")
    print("  - Generate new content following that motion")
    print("  - Use different control types (pose, depth, edge)")


def main():
    parser = argparse.ArgumentParser(description='Alternative video models')
    parser.add_argument('--list', action='store_true', help='List all models')
    parser.add_argument('--model', type=str, help='Model to try (t2v-zero, hotshot, controlnet)')
    parser.add_argument('--video', type=str, help='Input video (for controlnet)')
    parser.add_argument('--prompt', type=str, help='Text prompt')
    parser.add_argument('--frames', type=int, default=24, help='Number of frames')
    args = parser.parse_args()
    
    if args.list:
        list_alternative_models()
    elif args.model == "t2v-zero":
        prompt = args.prompt or "A cat walking on the beach"
        try_text2video_zero(prompt, args.frames)
    elif args.model == "hotshot":
        prompt = args.prompt or "A beautiful sunset"
        try_hotshot_xl(prompt)
    elif args.model == "controlnet" and args.video:
        process_with_controlnet_video(args.video)
    else:
        print("Alternative Video Models Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  List models:      python alternative_video_models.py --list")
        print("  Text2Video-Zero:  python alternative_video_models.py --model t2v-zero --prompt 'Your prompt'")
        print("  Hotshot-XL:       python alternative_video_models.py --model hotshot --prompt 'Your prompt'")  
        print("  ControlNet:       python alternative_video_models.py --model controlnet --video input.mov")
        print("\nRecommended to try first:")
        print("  1. Text2Video-Zero (most memory efficient)")
        print("  2. ControlNet (can use your video's motion)")


if __name__ == '__main__':
    main()