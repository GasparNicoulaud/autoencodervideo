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
    
    ZeroScope advantages:
    - More memory efficient than SVD
    - Better motion than AnimateDiff
    - Good quality/speed balance
    - Works well on MPS
    """
    try:
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError:
        print("❌ Diffusers not available. Install with:")
        print("   pip install diffusers transformers accelerate")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load video and get first frame
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    first_frame = reader.get_data(0)
    reader.close()
    
    # Prepare first frame for ZeroScope (576x320 native)
    first_frame_pil = Image.fromarray(first_frame)
    width, height = first_frame_pil.size
    print(f"Original size: {width}x{height}")
    
    # ZeroScope native resolution is 576x320 (widescreen)
    if width > height:
        # Landscape - resize to 576x320
        target_width, target_height = 576, 320
    else:
        # Portrait - resize to 320x576
        target_width, target_height = 320, 576
    
    first_frame_pil = first_frame_pil.resize((target_width, target_height), Image.Resampling.LANCZOS)
    print(f"Resized to: {target_width}x{target_height}")
    
    # Save input frame
    first_frame_pil.save(f"{output_dir}/input_frame.png")
    print(f"Saved input frame: {output_dir}/input_frame.png")\n    \n    # Load ZeroScope\n    print(\"\\nLoading ZeroScope V2...\")\n    device = \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n    print(f\"Using device: {device}\")\n    \n    # Load the model\n    pipe = DiffusionPipeline.from_pretrained(\n        \"cerspense/zeroscope_v2_576w\",\n        torch_dtype=torch.float32,\n    )\n    pipe = pipe.to(device)\n    \n    # Memory optimizations\n    pipe.enable_attention_slicing()\n    if hasattr(pipe, 'enable_vae_slicing'):\n        pipe.enable_vae_slicing()\n    \n    print(f\"\\nGenerating {num_frames} frames with ZeroScope...\")\n    \n    # For ZeroScope, we need to use text prompt since it's text-to-video\n    # We'll use a generic prompt that should match most videos\n    prompt = \"high quality video, smooth motion, detailed\"\n    \n    # Generate video\n    generator = torch.manual_seed(42)\n    \n    result = pipe(\n        prompt=prompt,\n        num_frames=num_frames,\n        height=target_height,\n        width=target_width,\n        num_inference_steps=20,  # Reasonable quality/speed\n        generator=generator,\n    )\n    \n    frames = result.frames[0]\n    \n    # Save video\n    export_to_video(frames, f\"{output_dir}/zeroscope_generated.mp4\", fps=8)\n    print(f\"\\nSaved generated video: {output_dir}/zeroscope_generated.mp4\")\n    \n    # Save as GIF\n    frames[0].save(\n        f\"{output_dir}/zeroscope_generated.gif\",\n        save_all=True,\n        append_images=frames[1:],\n        duration=125,  # 8 fps\n        loop=0\n    )\n    print(f\"Saved as GIF: {output_dir}/zeroscope_generated.gif\")\n    \n    print(\"\\n✅ ZeroScope processing complete!\")\n    print(f\"   Generated {len(frames)} frames\")\n    print(f\"   Resolution: {target_width}x{target_height}\")\n    print(\"\\n💡 Note: ZeroScope is text-to-video, so it generates new content\")\n    print(\"   rather than processing your input video directly.\")\n\n\ndef process_with_custom_prompt(prompt, output_dir=\"output_zeroscope\", num_frames=24, width=576, height=320):\n    \"\"\"Generate video from custom text prompt\"\"\"\n    try:\n        from diffusers import DiffusionPipeline\n        from diffusers.utils import export_to_video\n    except ImportError:\n        print(\"❌ Diffusers not available\")\n        return\n    \n    Path(output_dir).mkdir(exist_ok=True)\n    \n    print(f\"Generating video from prompt: '{prompt}'\")\n    \n    device = \"mps\" if torch.backends.mps.is_available() else \"cpu\"\n    \n    pipe = DiffusionPipeline.from_pretrained(\n        \"cerspense/zeroscope_v2_576w\",\n        torch_dtype=torch.float32,\n    ).to(device)\n    \n    pipe.enable_attention_slicing()\n    if hasattr(pipe, 'enable_vae_slicing'):\n        pipe.enable_vae_slicing()\n    \n    generator = torch.manual_seed(42)\n    \n    result = pipe(\n        prompt=prompt,\n        num_frames=num_frames,\n        height=height,\n        width=width,\n        num_inference_steps=20,\n        generator=generator,\n    )\n    \n    frames = result.frames[0]\n    \n    # Save outputs\n    export_to_video(frames, f\"{output_dir}/custom_prompt.mp4\", fps=8)\n    print(f\"Saved: {output_dir}/custom_prompt.mp4\")\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='ZeroScope V2 - Efficient video generation')\n    parser.add_argument('--video', type=str, help='Input video path (for reference)')\n    parser.add_argument('--prompt', type=str, help='Text prompt for generation')\n    parser.add_argument('--frames', type=int, default=24, help='Number of frames (8-64)')\n    parser.add_argument('--output', type=str, default='output_zeroscope', help='Output directory')\n    args = parser.parse_args()\n    \n    if args.prompt:\n        process_with_custom_prompt(args.prompt, args.output, args.frames)\n    elif args.video:\n        process_with_zeroscope(args.video, args.output, args.frames)\n    else:\n        print(\"ZeroScope V2 Video Tool\")\n        print(\"=\" * 50)\n        print(\"\\nUsage:\")\n        print(\"  From video:  python use_zeroscope.py --video input.mov\")\n        print(\"  From prompt: python use_zeroscope.py --prompt 'A cat playing piano'\")\n        print(\"\\nFeatures:\")\n        print(\"  - Memory efficient (works well on M1 Mac)\")\n        print(\"  - Good motion quality\")\n        print(\"  - Fast generation\")\n        print(\"  - Native resolution: 576x320 (widescreen)\")\n        print(\"  - Up to 64 frames\")\n\n\nif __name__ == '__main__':\n    main()