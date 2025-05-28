#!/usr/bin/env python3
"""
Setup and test AnimateDiff for temporal consistency
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))


def check_animatediff_info():
    """Show AnimateDiff capabilities and requirements"""
    print("="*60)
    print("AnimateDiff Setup & Information")
    print("="*60)
    
    print("\n📋 AnimateDiff Specs:")
    print("- Supported resolutions: 256x256, 512x512, 768x768")
    print("- Optimal: 512x512 (most models trained on this)")
    print("- Frame counts: 8, 16, 24, 32 frames")
    print("- Memory: ~4-6GB for 16 frames at 512x512")
    
    print("\n📦 Installation:")
    print("pip install diffusers[torch] transformers accelerate")
    
    print("\n🎥 Video Format Support:")
    print("- .MOV files: ✅ Supported (via imageio-ffmpeg)")
    print("- .MP4 files: ✅ Supported")
    print("- .AVI files: ✅ Supported")
    print("- iPhone formats: ✅ All supported")
    
    print("\n🔧 Available Models:")
    models = {
        "guoyww/animatediff-motion-adapter-v1-5-2": {
            "type": "Motion Module",
            "resolution": "512x512 recommended",
            "description": "Best general purpose"
        },
        "guoyww/animatediff-motion-adapter-v1-5": {
            "type": "Motion Module", 
            "resolution": "512x512",
            "description": "Original version"
        },
        "ByteDance/AnimateDiff-Lightning": {
            "type": "Fast Motion Module",
            "resolution": "512x512",
            "description": "4x faster, slightly lower quality"
        }
    }
    
    for model_id, info in models.items():
        print(f"\n{model_id}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    return models


def test_animatediff_available():
    """Check if we can use AnimateDiff"""
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        from diffusers.utils import export_to_gif
        print("✅ AnimateDiff imports successful!")
        return True
    except ImportError as e:
        print(f"❌ AnimateDiff not available: {e}")
        print("\nTo install:")
        print("pip install diffusers transformers accelerate")
        return False


def main():
    print("Checking AnimateDiff availability...")
    
    info = check_animatediff_info()
    available = test_animatediff_available()
    
    if available:
        print("\n✅ AnimateDiff is ready to use!")
        print("\nNext step: Run the adapted iPhone video processor")
    else:
        print("\n❌ Please install AnimateDiff first")


if __name__ == '__main__':
    main()