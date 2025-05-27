#!/usr/bin/env python3
"""
Use a real VAE from Stable Diffusion on video frames
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.huggingface_models import HuggingFaceVideoAE
from src.utils import save_video, create_video_grid
from generate_colorful_samples import generate_colorful_checkerboard_video


def demo_real_vae():
    """Demo using Stable Diffusion's VAE on video frames"""
    
    print("This demo will use Stable Diffusion's VAE")
    print("First, we need to install diffusers...")
    
    # Try to import diffusers
    try:
        from diffusers import AutoencoderKL
        print("✓ diffusers is installed")
    except ImportError:
        print("✗ diffusers not found")
        print("\nTo use real models, install with:")
        print("pip install diffusers")
        print("\nFor now, let's summarize what you've learned...")
        summarize_learning()
        return
    
    print("\nLoading Stable Diffusion VAE...")
    print("This will download ~335MB on first run")
    
    try:
        # Load the VAE
        vae = HuggingFaceVideoAE.load_simple_vae_for_frames()
        
        if vae is None:
            print("Failed to load VAE")
            summarize_learning()
            return
            
        print("✓ VAE loaded successfully!")
        
        # Generate test video
        print("\nGenerating colorful test video...")
        test_video = generate_colorful_checkerboard_video(
            frames=8, 
            height=512,  # SD VAE expects 512x512
            width=512, 
            block_size=64
        )
        test_video = test_video.unsqueeze(0).cuda() if torch.cuda.is_available() else test_video.unsqueeze(0)
        
        print("Encoding video with real VAE...")
        with torch.no_grad():
            z, mu, logvar = vae.encode(test_video)
            print(f"Latent shape: {z.shape}")
            
            # Decode back
            reconstructed = vae.decode(z)
            
        save_video(test_video, "output/real_vae_original.mp4")
        save_video(reconstructed, "output/real_vae_reconstruction.mp4")
        
        print("\n✅ Success with real VAE!")
        print("Generated:")
        print("- output/real_vae_original.mp4")
        print("- output/real_vae_reconstruction.mp4")
        
    except Exception as e:
        print(f"Error: {e}")
        summarize_learning()


def summarize_learning():
    """Summarize what we've learned about video autoencoders"""
    
    print("\n" + "="*60)
    print("SUMMARY: Video Autoencoders Journey")
    print("="*60)
    
    print("\n1. WHAT WE BUILT:")
    print("   ✓ Complete framework for video autoencoder experimentation")
    print("   ✓ Models: VAE and VQ-VAE architectures")
    print("   ✓ Latent space manipulation tools")
    print("   ✓ Interpolation methods")
    print("   ✓ Programmatic weight control")
    
    print("\n2. WHAT WE TRAINED:")
    print("   ✓ Simple autoencoder on synthetic patterns")
    print("   ✓ Saw how latent space manipulation affects output")
    print("   ✓ Demonstrated interpolation and scaling")
    
    print("\n3. REAL MODELS AVAILABLE:")
    print("   - CogVideoX VAE (state-of-the-art)")
    print("   - Stable Video Diffusion VAE")
    print("   - MAGVIT2 (VQ-VAE)")
    print("   - Or use image VAEs frame-by-frame")
    
    print("\n4. KEY INSIGHTS:")
    print("   - Autoencoders compress videos to latent representations")
    print("   - You can manipulate these latents programmatically")
    print("   - Real models need real training data (not just patterns)")
    print("   - Pre-trained models save weeks of training time")
    
    print("\n5. NEXT STEPS:")
    print("   - Install diffusers: pip install diffusers")
    print("   - Try real models from Hugging Face")
    print("   - Train on your own video dataset")
    print("   - Experiment with latent space control on real videos")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    demo_real_vae()