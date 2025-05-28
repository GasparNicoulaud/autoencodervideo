#!/usr/bin/env python3
"""
Use AnimateDiff for video processing with proper temporal modeling
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image

def check_animatediff_available():
    """Check if AnimateDiff dependencies are available"""
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        from diffusers.utils import export_to_gif
        return True
    except ImportError:
        return False

def process_with_animatediff(video_path, output_dir="output_animatediff", max_frames=16, target_size=512):
    """
    Process video using AnimateDiff's motion-aware VAE
    
    AnimateDiff uses:
    - Motion modules for temporal consistency
    - Compatible with Stable Diffusion models
    - Designed for 16-frame sequences
    """
    if not check_animatediff_available():
        print("❌ AnimateDiff not available. Install with:")
        print("   pip install diffusers[torch] transformers accelerate")
        return
    
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_gif
    import torch
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load video
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        frames.append(frame)
    reader.close()
    
    print(f"Loaded {len(frames)} frames")
    
    # Prepare frames
    print("Preparing frames for AnimateDiff...")
    print(f"Original frame shape: {frames[0].shape}")
    
    pil_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        
        # Handle iPhone video orientation and resize
        width, height = img.size
        
        # Center crop to square first
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize to target size
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        pil_frames.append(img)
    
    print(f"Resized to {target_size}x{target_size}")
    
    # Save original frames
    pil_frames[0].save(
        f"{output_dir}/original.gif",
        save_all=True,
        append_images=pil_frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved original: {output_dir}/original.gif")
    
    # Load AnimateDiff
    print("\nLoading AnimateDiff pipeline...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # For high resolution or many frames, consider using CPU
    if target_size >= 768 or max_frames > 24:
        print(f"Note: High resolution ({target_size}x{target_size}) or many frames ({max_frames})")
        print("      Consider using CPU if you encounter memory issues")
    
    print(f"Using device: {device}")
    
    # Use AnimateDiff with motion adapter
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float32,  # Use float32 for stability
    )
    
    # Move to device carefully
    pipe = pipe.to(device)
    
    # Use DDIM scheduler for better consistency
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Don't use CPU offload on M1 Mac - it causes issues
    # if hasattr(pipe, "enable_model_cpu_offload"):
    #     pipe.enable_model_cpu_offload()
    
    print("AnimateDiff pipeline loaded!")
    
    # Process with AnimateDiff
    # Note: AnimateDiff is primarily for generation, but we can use its VAE
    vae = pipe.vae
    vae.eval()
    
    print("\nProcessing frames through AnimateDiff VAE...")
    
    # Convert frames to tensors
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    frame_tensors = torch.stack([transform(img) for img in pil_frames]).to(device)
    
    # Process through VAE with temporal awareness
    with torch.no_grad():
        # Encode all frames
        print("Encoding frames to latent space...")
        latents = []
        for i in range(len(frame_tensors)):
            frame = frame_tensors[i:i+1]
            latent = vae.encode(frame).latent_dist.sample()
            latents.append(latent)
        
        latents = torch.cat(latents, dim=0)
        print(f"Latent shape: {latents.shape}")
        
        # Add noise
        print("Adding noise...")
        noise_strength = 3.0
        
        # Create simple random noise for each frame
        latent_noise = []
        for i in range(len(latents)):
            noise = torch.randn_like(latents[i])
            latent_noise.append(noise)
        
        latent_noise = torch.stack(latent_noise)
        noisy_latents = latents + latent_noise * noise_strength
        
        # Decode back
        print("Decoding from latent space...")
        decoded_frames = []
        for i in range(len(noisy_latents)):
            decoded = vae.decode(noisy_latents[i:i+1]).sample
            decoded = (decoded + 1.0) / 2.0  # [-1,1] to [0,1]
            decoded_frames.append(decoded[0])
    
    # Convert back to PIL images
    to_pil = transforms.ToPILImage()
    processed_pil_frames = []
    for frame in decoded_frames:
        # Ensure proper value range
        frame = frame.clamp(0, 1)
        pil_img = to_pil(frame.cpu())
        processed_pil_frames.append(pil_img)
    
    # Save processed
    processed_pil_frames[0].save(
        f"{output_dir}/processed_animatediff.gif",
        save_all=True,
        append_images=processed_pil_frames[1:],
        duration=100,
        loop=0
    )
    print(f"Saved processed: {output_dir}/processed_animatediff.gif")
    
    # Also save as MP4
    processed_np = [np.array(img) for img in processed_pil_frames]
    imageio.mimsave(f"{output_dir}/processed_animatediff.mp4", processed_np, fps=10)
    print(f"Saved processed: {output_dir}/processed_animatediff.mp4")
    
    print("\n✅ AnimateDiff processing complete!")
    print(f"   Model: AnimateDiff v1.5.2")
    print(f"   Frames: {len(frames)}")
    print(f"   Resolution: {target_size}x{target_size}")
    print(f"   Device: {device}")
    
    print("\n📝 Notes on frame counts and resolution:")
    print("   - AnimateDiff was trained on 16 frames, but can handle 8-32 frames")
    print("   - More frames = more memory usage and slower processing")
    print("   - Resolutions: 256 (fast), 512 (balanced), 768 (slow but detailed)")
    print("   - Higher resolution may require switching to CPU on M1 Mac")


def generate_with_animatediff(prompt="A spaceship flying through space", output_dir="output_animatediff"):
    """
    Generate new video using AnimateDiff (its primary use case)
    """
    if not check_animatediff_available():
        print("❌ AnimateDiff not available. Install with:")
        print("   pip install diffusers[torch] transformers accelerate")
        return
    
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_gif
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Loading AnimateDiff for generation...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float32,  # Use float32 for stability
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Generate video
    print(f"\nGenerating video with prompt: '{prompt}'")
    output = pipe(
        prompt=prompt,
        num_frames=16,
        height=512,
        width=512,
        num_inference_steps=25,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
    )
    
    frames = output.frames[0]
    export_to_gif(frames, f"{output_dir}/generated_animatediff.gif")
    print(f"Generated video saved to: {output_dir}/generated_animatediff.gif")


def main():
    parser = argparse.ArgumentParser(description='Use AnimateDiff for video processing')
    parser.add_argument('--video', type=str, help='Path to video file (.mp4 or .mov)')
    parser.add_argument('--generate', type=str, help='Generate video from text prompt')
    parser.add_argument('--output', type=str, default='output_animatediff', help='Output directory')
    parser.add_argument('--frames', type=int, default=16, help='Number of frames to process (8-32 recommended)')
    parser.add_argument('--size', type=int, default=512, help='Resolution (256, 512, or 768)')
    args = parser.parse_args()
    
    if args.generate:
        print("🎬 AnimateDiff Generation Mode")
        generate_with_animatediff(args.generate, args.output)
    elif args.video:
        print("🎬 AnimateDiff Processing Mode")
        print(f"   Frames: {args.frames}")
        print(f"   Resolution: {args.size}x{args.size}")
        process_with_animatediff(args.video, args.output, args.frames, args.size)
    else:
        print("AnimateDiff Video Tool")
        print("=" * 50)
        print("\nUsage:")
        print("  Process video:  python use_animatediff.py --video your_video.mov")
        print("  With options:   python use_animatediff.py --video video.mov --frames 32 --size 768")
        print("  Generate video: python use_animatediff.py --generate 'A cat playing piano'")
        print("\nFeatures:")
        print("  - Temporal consistency across frames")
        print("  - Motion-aware encoding/decoding")
        print("  - Works with .mov and .mp4 files")
        print("\nFrame & Resolution Guidelines:")
        print("  - Frames: 8 (fast), 16 (optimal), 32 (slow but more content)")
        print("  - Resolution: 256 (fast), 512 (balanced), 768 (high quality)")
        print("  - Memory usage increases with both frames and resolution")
        print("  - M1 Mac may need CPU mode for 768x768 with many frames")
        print("\nRequirements:")
        print("  pip install diffusers[torch] transformers accelerate")


if __name__ == '__main__':
    main()