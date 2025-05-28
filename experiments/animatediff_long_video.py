#!/usr/bin/env python3
"""
Create longer videos with AnimateDiff by processing in batches
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image

def process_long_video_animatediff(video_path, total_frames=48, batch_size=16, output_dir="output_long_animatediff"):
    """
    Process long videos by breaking them into AnimateDiff-sized chunks
    """
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    except ImportError:
        print("❌ AnimateDiff not available")
        return
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Processing {total_frames} frames in batches of {batch_size}")
    
    # Load video
    reader = imageio.get_reader(video_path)
    frames = []
    for i, frame in enumerate(reader):
        if i >= total_frames:
            break
        frames.append(frame)
    reader.close()
    
    print(f"Loaded {len(frames)} frames")
    
    # Load AnimateDiff
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
    pipe = AnimateDiffPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        motion_adapter=adapter,
        torch_dtype=torch.float32,
    ).to(device)
    
    vae = pipe.vae
    vae.eval()
    
    # Process in batches
    all_processed_frames = []
    
    for batch_start in range(0, len(frames), batch_size):
        batch_end = min(batch_start + batch_size, len(frames))
        batch_frames = frames[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}: frames {batch_start}-{batch_end}")
        
        # Prepare batch
        pil_frames = []
        for frame in batch_frames:
            img = Image.fromarray(frame)
            # Center crop and resize
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
            img = img.resize((512, 512), Image.Resampling.LANCZOS)
            pil_frames.append(img)
        
        # Convert to tensors
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        frame_tensors = torch.stack([transform(img) for img in pil_frames]).to(device)
        
        # Process through VAE
        with torch.no_grad():
            latents = []
            for i in range(len(frame_tensors)):
                frame = frame_tensors[i:i+1]
                latent = vae.encode(frame).latent_dist.sample()
                latents.append(latent)
            
            latents = torch.stack(latents, dim=0)
            
            # Add noise
            noise_strength = 2.0
            noise = torch.randn_like(latents) * noise_strength
            noisy_latents = latents + noise
            
            # Decode
            decoded_frames = []
            for i in range(len(noisy_latents)):
                decoded = vae.decode(noisy_latents[i:i+1]).sample
                decoded = (decoded + 1.0) / 2.0
                decoded_frames.append(decoded[0])
        
        # Convert back to images
        to_pil = transforms.ToPILImage()
        batch_processed = [to_pil(frame.clamp(0, 1).cpu()) for frame in decoded_frames]
        all_processed_frames.extend(batch_processed)
        
        print(f"Batch {batch_start//batch_size + 1} complete")
    
    # Save long video
    processed_np = [np.array(img) for img in all_processed_frames]
    imageio.mimsave(f"{output_dir}/long_video_{total_frames}frames.mp4", processed_np, fps=12)
    
    # Save GIF
    all_processed_frames[0].save(
        f"{output_dir}/long_video_{total_frames}frames.gif",
        save_all=True,
        append_images=all_processed_frames[1:],
        duration=83,  # 12 fps
        loop=0
    )
    
    print(f"\n✅ Long video complete!")
    print(f"   Total frames: {len(all_processed_frames)}")
    print(f"   Saved: {output_dir}/long_video_{total_frames}frames.mp4")

def main():
    parser = argparse.ArgumentParser(description='Create long videos with AnimateDiff')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--frames', type=int, default=48, help='Total frames to process')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--output', type=str, default='output_long_animatediff', help='Output directory')
    args = parser.parse_args()
    
    process_long_video_animatediff(args.video, args.frames, args.batch, args.output)

if __name__ == '__main__':
    main()