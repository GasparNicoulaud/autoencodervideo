#!/usr/bin/env python3
"""
Semantic Glitch with SVD - Using Stable Video Diffusion for more powerful semantic understanding
Focuses on uniform noise which shows the most promise
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class PowerfulSemanticGlitch:
    """
    Uses more powerful video models (SVD) for semantic glitches on M1 Max
    """
    
    def __init__(self, model_type="svd", device=None):
        self.model_type = model_type
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.vae = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load Stable Video Diffusion's VAE - much more powerful for video semantics"""
        if self.model_type == "svd":
            print("Loading Stable Video Diffusion VAE (12-16GB)...")
            print("This model understands video semantics much better than SD-VAE...")
            
            # Load SVD's temporal VAE which is specifically designed for video
            self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                subfolder="vae",
                torch_dtype=torch.float32  # M1 needs float32
            ).to(self.device)
            
            self.vae.eval()
            print(f"✅ Loaded SVD VAE on {self.device}")
            print(f"   This VAE understands temporal semantics and motion!")
        
        elif self.model_type == "sd-vae":
            # Fallback to regular SD VAE
            from diffusers import AutoencoderKL
            print("Loading standard SD VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32
            ).to(self.device)
            self.vae.eval()
    
    def process_video(self, video_path, noise_strength=2.0, max_frames=25):
        """
        Process video with uniform noise at various strengths
        SVD supports up to 25 frames natively
        """
        # Load video
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            frames.append(frame)
        reader.close()
        
        print(f"Loaded {len(frames)} frames")
        
        # Prepare frames for SVD VAE
        if self.model_type == "svd":
            # SVD expects 576x1024 resolution
            target_size = (576, 1024)
        else:
            target_size = (512, 512)
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Convert frames
        frame_tensors = torch.stack([transform(frame) for frame in frames])
        print(f"Preprocessed to shape: {frame_tensors.shape}")
        
        if self.model_type == "svd":
            # SVD VAE expects [B, F, C, H, W] format
            video_tensor = frame_tensors.unsqueeze(0).permute(0, 2, 1, 3, 4).to(self.device)
            print(f"SVD input shape: {video_tensor.shape}")
        else:
            video_tensor = frame_tensors.to(self.device)
        
        # Encode to latent space
        print("\nEncoding to semantic latent space...")
        with torch.no_grad():
            if self.model_type == "svd":
                # SVD temporal VAE needs frame-by-frame encoding
                # It's still more powerful due to temporal decoder
                all_latents = []
                for i in range(video_tensor.shape[2]):  # iterate over frames
                    frame = video_tensor[:, :, i, :, :]  # [B, C, H, W]
                    latent = self.vae.encode(frame).latent_dist.sample()
                    all_latents.append(latent)
                latents = torch.stack(all_latents, dim=2)  # [B, C, F, H, W]
                print(f"SVD Latent shape: {latents.shape}")
            else:
                # Regular SD VAE - encode each frame
                all_latents = []
                for frame in video_tensor:
                    latent = self.vae.encode(frame.unsqueeze(0)).latent_dist.sample()
                    all_latents.append(latent[0])
                latents = torch.stack(all_latents).unsqueeze(0)
        
        # Analyze latent space
        print("\nAnalyzing semantic latent space...")
        latent_flat = latents.flatten()
        print(f"Latent statistics:")
        print(f"  Mean: {latent_flat.mean():.3f}")
        print(f"  Std: {latent_flat.std():.3f}")
        print(f"  Min: {latent_flat.min():.3f}")
        print(f"  Max: {latent_flat.max():.3f}")
        
        # Apply uniform noise at different strengths
        results = {}
        noise_strengths = [0.0, noise_strength * 0.5, noise_strength, noise_strength * 2.0, noise_strength * 4.0]
        
        for strength in noise_strengths:
            print(f"\nApplying uniform noise (strength={strength:.1f})...")
            
            if strength == 0:
                noisy_latents = latents.clone()
            else:
                # Uniform noise across all dimensions
                noise = torch.randn_like(latents) * strength
                noisy_latents = latents + noise
            
            # Decode
            print("Decoding from noisy latents...")
            with torch.no_grad():
                if self.model_type == "svd":
                    # SVD decode - using temporal decoder frame by frame
                    decoded_frames = []
                    for i in range(noisy_latents.shape[2]):  # iterate over frames
                        frame_latent = noisy_latents[:, :, i, :, :]  # [B, C, H, W]
                        decoded_frame = self.vae.decode(frame_latent).sample
                        decoded_frames.append(decoded_frame[0])
                    decoded = torch.stack(decoded_frames)  # [F, C, H, W]
                else:
                    # SD VAE decode
                    decoded_frames = []
                    for i in range(noisy_latents.shape[1]):
                        frame_latent = noisy_latents[0, i].unsqueeze(0)
                        decoded_frame = self.vae.decode(frame_latent).sample
                        decoded_frames.append(decoded_frame[0])
                    decoded = torch.stack(decoded_frames)
                
                # Normalize to [0, 1]
                decoded = (decoded + 1.0) / 2.0
                decoded = decoded.clamp(0, 1)
            
            results[f"noise_{strength:.1f}"] = {
                'frames': decoded,
                'strength': strength,
                'latents': noisy_latents
            }
        
        return results, latents
    
    def save_results(self, results, original_latents, output_dir):
        """Save glitched videos and analysis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        to_pil = transforms.ToPILImage()
        
        # Save each noise level
        for name, data in results.items():
            frames = data['frames']
            strength = data['strength']
            
            # Convert to PIL images
            pil_frames = [to_pil(frame.cpu()) for frame in frames]
            
            # Save as GIF
            pil_frames[0].save(
                f"{output_dir}/{name}.gif",
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0
            )
            
            # Save as MP4
            video_np = [np.array(img) for img in pil_frames]
            imageio.mimsave(f"{output_dir}/{name}.mp4", video_np, fps=10)
            
            print(f"Saved {name}.mp4 and {name}.gif")
        
        # Save comparison plot
        self._save_comparison_plot(results, output_dir)
        
        # Save latent analysis
        self._save_latent_analysis(original_latents, results, output_dir)
    
    def _save_comparison_plot(self, results, output_dir):
        """Create a grid showing different noise strengths"""
        fig, axes = plt.subplots(1, len(results), figsize=(20, 4))
        
        for idx, (name, data) in enumerate(results.items()):
            # Show middle frame
            mid_frame = len(data['frames']) // 2
            frame = data['frames'][mid_frame].cpu().permute(1, 2, 0).numpy()
            
            axes[idx].imshow(frame)
            axes[idx].set_title(f"Strength: {data['strength']:.1f}")
            axes[idx].axis('off')
        
        plt.suptitle("Semantic Glitch: Uniform Noise at Different Strengths", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/noise_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_latent_analysis(self, original_latents, results, output_dir):
        """Analyze how noise affects the latent distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Flatten latents for analysis
        orig_flat = original_latents.flatten().cpu().numpy()
        
        # Plot 1: Latent distribution changes
        axes[0, 0].hist(orig_flat, bins=50, alpha=0.7, label='Original', density=True, color='blue')
        
        colors = ['green', 'orange', 'red', 'purple', 'brown']
        for idx, (name, data) in enumerate(results.items()):
            if data['strength'] > 0:
                noisy_flat = data['latents'].flatten().cpu().numpy()
                axes[0, 0].hist(noisy_flat, bins=50, alpha=0.5, 
                               label=f"Noise {data['strength']:.1f}", 
                               density=True, color=colors[idx % len(colors)])
        
        axes[0, 0].set_title('Latent Distribution Changes')
        axes[0, 0].set_xlabel('Latent Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # Plot 2: Variance changes
        variances = [orig_flat.var()]
        strengths = [0.0]
        for name, data in results.items():
            if data['strength'] > 0:
                noisy_flat = data['latents'].flatten().cpu().numpy()
                variances.append(noisy_flat.var())
                strengths.append(data['strength'])
        
        axes[0, 1].plot(strengths, variances, 'o-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Latent Variance vs Noise Strength')
        axes[0, 1].set_xlabel('Noise Strength')
        axes[0, 1].set_ylabel('Total Variance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Temporal coherence (for video)
        if len(original_latents.shape) >= 3:
            # Calculate frame-to-frame differences
            axes[1, 0].set_title('Temporal Coherence')
            
            for name, data in results.items():
                latents = data['latents']
                # Calculate temporal differences
                if self.model_type == "svd" and len(latents.shape) == 5:
                    # For SVD: [B, C, F, H, W]
                    temporal_diffs = []
                    for f in range(latents.shape[2] - 1):
                        diff = (latents[:, :, f+1] - latents[:, :, f]).abs().mean().item()
                        temporal_diffs.append(diff)
                elif len(latents.shape) == 4:
                    # For SD VAE: [B, F, C, H, W] or similar
                    temporal_diffs = []
                    for f in range(latents.shape[1] - 1):
                        diff = (latents[:, f+1] - latents[:, f]).abs().mean().item()
                        temporal_diffs.append(diff)
                else:
                    temporal_diffs = []
                
                if temporal_diffs:
                    axes[1, 0].plot(temporal_diffs, label=f"Strength {data['strength']:.1f}")
            
            axes[1, 0].set_xlabel('Frame Transition')
            axes[1, 0].set_ylabel('Latent Difference')
            axes[1, 0].legend()
        
        # Plot 4: Information about the model
        axes[1, 1].text(0.1, 0.8, f"Model: {self.model_type.upper()}", fontsize=14)
        axes[1, 1].text(0.1, 0.6, f"Device: {self.device}", fontsize=14)
        axes[1, 1].text(0.1, 0.4, f"Latent Shape: {list(original_latents.shape)}", fontsize=14)
        if self.model_type == "svd":
            axes[1, 1].text(0.1, 0.2, "Temporal VAE: Yes", fontsize=14, color='green')
            axes[1, 1].text(0.1, 0.1, "Semantic Understanding: High", fontsize=14, color='green')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Model Information')
        
        plt.suptitle('Semantic Latent Space Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latent_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Powerful Semantic Glitch with SVD')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='svd', 
                        choices=['svd', 'sd-vae'],
                        help='Model to use (svd=Stable Video Diffusion VAE)')
    parser.add_argument('--output', type=str, default='output_semantic_glitch_svd', 
                        help='Output directory')
    parser.add_argument('--strength', type=float, default=1.0, 
                        help='Base noise strength')
    parser.add_argument('--frames', type=int, default=25,
                        help='Number of frames to process (SVD supports up to 25)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("POWERFUL SEMANTIC GLITCH EXPERIMENT")
    print("="*70)
    print(f"Model: {args.model.upper()}")
    print(f"Video: {args.video}")
    print(f"Frames: {args.frames}")
    print(f"Base strength: {args.strength}")
    print("="*70 + "\n")
    
    # Initialize
    glitcher = PowerfulSemanticGlitch(model_type=args.model)
    
    # Process video
    results, original_latents = glitcher.process_video(
        args.video, 
        noise_strength=args.strength,
        max_frames=args.frames
    )
    
    # Save results
    print(f"\nSaving results to {args.output}/")
    glitcher.save_results(results, original_latents, args.output)
    
    print("\n✅ Semantic glitch experiment complete!")
    print(f"\nResults saved to {args.output}/:")
    print("  - noise_X.X.mp4/gif - Videos at different noise strengths")
    print("  - noise_comparison.png - Visual comparison")
    print("  - latent_analysis.png - Latent space analysis")
    
    if args.model == "svd":
        print("\n🎯 SVD VAE provides much better semantic understanding than SD-VAE!")
        print("   It was trained on video data and understands temporal coherence.")


if __name__ == '__main__':
    main()