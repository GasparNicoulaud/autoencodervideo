#!/usr/bin/env python3
"""
Advanced Semantic Glitch - Using CLIP-guided noise for truly semantic changes
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from diffusers import AutoencoderKL
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


class CLIPGuidedSemanticGlitch:
    """
    Uses CLIP to find semantic directions in latent space for meaningful glitches
    """
    
    def __init__(self, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.vae = None
        self.clip_model = None
        self.clip_processor = None
        self._load_models()
    
    def _load_models(self):
        """Load VAE and CLIP for semantic guidance"""
        print("Loading models for semantic glitch...")
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        
        # Try to load CLIP for semantic guidance
        try:
            from transformers import CLIPModel, CLIPProcessor
            print("Loading CLIP for semantic understanding...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            print("✅ CLIP loaded - semantic directions available!")
        except:
            print("⚠️  CLIP not available - using variance-based semantic detection")
            self.clip_model = None
    
    def find_semantic_directions(self, frames, latents):
        """
        Find semantic directions in latent space using CLIP or variance analysis
        """
        if self.clip_model is not None:
            # Use CLIP to find semantic features
            print("Analyzing semantic content with CLIP...")
            
            # Get CLIP embeddings for frames
            clip_features = []
            for frame in frames[:5]:  # Sample first 5 frames
                # Convert to PIL and process
                pil_frame = transforms.ToPILImage()(frame.cpu())
                inputs = self.clip_processor(images=pil_frame, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                    clip_features.append(features)
            
            clip_features = torch.cat(clip_features, dim=0)
            
            # Find directions that correlate with CLIP features
            # This is a simplified version - in practice you'd use more sophisticated methods
            semantic_dims = self._correlate_with_clip(latents, clip_features)
        else:
            # Fallback to variance-based analysis
            semantic_dims = self._variance_based_analysis(latents)
        
        return semantic_dims
    
    def _correlate_with_clip(self, latents, clip_features):
        """Find latent dimensions that correlate with CLIP features"""
        # Flatten latents
        flat_latents = latents.reshape(latents.shape[0], -1)
        
        # Compute correlation between each latent dim and CLIP features
        correlations = []
        for i in range(flat_latents.shape[1]):
            if i % 1000 == 0:
                print(f"  Analyzing dimension {i}/{flat_latents.shape[1]}...")
            
            # Correlation with CLIP
            corr = torch.corrcoef(torch.stack([
                flat_latents[:5, i],  # First 5 frames
                clip_features.mean(dim=1)  # Average CLIP feature
            ]))[0, 1].abs().item()
            correlations.append(corr)
        
        # Get top correlated dimensions
        correlations = torch.tensor(correlations)
        semantic_dims = torch.argsort(correlations, descending=True)[:100]
        
        print(f"Found {len(semantic_dims)} semantic dimensions")
        return semantic_dims
    
    def _variance_based_analysis(self, latents):
        """Fallback: find high-variance dimensions"""
        flat_latents = latents.reshape(latents.shape[0], -1)
        variance = flat_latents.var(dim=0)
        semantic_dims = torch.argsort(variance, descending=True)[:100]
        return semantic_dims
    
    def generate_semantic_noise(self, latents, semantic_dims, mode="targeted", strength=1.0):
        """
        Generate noise that affects semantic dimensions
        """
        noise = torch.zeros_like(latents)
        
        if mode == "targeted":
            # Only affect semantic dimensions
            flat_noise = noise.reshape(noise.shape[0], -1)
            flat_latents = latents.reshape(latents.shape[0], -1)
            
            # Strong noise on semantic dims
            for dim in semantic_dims[:50]:  # Top 50 semantic dims
                # Analyze the current values
                current_values = flat_latents[:, dim]
                
                # Generate structured noise based on current distribution
                if current_values.std() > 0:
                    # Flip or enhance based on distribution
                    noise_values = torch.randn(latents.shape[0], device=latents.device) * current_values.std() * strength * 5
                    
                    # Add some structure - e.g., gradual change over time
                    for t in range(len(noise_values)):
                        time_factor = t / len(noise_values)
                        noise_values[t] *= (1 + time_factor)
                    
                    flat_noise[:, dim] = noise_values
            
            noise = flat_noise.reshape(latents.shape)
            
        elif mode == "semantic_flip":
            # Flip the sign of semantic dimensions
            flat_noise = noise.reshape(noise.shape[0], -1)
            flat_latents = latents.reshape(latents.shape[0], -1)
            
            for dim in semantic_dims[:30]:
                # Invert high activations
                mask = flat_latents[:, dim].abs() > flat_latents[:, dim].abs().mean()
                flat_noise[mask, dim] = -2 * flat_latents[mask, dim]
            
            noise = flat_noise.reshape(latents.shape)
            
        elif mode == "semantic_enhance":
            # Enhance existing semantic features
            flat_noise = noise.reshape(noise.shape[0], -1)
            flat_latents = latents.reshape(latents.shape[0], -1)
            
            for dim in semantic_dims[:50]:
                # Amplify existing patterns
                pattern = flat_latents[:, dim]
                enhanced = pattern * strength * 2
                flat_noise[:, dim] = enhanced - pattern
            
            noise = flat_noise.reshape(latents.shape)
        
        else:  # uniform
            noise = torch.randn_like(latents) * strength
        
        return latents + noise
    
    def process_video(self, video_path, noise_strength=2.0, max_frames=25):
        """Process video with semantic noise"""
        # Load video
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            frames.append(frame)
        reader.close()
        
        print(f"Loaded {len(frames)} frames")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        frame_tensors = torch.stack([transform(frame) for frame in frames]).to(self.device)
        
        # Encode
        print("\nEncoding to latent space...")
        all_latents = []
        with torch.no_grad():
            for frame in frame_tensors:
                latent = self.vae.encode(frame.unsqueeze(0)).latent_dist.sample()
                all_latents.append(latent[0])
        
        latents = torch.stack(all_latents)
        print(f"Latent shape: {latents.shape}")
        
        # Find semantic directions
        semantic_dims = self.find_semantic_directions(frame_tensors, latents)
        
        # Generate results with different semantic noise types
        results = {}
        noise_configs = [
            ("original", 0, "none"),
            ("uniform", noise_strength, "uniform"),
            ("semantic_targeted", noise_strength, "targeted"),
            ("semantic_flip", 1.0, "semantic_flip"),
            ("semantic_enhance", noise_strength * 0.5, "semantic_enhance")
        ]
        
        for name, strength, mode in noise_configs:
            print(f"\nApplying {name} (strength={strength})...")
            
            if strength == 0:
                noisy_latents = latents.clone()
            else:
                noisy_latents = self.generate_semantic_noise(
                    latents, semantic_dims, mode=mode, strength=strength
                )
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    decoded = self.vae.decode(latent.unsqueeze(0)).sample
                    decoded = (decoded + 1.0) / 2.0
                    decoded_frames.append(decoded[0].clamp(0, 1))
            
            results[name] = {
                'frames': decoded_frames,
                'strength': strength,
                'mode': mode,
                'latents': noisy_latents
            }
        
        return results, latents, semantic_dims
    
    def save_results(self, results, original_latents, semantic_dims, output_dir):
        """Save results with analysis"""
        Path(output_dir).mkdir(exist_ok=True)
        
        to_pil = transforms.ToPILImage()
        
        # Save videos
        for name, data in results.items():
            frames = data['frames']
            
            # Convert to PIL
            pil_frames = [to_pil(frame.cpu()) for frame in frames]
            
            # Save GIF
            pil_frames[0].save(
                f"{output_dir}/{name}.gif",
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0
            )
            
            # Save MP4
            video_np = [np.array(img) for img in pil_frames]
            imageio.mimsave(f"{output_dir}/{name}.mp4", video_np, fps=10)
        
        # Save semantic analysis
        self._save_semantic_analysis(original_latents, semantic_dims, results, output_dir)
    
    def _save_semantic_analysis(self, latents, semantic_dims, results, output_dir):
        """Visualize semantic dimensions and their effects"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Flatten latents
        flat_latents = latents.reshape(latents.shape[0], -1).cpu().numpy()
        
        # Plot 1: Semantic dimension importance
        if len(semantic_dims) > 0:
            importance = flat_latents.var(axis=0)[semantic_dims[:50].cpu()]
            axes[0, 0].bar(range(len(importance)), importance)
            axes[0, 0].set_title('Top 50 Semantic Dimensions (by variance)')
            axes[0, 0].set_xlabel('Dimension Rank')
            axes[0, 0].set_ylabel('Variance')
        
        # Plot 2: Noise comparison
        noise_types = []
        total_changes = []
        
        for name, data in results.items():
            if name != "original":
                noise_types.append(name)
                diff = (data['latents'] - latents).abs().mean().item()
                total_changes.append(diff)
        
        axes[0, 1].bar(range(len(noise_types)), total_changes)
        axes[0, 1].set_xticks(range(len(noise_types)))
        axes[0, 1].set_xticklabels(noise_types, rotation=45)
        axes[0, 1].set_title('Total Latent Change by Noise Type')
        axes[0, 1].set_ylabel('Mean Absolute Change')
        
        # Plot 3: Temporal coherence
        axes[0, 2].set_title('Temporal Coherence')
        for name, data in results.items():
            latents_data = data['latents']
            temporal_diffs = []
            for t in range(len(latents_data) - 1):
                diff = (latents_data[t+1] - latents_data[t]).abs().mean().item()
                temporal_diffs.append(diff)
            if temporal_diffs:
                axes[0, 2].plot(temporal_diffs, label=name)
        axes[0, 2].set_xlabel('Frame Transition')
        axes[0, 2].set_ylabel('Latent Difference')
        axes[0, 2].legend()
        
        # Plot 4: Distribution changes
        axes[1, 0].set_title('Latent Distribution Changes')
        for name, data in results.items():
            flat = data['latents'].flatten().cpu().numpy()
            axes[1, 0].hist(flat, bins=50, alpha=0.5, label=name, density=True)
        axes[1, 0].set_xlabel('Latent Value')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # Plot 5: Semantic dimension changes
        if len(semantic_dims) > 0:
            axes[1, 1].set_title('Semantic Dimension Changes')
            
            orig_semantic = flat_latents[:, semantic_dims[:20].cpu()]
            
            for name, data in results.items():
                if name != "original":
                    mod_flat = data['latents'].reshape(data['latents'].shape[0], -1).cpu().numpy()
                    mod_semantic = mod_flat[:, semantic_dims[:20].cpu()]
                    
                    changes = (mod_semantic - orig_semantic).mean(axis=0)
                    axes[1, 1].plot(changes, label=name, marker='o')
            
            axes[1, 1].set_xlabel('Semantic Dimension Index')
            axes[1, 1].set_ylabel('Mean Change')
            axes[1, 1].legend()
        
        # Plot 6: Frame comparison
        axes[1, 2].set_title('Visual Comparison (Middle Frame)')
        n_results = len(results)
        for idx, (name, data) in enumerate(results.items()):
            mid_frame = len(data['frames']) // 2
            frame = data['frames'][mid_frame].cpu().permute(1, 2, 0).numpy()
            
            # Create small subplot
            ax = plt.subplot(2, 3, 6)
            ax.clear()
            
            # Create grid of frames
            grid_size = int(np.ceil(np.sqrt(n_results)))
            ax_sub = plt.subplot(2, 3, 6, projection=None)
            ax_sub.axis('off')
            
            # Just show text instead of complex grid
            ax_sub.text(0.5, 0.5, f"{n_results} variants created\nSee individual files", 
                       ha='center', va='center', fontsize=12)
        
        plt.suptitle('Semantic Glitch Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/semantic_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Saved semantic analysis to {output_dir}/semantic_analysis.png")


def main():
    parser = argparse.ArgumentParser(description='Advanced Semantic Glitch with CLIP')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='output_semantic_advanced', 
                        help='Output directory')
    parser.add_argument('--strength', type=float, default=2.0, 
                        help='Base noise strength')
    parser.add_argument('--frames', type=int, default=25,
                        help='Number of frames to process')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ADVANCED SEMANTIC GLITCH EXPERIMENT")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Frames: {args.frames}")
    print(f"Base strength: {args.strength}")
    print("="*70 + "\n")
    
    # Initialize
    glitcher = CLIPGuidedSemanticGlitch()
    
    # Process
    results, original_latents, semantic_dims = glitcher.process_video(
        args.video, 
        noise_strength=args.strength,
        max_frames=args.frames
    )
    
    # Save
    print(f"\nSaving results to {args.output}/")
    glitcher.save_results(results, original_latents, semantic_dims, args.output)
    
    print("\n✅ Advanced semantic glitch complete!")
    print(f"\nNoise types applied:")
    print("  - uniform: Standard random noise (baseline)")
    print("  - semantic_targeted: Noise only on semantic dimensions")
    print("  - semantic_flip: Invert semantic features")
    print("  - semantic_enhance: Amplify existing semantic patterns")
    
    print(f"\nResults in {args.output}/")


if __name__ == '__main__':
    main()