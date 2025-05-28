#!/usr/bin/env python3
"""
Semantic Glitch Experiment - Inducing semantic changes through latent space noise
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from diffusers import AutoencoderKL
from transformers import VideoMAEImageProcessor, VideoMAEForPreTraining
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt


class SemanticGlitchInducer:
    """
    Induces semantic glitches in video by strategically adding noise to latent representations
    """
    
    def __init__(self, model_type="sd-vae", device=None):
        self.model_type = model_type
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the specified autoencoder model"""
        if self.model_type == "sd-vae":
            print("Loading Stable Diffusion VAE...")
            self.model = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32
            ).to(self.device)
        elif self.model_type == "videomae":
            print("Loading VideoMAE...")
            self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
            self.model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
            self.model = self.model.to(self.device)
        self.model.eval()
    
    def analyze_latent_semantics(self, latents):
        """
        Analyze latent space to identify semantic regions
        Returns principal components that might encode semantic features
        """
        # Flatten latents for analysis
        if len(latents.shape) > 2:
            flat_latents = latents.reshape(latents.shape[0], -1)
        else:
            flat_latents = latents
        
        # Compute statistics
        mean = flat_latents.mean(dim=0)
        std = flat_latents.std(dim=0)
        
        # Find high-variance dimensions (likely semantic)
        variance = std ** 2
        semantic_dims = torch.argsort(variance, descending=True)[:100]  # Top 100 dims
        
        return {
            'mean': mean,
            'std': std,
            'variance': variance,
            'semantic_dims': semantic_dims,
            'activation_pattern': flat_latents
        }
    
    def generate_semantic_noise(self, latents, noise_type="targeted", strength=0.5):
        """
        Generate different types of noise for semantic manipulation
        """
        analysis = self.analyze_latent_semantics(latents)
        
        if noise_type == "targeted":
            # Target high-variance dimensions (likely semantic)
            noise = torch.zeros_like(latents)
            flat_noise = noise.reshape(noise.shape[0], -1)
            
            # Add strong noise only to semantic dimensions
            for dim in analysis['semantic_dims'][:20]:  # Top 20 semantic dims
                flat_noise[:, dim] = torch.randn(latents.shape[0]) * strength * 10
            
            noise = flat_noise.reshape(latents.shape)
            
        elif noise_type == "gradient":
            # Create gradient noise that changes over time
            noise = torch.zeros_like(latents)
            for t in range(latents.shape[0]):
                time_factor = t / latents.shape[0]
                noise[t] = torch.randn_like(latents[0]) * strength * time_factor
        
        elif noise_type == "flip":
            # Flip sign of certain activations (semantic inversion)
            noise = torch.zeros_like(latents)
            mask = (latents.abs() > latents.abs().mean())
            noise[mask] = -2 * latents[mask]  # Flip high activations
        
        elif noise_type == "structured":
            # Add structured patterns that might encode motion
            noise = torch.zeros_like(latents)
            # Create wave pattern
            if len(latents.shape) == 4:  # [T, C, H, W]
                for t in range(latents.shape[0]):
                    phase = 2 * np.pi * t / latents.shape[0]
                    for h in range(latents.shape[2]):
                        for w in range(latents.shape[3]):
                            spatial_phase = 2 * np.pi * (h + w) / (latents.shape[2] + latents.shape[3])
                            noise[t, :, h, w] = strength * np.sin(phase + spatial_phase)
        
        elif noise_type == "channel_swap":
            # Swap channels to induce color/feature changes
            noise = torch.zeros_like(latents)
            if len(latents.shape) >= 2:
                # Randomly permute channels
                perm = torch.randperm(latents.shape[1])
                return latents[:, perm]
        
        else:  # uniform
            noise = torch.randn_like(latents) * strength
        
        return latents + noise
    
    def process_video_with_glitches(self, video_path, noise_configs):
        """
        Process video with multiple types of semantic glitches
        """
        # Load video
        frames = self._load_video(video_path, max_frames=16)
        
        if self.model_type == "sd-vae":
            return self._process_with_sdvae(frames, noise_configs)
        elif self.model_type == "videomae":
            return self._process_with_videomae(frames, noise_configs)
    
    def _load_video(self, video_path, max_frames=16):
        """Load and preprocess video frames"""
        reader = imageio.get_reader(video_path)
        frames = []
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            frames.append(frame)
        reader.close()
        return frames
    
    def _process_with_sdvae(self, frames, noise_configs):
        """Process using Stable Diffusion VAE"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Convert frames
        frame_tensors = torch.stack([transform(frame) for frame in frames]).to(self.device)
        
        # Encode all frames
        all_latents = []
        with torch.no_grad():
            for frame in frame_tensors:
                latent = self.model.encode(frame.unsqueeze(0)).latent_dist.sample()
                all_latents.append(latent[0])
        
        latents = torch.stack(all_latents)
        print(f"Latent shape: {latents.shape}")
        
        # Apply different noise types
        results = {}
        for config in noise_configs:
            noise_type = config['type']
            strength = config.get('strength', 0.5)
            
            print(f"\nApplying {noise_type} noise (strength={strength})...")
            noisy_latents = self.generate_semantic_noise(latents, noise_type, strength)
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    decoded = self.model.decode(latent.unsqueeze(0)).sample
                    decoded = (decoded + 1.0) / 2.0
                    decoded_frames.append(decoded[0])
            
            results[noise_type] = {
                'frames': decoded_frames,
                'latents': noisy_latents,
                'original_latents': latents
            }
        
        return results
    
    def _process_with_videomae(self, frames, noise_configs):
        """Process using VideoMAE"""
        # Resize frames for VideoMAE
        resized_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            img = img.resize((224, 224), Image.Resampling.LANCZOS)
            resized_frames.append(np.array(img))
        
        # Process with VideoMAE
        inputs = self.processor(list(resized_frames), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # No masking for full encoding
            batch_size = inputs['pixel_values'].shape[0]
            seq_length = 1568
            bool_masked_pos = torch.zeros((batch_size, seq_length), dtype=torch.bool).to(self.device)
            
            outputs = self.model(**inputs, bool_masked_pos=bool_masked_pos)
            latents = outputs.last_hidden_state
        
        print(f"VideoMAE latent shape: {latents.shape}")
        
        # Apply noise (VideoMAE doesn't have direct decoder, so we return latents)
        results = {}
        for config in noise_configs:
            noise_type = config['type']
            strength = config.get('strength', 0.5)
            
            noisy_latents = self.generate_semantic_noise(latents, noise_type, strength)
            results[noise_type] = {
                'latents': noisy_latents,
                'original_latents': latents
            }
        
        return results
    
    def visualize_glitch_effects(self, results, output_dir):
        """Visualize the effects of different semantic glitches"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save videos/gifs for each glitch type
        from torchvision import transforms
        to_pil = transforms.ToPILImage()
        
        for glitch_type, data in results.items():
            if 'frames' in data:
                # Save as GIF
                pil_frames = [to_pil(frame.clamp(0, 1).cpu()) for frame in data['frames']]
                pil_frames[0].save(
                    f"{output_dir}/{glitch_type}_glitch.gif",
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=100,
                    loop=0
                )
                
                # Save as video
                video_np = [np.array(img) for img in pil_frames]
                imageio.mimsave(f"{output_dir}/{glitch_type}_glitch.mp4", video_np, fps=10)
            
            # Save latent analysis
            if 'latents' in data:
                self._save_latent_analysis(
                    data['original_latents'], 
                    data['latents'], 
                    f"{output_dir}/{glitch_type}_analysis.png",
                    glitch_type
                )
    
    def _save_latent_analysis(self, original, modified, save_path, title):
        """Save visualization of latent space changes"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Flatten for visualization
        orig_flat = original.reshape(original.shape[0], -1).cpu().numpy()
        mod_flat = modified.reshape(modified.shape[0], -1).cpu().numpy()
        
        # Plot 1: Mean activation over time
        axes[0, 0].plot(orig_flat.mean(axis=1), label='Original')
        axes[0, 0].plot(mod_flat.mean(axis=1), label='Modified')
        axes[0, 0].set_title('Mean Activation Over Time')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].legend()
        
        # Plot 2: Variance changes
        axes[0, 1].bar(range(2), [orig_flat.var(), mod_flat.var()])
        axes[0, 1].set_xticks(range(2))
        axes[0, 1].set_xticklabels(['Original', 'Modified'])
        axes[0, 1].set_title('Total Variance')
        
        # Plot 3: Difference heatmap
        diff = mod_flat - orig_flat
        im = axes[1, 0].imshow(diff[:, :100], aspect='auto', cmap='RdBu')
        axes[1, 0].set_title('Latent Differences (first 100 dims)')
        axes[1, 0].set_xlabel('Latent Dimension')
        axes[1, 0].set_ylabel('Frame')
        plt.colorbar(im, ax=axes[1, 0])
        
        # Plot 4: Distribution change
        axes[1, 1].hist(orig_flat.flatten(), bins=50, alpha=0.5, label='Original', density=True)
        axes[1, 1].hist(mod_flat.flatten(), bins=50, alpha=0.5, label='Modified', density=True)
        axes[1, 1].set_title('Latent Distribution')
        axes[1, 1].legend()
        
        plt.suptitle(f'Semantic Glitch Analysis: {title}')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Semantic Glitch Experiment')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='sd-vae', choices=['sd-vae', 'videomae'],
                        help='Model to use')
    parser.add_argument('--output', type=str, default='output_semantic_glitch', 
                        help='Output directory')
    parser.add_argument('--strength', type=float, default=0.5, 
                        help='Base noise strength')
    args = parser.parse_args()
    
    # Define different semantic glitch configurations
    noise_configs = [
        {'type': 'targeted', 'strength': args.strength * 5.0},
        {'type': 'gradient', 'strength': args.strength * 4.0},
        {'type': 'flip', 'strength': 1.0},  # Flip doesn't use strength
        {'type': 'structured', 'strength': args.strength * 7.5},
        {'type': 'channel_swap', 'strength': 1.0},
        {'type': 'uniform', 'strength': args.strength * 1.5}
    ]
    
    # Initialize glitch inducer
    print(f"Initializing Semantic Glitch Inducer with {args.model}...")
    inducer = SemanticGlitchInducer(model_type=args.model)
    
    # Process video with glitches
    print(f"\nProcessing video: {args.video}")
    results = inducer.process_video_with_glitches(args.video, noise_configs)
    
    # Visualize results
    print(f"\nSaving results to {args.output}/")
    inducer.visualize_glitch_effects(results, args.output)
    
    print("\n✅ Semantic glitch experiment complete!")
    print("\nGlitch types applied:")
    print("  - targeted: Noise on high-variance (semantic) dimensions")
    print("  - gradient: Time-varying noise (motion change)")
    print("  - flip: Sign inversion of strong activations")
    print("  - structured: Wave patterns for motion effects")
    print("  - channel_swap: Feature channel permutation")
    print("  - uniform: Baseline random noise")
    
    print(f"\nCheck {args.output}/ for results!")


if __name__ == '__main__':
    main()