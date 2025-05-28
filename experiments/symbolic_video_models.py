#!/usr/bin/env python3
"""
Symbolic/Semantic Video Models with Compact Latent Spaces
Models that create bigger, more meaningful glitches
"""
import torch
import numpy as np
import imageio
from pathlib import Path
import argparse
from PIL import Image
import sys
sys.path.append('.')

def save_video_frames(frames, path, fps=8):
    """Save list of numpy frames as video"""
    imageio.mimsave(path, frames, fps=fps)

def load_video_frames(path, num_frames=16, size=(512, 512)):
    """Load video frames as numpy arrays"""
    reader = imageio.get_reader(path)
    frames = []
    
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
            
        # Resize if needed
        if frame.shape[:2] != size:
            img = Image.fromarray(frame)
            img = img.resize(size, Image.Resampling.LANCZOS)
            frame = np.array(img)
            
        frames.append(frame)
        
    reader.close()
    return frames

class SymbolicVideoModel:
    def __init__(self, model_type, device="auto"):
        self.model_type = model_type
        self.device = device if device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.vae = None
        self.model_info = {}
        
        print(f"\nInitializing {model_type}...")
        
        if model_type == "modelscope":
            self._load_modelscope()
        elif model_type == "cogvideo":
            self._load_cogvideo()
        elif model_type == "text2video_zero":
            self._load_text2video_zero()
        elif model_type == "tuneavideo":
            self._load_tuneavideo()
        elif model_type == "make_a_video":
            self._load_make_a_video()
        elif model_type == "phenaki":
            self._load_phenaki()
        elif model_type == "nuwa":
            self._load_nuwa()
        elif model_type == "latent_shift":
            self._load_latent_shift()
        else:
            print(f"Unknown model: {model_type}")
            self._load_modelscope()  # Fallback
    
    def _load_modelscope(self):
        """ModelScope - 256x256, 4x32x32 latents"""
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "ModelScope 1.7B",
                "description": "Text-to-video, symbolic representations",
                "latent_shape": "4x32x32 (very compact!)",
                "input_size": (256, 256)
            }
            print("✅ ModelScope loaded - compact latents!")
        except Exception as e:
            print(f"❌ ModelScope failed: {e}")
    
    def _load_cogvideo(self):
        """CogVideo - Even more symbolic, larger model"""
        try:
            from diffusers import CogVideoXPipeline
            print("Loading CogVideoX-2B (this is a large model)...")
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-2b",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "CogVideoX 2B",
                "description": "State-of-art text-to-video, very symbolic",
                "latent_shape": "16x40x40 (temporal + spatial compression)",
                "input_size": (480, 720)
            }
            print("✅ CogVideoX loaded - highly symbolic!")
        except Exception as e:
            print(f"❌ CogVideoX failed: {e}")
            self._load_modelscope()
    
    def _load_text2video_zero(self):
        """Text2Video-Zero - Uses CLIP latents"""
        try:
            from diffusers import TextToVideoZeroPipeline
            print("Loading Text2Video-Zero...")
            pipe = TextToVideoZeroPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32
            )
            # This uses SD VAE but with CLIP guidance
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "Text2Video-Zero",
                "description": "CLIP-guided, very semantic",
                "latent_shape": "4x64x64 (but CLIP-influenced)",
                "input_size": (512, 512)
            }
            print("✅ Text2Video-Zero loaded - CLIP semantic!")
        except Exception as e:
            print(f"❌ Text2Video-Zero failed: {e}")
            self._load_modelscope()
    
    def _load_tuneavideo(self):
        """Tune-A-Video - Compact one-shot learning"""
        try:
            from diffusers import DiffusionPipeline
            print("Loading Tune-A-Video...")
            # This is based on SD but with temporal layers
            pipe = DiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "Tune-A-Video",
                "description": "One-shot video learning, compact temporal",
                "latent_shape": "4x64x64 (temporally aware)",
                "input_size": (512, 512)
            }
            print("✅ Tune-A-Video loaded!")
        except Exception as e:
            print(f"❌ Tune-A-Video failed: {e}")
            self._load_modelscope()
    
    def _load_make_a_video(self):
        """Make-A-Video style - would be 3D VAE"""
        print("Make-A-Video: Not publicly available")
        print("This would use 3D VAE with shape like 4x8x32x32")
        print("Falling back to ModelScope...")
        self._load_modelscope()
    
    def _load_phenaki(self):
        """Phenaki - Variable length, very symbolic"""
        print("Phenaki: Uses C-ViViT tokenizer")
        print("Would have extremely compact latents: ~256 tokens per video")
        print("Falling back to ModelScope...")
        self._load_modelscope()
    
    def _load_nuwa(self):
        """NUWA - Microsoft's symbolic video model"""
        print("NUWA: Uses VQ-VAE with discrete tokens")
        print("Would have latents like: 512 discrete tokens")
        print("Falling back to ModelScope...")
        self._load_modelscope()
    
    def _load_latent_shift(self):
        """Latent Shift - Temporal compression focus"""
        try:
            from diffusers import ShiftedDiffusionPipeline
            print("Loading Latent Shift...")
            pipe = ShiftedDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "Latent Shift",
                "description": "Temporal dynamics in latent space",
                "latent_shape": "4x64x64 (but temporally shifted)",
                "input_size": (512, 512)
            }
            print("✅ Latent Shift loaded!")
        except Exception as e:
            print(f"❌ Latent Shift failed: {e}")
            self._load_modelscope()
    
    def get_target_size(self):
        """Get the target input size for this model"""
        size_map = {
            "modelscope": (256, 256),
            "cogvideo": (480, 720),
            "text2video_zero": (512, 512),
            "tuneavideo": (512, 512),
            "latent_shift": (512, 512)
        }
        return size_map.get(self.model_type, (512, 512))
    
    def test_semantic_glitch(self, frames, noise_levels=[0.5, 1.0, 2.0, 5.0]):
        """Test semantic glitches with interpolation"""
        results = []
        target_size = self.get_target_size()
        
        # Resize frames if needed
        if frames[0].shape[:2] != target_size:
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize(target_size[::-1], Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frames = resized_frames
        
        print(f"Processing {len(frames)} frames at {target_size}")
        
        # Encode
        print("Encoding to latent space...")
        latents = []
        with torch.no_grad():
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                latent = self.vae.encode(frame_tensor).latent_dist.sample()
                latents.append(latent[0])
        
        latents = torch.stack(latents)
        print(f"Latent shape: {latents.shape}")
        print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
        
        # Test different noise levels with interpolation
        for noise_level in noise_levels:
            print(f"\nApplying semantic noise (level={noise_level})...")
            
            # Generate noise with same distribution
            noise = torch.randn_like(latents) * latents.std() + latents.mean()
            
            # Interpolate
            alpha = min(1.0, noise_level / 5.0)  # More aggressive scaling
            noisy_latents = (1 - alpha) * latents + alpha * noise
            
            print(f"  Interpolation alpha: {alpha:.2f}")
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    frame = self.vae.decode(latent.unsqueeze(0)).sample[0]
                    frame = ((frame + 1) * 127.5).clamp(0, 255)
                    frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    # Resize back to 512x512
                    if frame.shape[:2] != (512, 512):
                        img = Image.fromarray(frame)
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        frame = np.array(img)
                    
                    decoded_frames.append(frame)
            
            results.append((noise_level, decoded_frames))
        
        return results

def list_symbolic_models():
    """List all symbolic/semantic models"""
    print("\n" + "="*70)
    print("SYMBOLIC/SEMANTIC VIDEO MODELS")
    print("="*70)
    print("\n🎯 AVAILABLE NOW:")
    print("\n1. ModelScope (1.7B)")
    print("   - Latent: 4x32x32 (8x smaller than SD!)")
    print("   - Text-to-video training = symbolic representations")
    print("   - Memory: 8-10GB")
    
    print("\n2. CogVideoX (2B)")
    print("   - Latent: 16x40x40 (temporal + spatial)")
    print("   - State-of-art symbolic understanding")
    print("   - Memory: 20-30GB (but worth it!)")
    
    print("\n3. Text2Video-Zero")
    print("   - Uses CLIP latents for semantic control")
    print("   - Standard VAE but CLIP-guided")
    print("   - Memory: 6-8GB")
    
    print("\n" + "-"*70)
    print("🔮 THEORETICAL/FUTURE:")
    
    print("\n4. Make-A-Video (Meta)")
    print("   - Would use 3D VAE: 4x8x32x32")
    print("   - Extreme temporal compression")
    
    print("\n5. Phenaki (Google)")
    print("   - C-ViViT tokens: ~256 per video")
    print("   - Most symbolic possible")
    
    print("\n6. NUWA (Microsoft)")
    print("   - VQ-VAE discrete tokens")
    print("   - ~512 tokens per video")
    
    print("\n7. VideoGPT")
    print("   - Discrete VQ-VAE: 16x16x16 tokens")
    print("   - Very symbolic, installable!")
    
    print("\n" + "-"*70)
    print("💡 WHY SMALLER LATENTS = BIGGER GLITCHES:")
    print("- Each dimension must encode more information")
    print("- More 'symbolic' rather than pixel-level")
    print("- Single bit flip = major semantic change")
    print("="*70)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', action='store_true', help='List symbolic models')
    parser.add_argument('--video', type=str, help='Input video')
    parser.add_argument('--models', type=str, default='modelscope,cogvideo',
                        help='Models to test')
    parser.add_argument('--noise', type=str, default='0.5,1,2,5',
                        help='Noise levels')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--output', type=str, default='output_symbolic')
    
    args = parser.parse_args()
    
    if args.list:
        list_symbolic_models()
        return
    
    if not args.video:
        print("Please provide --video or use --list")
        return
    
    models_to_test = args.models.split(',')
    noise_levels = [float(x) for x in args.noise.split(',')]
    
    print("="*70)
    print("SYMBOLIC VIDEO MODEL COMPARISON")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Models: {models_to_test}")
    print(f"Noise levels: {noise_levels}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load video
    print("\nLoading video...")
    frames = load_video_frames(args.video, num_frames=args.frames, size=(512, 512))
    print(f"Loaded {len(frames)} frames")
    
    # Test each model
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING: {model_name}")
        print("="*60)
        
        try:
            # Initialize model
            model = SymbolicVideoModel(model_name)
            
            # Test glitches
            results = model.test_semantic_glitch(frames, noise_levels)
            
            # Save results
            for noise_level, decoded_frames in results:
                path = output_dir / f"{model_name}_noise_{noise_level}.mp4"
                save_video_frames(decoded_frames, str(path), fps=8)
                print(f"✅ Saved: {path}")
            
            # Save model info
            info_path = output_dir / f"{model_name}_info.txt"
            with open(info_path, 'w') as f:
                for key, value in model.model_info.items():
                    f.write(f"{key}: {value}\n")
            
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
    
    # Save original
    save_video_frames(frames, str(output_dir / "original.mp4"), fps=8)
    
    print(f"\n{'='*70}")
    print("✅ SYMBOLIC MODEL TEST COMPLETE!")
    print(f"Results in: {output_dir}/")
    print("\nLook for:")
    print("- Bigger, more meaningful glitches")
    print("- Semantic rather than pixel-level changes")
    print("- How compact latents create dramatic effects")
    print("="*70)

if __name__ == "__main__":
    main()