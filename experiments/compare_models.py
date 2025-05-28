#!/usr/bin/env python3
"""
Compare Multiple Video Models with Uniform Noise and Pure Noise Decoding
Tests how different autoencoders handle noise in latent space
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

def get_optimal_size_for_model(model_type, input_size):
    """Get the optimal processing size for each model"""
    input_min_dim = min(input_size)
    
    # Model-specific optimal sizes (preferred resolutions)
    model_preferred_sizes = {
        "sd_vae": [512, 768, 1024],
        "sd_vae_mse": [512, 768, 1024], 
        "sd_vae_ema": [512, 768, 1024],
        "animatediff": [512, 768],  # AnimateDiff works best at these
        "zeroscope": [320, 576],  # ZeroScope native sizes
        "modelscope": [256, 512],  # ModelScope optimal
        "svd": [576, 1024],  # SVD preferred
        "cogvideo": [480, 720],  # CogVideo sizes
        "show1": [256, 512],
        "hotshot": [512, 768],
        "videogpt": [64, 128],  # VideoGPT requires exactly 64 or 128
        "magvit": [256, 512],
        "temporal_vae": [512, 768, 1024]
    }
    
    preferred = model_preferred_sizes.get(model_type, [512, 768, 1024])
    
    # Find the largest size that doesn't exceed input
    optimal_size = 512  # Default fallback
    for size in preferred:
        if size <= input_min_dim:
            optimal_size = size
        else:
            break
    
    return optimal_size

def load_video_frames(path, num_frames=16, target_size=None):
    """Load video frames with optimal sizing"""
    reader = imageio.get_reader(path)
    frames = []
    
    # Get first frame to determine input size
    first_frame = reader.get_data(0)
    input_height, input_width = first_frame.shape[:2]
    input_size = (input_width, input_height)
    
    # Use provided target_size or calculate optimal
    if target_size is None:
        # Use the minimum dimension as square size
        target_size = min(input_width, input_height)
        target_size = min(target_size, 1024)  # Cap at 1024 for memory
    
    print(f"Input video size: {input_width}x{input_height}")
    print(f"Processing at: {target_size}x{target_size}")
    
    reader.close()
    reader = imageio.get_reader(path)  # Restart reader
    
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
            
        # Process frame with top cropping for portrait videos
        if frame.shape[:2] != (target_size, target_size):
            img = Image.fromarray(frame)
            
            # Get current dimensions
            width, height = img.size
            
            # For portrait videos, crop from top to get square
            if height > width:
                # Portrait: crop from top, take top square portion
                crop_size = min(width, height)
                left = (width - crop_size) // 2  # Center horizontally
                top = 0  # Start from top
                right = left + crop_size
                bottom = top + crop_size
                img = img.crop((left, top, right, bottom))
            else:
                # Landscape: use center crop
                crop_size = min(width, height)
                left = (width - crop_size) // 2
                top = (height - crop_size) // 2
                right = left + crop_size
                bottom = top + crop_size
                img = img.crop((left, top, right, bottom))
            
            # Now resize to target size
            img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            frame = np.array(img)
            
        frames.append(frame)
        
    reader.close()
    return frames, input_size

class VideoModelTester:
    def __init__(self, model_type, device="auto"):
        self.model_type = model_type
        self.device = device if device != "auto" else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.vae = None
        self.model_info = {}
        self.is_video_level = False  # Default to frame-by-frame
        
        print(f"\nInitializing {model_type}...")
        
        if model_type == "sd_vae":
            self._load_sd_vae()
        elif model_type == "animatediff":
            self._load_animatediff()
        elif model_type == "zeroscope":
            self._load_zeroscope()
        elif model_type == "modelscope":
            self._load_modelscope()
        elif model_type == "svd":
            self._load_svd()
        elif model_type == "sd_vae_mse":
            self._load_sd_vae_mse()
        elif model_type == "sd_vae_ema":
            self._load_sd_vae_ema()
        elif model_type == "cogvideo":
            self._load_cogvideo()
        elif model_type == "show1":
            self._load_show1()
        elif model_type == "hotshot":
            self._load_hotshot()
        elif model_type == "videogpt":
            self._load_videogpt()
        elif model_type == "magvit":
            self._load_magvit()
        elif model_type == "temporal_vae":
            self._load_temporal_vae()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_sd_vae(self):
        """Standard SD VAE"""
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        self.model_info = {
            "name": "SD VAE (MSE)",
            "description": "Stable Diffusion VAE - baseline",
            "latent_shape": "4x64x64 per frame"
        }
        print("✅ SD VAE loaded")
    
    def _load_sd_vae_mse(self):
        """SD VAE MSE variant"""
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        self.model_info = {
            "name": "SD VAE FT-MSE",
            "description": "Fine-tuned for better reconstruction",
            "latent_shape": "4x64x64 per frame"
        }
        print("✅ SD VAE FT-MSE loaded")
    
    def _load_sd_vae_ema(self):
        """SD VAE EMA variant"""
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema",
            torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval()
        self.model_info = {
            "name": "SD VAE FT-EMA",
            "description": "EMA weights for stability",
            "latent_shape": "4x64x64 per frame"
        }
        print("✅ SD VAE FT-EMA loaded")
    
    def _load_animatediff(self):
        """AnimateDiff VAE"""
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
            pipe = AnimateDiffPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                motion_adapter=adapter,
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "AnimateDiff VAE",
                "description": "Motion-aware encoding",
                "latent_shape": "4x64x64 per frame (motion-aware)"
            }
            print("✅ AnimateDiff VAE loaded")
        except Exception as e:
            print(f"❌ AnimateDiff failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_zeroscope(self):
        """ZeroScope V2 VAE"""
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                "cerspense/zeroscope_v2_576w",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "ZeroScope V2 VAE",
                "description": "Video-optimized VAE",
                "latent_shape": "4x40x72 per frame (576x320)"
            }
            print("✅ ZeroScope VAE loaded")
        except Exception as e:
            print(f"❌ ZeroScope failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_modelscope(self):
        """ModelScope VAE"""
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                "damo-vilab/text-to-video-ms-1.7b",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "ModelScope VAE",
                "description": "Text-to-video VAE",
                "latent_shape": "4x32x32 per frame (256x256)"
            }
            print("✅ ModelScope VAE loaded")
        except Exception as e:
            print(f"❌ ModelScope failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_svd(self):
        """Stable Video Diffusion VAE"""
        try:
            from diffusers import StableVideoDiffusionPipeline
            pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "SVD VAE",
                "description": "State-of-art video VAE",
                "latent_shape": "4x72x128 per frame (576x1024)"
            }
            print("✅ SVD VAE loaded")
        except Exception as e:
            print(f"❌ SVD failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_cogvideo(self):
        """CogVideoX - Highly symbolic video model"""
        try:
            from diffusers import CogVideoXPipeline
            print("Loading CogVideoX-2B (large model, ~20GB)...")
            pipe = CogVideoXPipeline.from_pretrained(
                "THUDM/CogVideoX-2b",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "CogVideoX 2B",
                "description": "Symbolic video model",
                "latent_shape": "16x40x40 (very compact!)"
            }
            print("✅ CogVideoX loaded - expect big glitches!")
        except Exception as e:
            print(f"❌ CogVideoX failed to load: {e}")
            self._load_modelscope()  # Fallback to similar model
    
    def _load_show1(self):
        """Show-1 - Compact video model"""
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                "showlab/show-1-base",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "Show-1",
                "description": "Efficient video model",
                "latent_shape": "4x32x32 per frame"
            }
            print("✅ Show-1 loaded")
        except Exception as e:
            print(f"❌ Show-1 failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_hotshot(self):
        """Hotshot-XL - GIF generation model"""
        try:
            from diffusers import DiffusionPipeline
            pipe = DiffusionPipeline.from_pretrained(
                "hotshotco/Hotshot-XL",
                torch_dtype=torch.float32
            )
            self.vae = pipe.vae.to(self.device)
            self.vae.eval()
            self.model_info = {
                "name": "Hotshot-XL",
                "description": "GIF-focused, 8 frames",
                "latent_shape": "4x32x32 (compact for GIFs)"
            }
            print("✅ Hotshot-XL loaded")
        except Exception as e:
            print(f"❌ Hotshot failed to load: {e}")
            self._load_sd_vae()  # Fallback
    
    def _load_videogpt(self):
        """VideoGPT - Video-level VQ-VAE"""
        try:
            from videogpt.download import load_vqvae
            print("Loading VideoGPT VQ-VAE...")
            
            # Choose model based on desired resolution
            # 'bair_stride4x2x2': 16 frames of 64x64
            # 'ucf101_stride4x4x4': 16 frames of 128x128
            # 'kinetics_stride4x4x4': 16 frames of 128x128
            # 'kinetics_stride2x4x4': 16 frames of 128x128 (higher quality)
            
            # Try different models if one fails
            models_to_try = [
                'bair_stride4x2x2',      # 16 frames of 64x64
                'ucf101_stride4x4x4',    # 16 frames of 128x128  
                'kinetics_stride4x4x4',  # 16 frames of 128x128
                'kinetics_stride2x4x4'   # 16 frames of 128x128 (best)
            ]
            
            loaded = False
            for model_name in models_to_try:
                try:
                    print(f"Trying to load {model_name}...")
                    self.vqvae = load_vqvae(model_name, device=self.device)
                    loaded = True
                    break
                except Exception as e:
                    print(f"  Failed: {str(e)[:100]}...")
                    continue
            
            if not loaded:
                raise Exception("Could not load any VideoGPT model")
            self.vae = None  # VideoGPT uses different interface
            self.is_video_level = True
            
            # Get the model's latent shape info
            latent_shape = self.vqvae.latent_shape
            
            # Determine input resolution based on model
            if 'bair' in model_name:
                input_res = 64
            else:
                input_res = 128
            
            self.model_info = {
                "name": f"VideoGPT ({model_name})",
                "description": "3D VQ-VAE for video compression",
                "latent_shape": f"{latent_shape} discrete codes for 16x{input_res}x{input_res} video",
                "input_resolution": input_res
            }
            print(f"✅ VideoGPT loaded - true video-level encoding!")
            print(f"   Model: {model_name}")
            print(f"   Input: 16 frames of {input_res}x{input_res}")
            print(f"   Latent shape: {latent_shape}")
            
        except Exception as e:
            print(f"❌ VideoGPT failed to load: {e}")
            print("   Make sure VideoGPT is installed in the current environment")
            self._load_sd_vae()  # Fallback
    
    def _load_magvit(self):
        """MAGVIT - 3D video tokenizer"""
        # MAGVIT is not yet in HuggingFace transformers
        print("ℹ️  MAGVIT-v2 is not yet available in transformers library")
        print("   It's a state-of-art 3D video tokenizer from Google")
        print("   Paper: https://arxiv.org/abs/2310.05737")
        print("   Using SD VAE as fallback...")
        self._load_sd_vae()
    
    def _load_temporal_vae(self):
        """Custom Temporal VAE - extends SD VAE with temporal layers"""
        try:
            from diffusers import AutoencoderKL
            import torch.nn as nn
            print("Creating Temporal VAE (SD VAE + temporal layers)...")
            
            # Force CPU for Conv3D operations (not supported on MPS)
            temp_device = "cpu" if self.device == "mps" else self.device
            
            # Load base SD VAE
            base_vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-mse",
                torch_dtype=torch.float32
            ).to(self.device)
            
            # Add temporal compression layer
            class TemporalVAE(nn.Module):
                def __init__(self, base_vae, device, temp_device):
                    super().__init__()
                    self.base_vae = base_vae
                    self.device = device
                    self.temp_device = temp_device
                    # Temporal compression: [T, C, H, W] -> [C_temp, H, W]
                    # Keep on CPU if MPS (Conv3D not supported)
                    self.temporal_encoder = nn.Conv3d(4, 8, kernel_size=(3, 1, 1), padding=(1, 0, 0)).to(temp_device)
                    self.temporal_decoder = nn.ConvTranspose3d(8, 4, kernel_size=(3, 1, 1), padding=(1, 0, 0)).to(temp_device)
                
                def encode_video(self, video):
                    # video: [T, C, H, W]
                    # First encode each frame spatially
                    spatial_latents = []
                    for frame in video:
                        latent = self.base_vae.encode(frame.unsqueeze(0)).latent_dist.sample()
                        # Apply SD VAE scaling factor
                        latent = latent * self.base_vae.config.scaling_factor
                        spatial_latents.append(latent[0])
                    
                    # Stack and move to temp device for 3D ops
                    spatial_stack = torch.stack(spatial_latents).unsqueeze(0)  # [1, T, C, H, W]
                    spatial_stack = spatial_stack.permute(0, 2, 1, 3, 4).to(self.temp_device)  # [1, C, T, H, W]
                    
                    # Temporal compression
                    temporal_latent = self.temporal_encoder(spatial_stack)  # [1, C_temp, T, H, W]
                    temporal_latent = temporal_latent.mean(dim=2)  # Average over time: [1, C_temp, H, W]
                    
                    # Move back to main device
                    return temporal_latent.to(self.device)
                
                def decode_video(self, temporal_latent, num_frames):
                    # Move to temp device for 3D ops
                    temporal_latent = temporal_latent.to(self.temp_device)
                    
                    # Expand temporal dimension
                    expanded = temporal_latent.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)  # [1, C_temp, T, H, W]
                    
                    # Temporal decompression
                    spatial_stack = self.temporal_decoder(expanded)  # [1, C, T, H, W]
                    spatial_stack = spatial_stack.permute(0, 2, 1, 3, 4).to(self.device)  # [1, T, C, H, W]
                    
                    # Decode each frame spatially
                    frames = []
                    for t in range(num_frames):
                        frame_latent = spatial_stack[0, t].unsqueeze(0)
                        # Apply inverse SD VAE scaling factor before decoding
                        frame_latent = frame_latent / self.base_vae.config.scaling_factor
                        frame = self.base_vae.decode(frame_latent).sample[0]
                        frames.append(frame)
                    
                    return torch.stack(frames)
            
            self.temporal_vae = TemporalVAE(base_vae, self.device, temp_device)
            self.vae = None  # Use custom interface
            self.is_video_level = True
            self.model_info = {
                "name": "Temporal VAE",
                "description": "SD VAE + temporal compression (video-level)",
                "latent_shape": "8x64x64 for entire video"
            }
            if self.device == "mps":
                print("✅ Temporal VAE created - using CPU for Conv3D ops (MPS limitation)")
            else:
                print("✅ Temporal VAE created - video-level encoding!")
        except Exception as e:
            print(f"❌ Temporal VAE failed: {e}")
            self._load_sd_vae()
    
    def get_latent_shape(self, frames):
        """Get the latent shape for this model"""
        with torch.no_grad():
            # Encode first frame to get shape
            frame_tensor = torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 127.5 - 1
            frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            latent = self.vae.encode(frame_tensor).latent_dist.sample()
            return latent.shape
    
    def test_pure_noise(self, num_frames=16, latent_shape=None, latent_stats=None):
        """Decode pure random noise from latent space"""
        print(f"\n🎲 Testing PURE NOISE decoding for {self.model_type}")
        
        # Handle video-level models differently
        if self.is_video_level:
            print("Video-level model - generating single latent for entire video")
            
            # Default shape for video-level models
            if latent_shape is None:
                if "temporal_vae" in self.model_type:
                    latent_shape = (1, 8, 128, 128)  # Temporal VAE shape (matches actual implementation)
                elif "videogpt" in self.model_type:
                    # VideoGPT uses discrete codes, but we'll generate random codes
                    if hasattr(self, 'vqvae'):
                        # Get actual latent shape from model
                        latent_shape = (4, 32, 32)  # Default for kinetics model
                        print(f"VideoGPT codebook size: {self.vqvae.n_codes}")
                    else:
                        latent_shape = (4, 32, 32)  # Placeholder
                elif "magvit" in self.model_type:
                    latent_shape = (1, 8, 32, 32)  # Placeholder
                else:
                    latent_shape = (1, 8, 64, 64)  # Default video-level shape
            
            print(f"Generating random video latent with shape: {latent_shape}")
            
            # Use provided stats or defaults
            if latent_stats:
                mean, std = latent_stats
            else:
                mean, std = 0.0, 1.0
            
            # Generate single random latent for entire video
            random_latent = torch.randn(latent_shape).to(self.device) * std + mean
            
            # Decode entire video at once
            with torch.no_grad():
                if hasattr(self, 'temporal_vae'):
                    decoded_video = self.temporal_vae.decode_video(random_latent, num_frames)
                    decoded_frames = []
                    for t in range(num_frames):
                        frame = decoded_video[t]
                        frame = ((frame + 1) * 127.5).clamp(0, 255)
                        frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        decoded_frames.append(frame)
                elif hasattr(self, 'vqvae'):
                    # VideoGPT - generate random discrete codes
                    random_codes = torch.randint(0, self.vqvae.n_codes, latent_shape).to(self.device)
                    decoded_video = self.vqvae.decode(random_codes)
                    if decoded_video.dim() == 5:  # [B, C, T, H, W]
                        decoded_video = decoded_video[0].permute(1, 0, 2, 3)  # [T, C, H, W]
                    
                    decoded_frames = []
                    for t in range(min(num_frames, decoded_video.shape[0])):
                        frame = decoded_video[t]
                        frame = ((frame + 1) * 127.5).clamp(0, 255)
                        frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        
                        # Resize if needed
                        if frame.shape[:2] != (512, 512):
                            img = Image.fromarray(frame)
                            img = img.resize((512, 512), Image.Resampling.LANCZOS)
                            frame = np.array(img)
                        
                        decoded_frames.append(frame)
                else:
                    # Fallback for other video-level models
                    print("Video-level model not fully implemented, using frame-level fallback")
                    return None
            
            return decoded_frames
        
        # Frame-level models (original code)
        if latent_shape is None:
            # Default shape based on model
            if "zeroscope" in self.model_type:
                latent_shape = (1, 4, 40, 72)  # ZeroScope shape
            elif "modelscope" in self.model_type:
                latent_shape = (1, 4, 32, 32)  # ModelScope shape
            elif "svd" in self.model_type:
                latent_shape = (1, 4, 72, 128)  # SVD shape
            elif "cogvideo" in self.model_type:
                latent_shape = (1, 16, 40, 40)  # CogVideo shape - temporal compression!
            elif "show1" in self.model_type:
                latent_shape = (1, 4, 32, 32)  # Show-1 shape
            elif "hotshot" in self.model_type:
                latent_shape = (1, 4, 32, 32)  # Hotshot shape
            else:
                latent_shape = (1, 4, 64, 64)  # Default SD shape
        
        print(f"Generating random latents with shape: {latent_shape}")
        
        # Use provided stats or defaults
        if latent_stats:
            mean, std = latent_stats
            print(f"Using latent stats: mean={mean:.3f}, std={std:.3f}")
        else:
            # Typical latent space statistics for diffusion models
            mean, std = 0.0, 1.0
            print(f"Using default stats: mean={mean:.3f}, std={std:.3f}")
        
        # Generate pure random latents
        decoded_frames = []
        with torch.no_grad():
            for i in range(num_frames):
                # Generate random latent with proper distribution
                random_latent = torch.randn(latent_shape).to(self.device) * std + mean
                
                # Decode
                frame = self.vae.decode(random_latent).sample[0]
                frame = ((frame + 1) * 127.5).clamp(0, 255)
                frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                
                # Resize if needed to standard size
                if frame.shape[:2] != (512, 512):
                    img = Image.fromarray(frame)
                    img = img.resize((512, 512), Image.Resampling.LANCZOS)
                    frame = np.array(img)
                
                decoded_frames.append(frame)
        
        return decoded_frames
    
    def test_uniform_noise(self, frames, input_size, noise_levels=[1.0, 3.0, 5.0, 10.0]):
        """Test uniform noise at different levels with optimal sizing"""
        if self.is_video_level:
            return self._test_video_level_noise(frames, input_size, noise_levels)
        else:
            return self._test_frame_level_noise(frames, input_size, noise_levels)
    
    def _test_frame_level_noise(self, frames, input_size, noise_levels):
        """Test frame-by-frame noise (traditional approach)"""
        results = []
        
        # Get optimal size for this model
        optimal_size = get_optimal_size_for_model(self.model_type, input_size)
        
        # Check if we need to resize from the loaded frames
        current_size = frames[0].shape[0]  # Assume square
        
        if current_size != optimal_size:
            print(f"Resizing from {current_size}x{current_size} to {optimal_size}x{optimal_size} for {self.model_type}")
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize((optimal_size, optimal_size), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frames = resized_frames
        
        print(f"🎞️ FRAME-BY-FRAME processing: {len(frames)} frames at {optimal_size}x{optimal_size}")
        
        # Encode frames
        print("Encoding to latent space...")
        latents = []
        with torch.no_grad():
            for frame in frames:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
                latent = self.vae.encode(frame_tensor).latent_dist.sample()
                latents.append(latent[0])
        
        latents = torch.stack(latents)
        print(f"Latent shape: {latents.shape} (each frame independent)")
        print(f"Latent stats: mean={latents.mean():.3f}, std={latents.std():.3f}")
        
        # Store latent stats for pure noise generation
        self._last_latent_stats = (latents.mean().item(), latents.std().item())
        
        # Test different noise levels
        for noise_level in noise_levels:
            print(f"\nApplying frame-by-frame noise (strength={noise_level})...")
            
            # Generate noise with same scale as latents
            noise = torch.randn_like(latents) * latents.std() + latents.mean()
            
            # Interpolate between original and noise
            alpha = min(1.0, noise_level / 10.0)
            noisy_latents = (1 - alpha) * latents + alpha * noise
            
            print(f"  Each frame gets independent noise (alpha: {alpha:.2f})")
            
            # Decode
            decoded_frames = []
            with torch.no_grad():
                for latent in noisy_latents:
                    frame = self.vae.decode(latent.unsqueeze(0)).sample[0]
                    frame = ((frame + 1) * 127.5).clamp(0, 255)
                    frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    # Resize back to 512x512 if needed
                    if frame.shape[:2] != (512, 512):
                        img = Image.fromarray(frame)
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        frame = np.array(img)
                    
                    decoded_frames.append(frame)
            
            results.append((noise_level, decoded_frames))
        
        return results
    
    def _test_video_level_noise(self, frames, input_size, noise_levels):
        """Test video-level noise (whole video gets one latent)"""
        results = []
        
        optimal_size = get_optimal_size_for_model(self.model_type, input_size)
        current_size = frames[0].shape[0]
        
        if current_size != optimal_size:
            print(f"Resizing from {current_size}x{current_size} to {optimal_size}x{optimal_size} for {self.model_type}")
            resized_frames = []
            for frame in frames:
                img = Image.fromarray(frame)
                img = img.resize((optimal_size, optimal_size), Image.Resampling.LANCZOS)
                resized_frames.append(np.array(img))
            frames = resized_frames
        
        print(f"🎬 VIDEO-LEVEL processing: entire video → single latent")
        
        # Convert frames to video tensor
        video_tensor = torch.stack([
            torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5 - 1
            for frame in frames
        ]).to(self.device)
        
        # Encode entire video
        print("Encoding entire video to single latent...")
        with torch.no_grad():
            if hasattr(self, 'temporal_vae'):
                # Custom temporal VAE
                video_latent = self.temporal_vae.encode_video(video_tensor)
                latent_type = "continuous"
            elif hasattr(self, 'vqvae'):
                # VideoGPT - uses discrete codes
                # VideoGPT expects [B, C, T, H, W] format
                video_batch = video_tensor.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
                
                # Encode to discrete codes and get embeddings
                encodings, embeddings = self.vqvae.encode(video_batch, include_embeddings=True)
                video_latent = embeddings  # Use continuous embeddings for manipulation
                self._video_encodings = encodings  # Store discrete codes for decoding
                latent_type = "discrete (VQ-VAE)"
                print(f"  Discrete codes shape: {encodings.shape}")
                print(f"  Embedding shape: {embeddings.shape}")
            else:
                # Fallback
                video_latent = torch.randn(1, 8, optimal_size//8, optimal_size//8).to(self.device)
                latent_type = "fallback"
        
        print(f"Video latent shape: {video_latent.shape} ({latent_type})")
        
        # Store stats
        if video_latent.dtype == torch.float32:
            self._last_latent_stats = (video_latent.mean().item(), video_latent.std().item())
        else:
            self._last_latent_stats = (0.0, 1.0)  # Default for discrete
        
        # Test different noise levels
        for noise_level in noise_levels:
            print(f"\nApplying video-level noise (strength={noise_level})...")
            
            if hasattr(self, 'vqvae'):
                # For VQ-VAE, we manipulate embeddings then re-quantize
                noise = torch.randn_like(video_latent) * video_latent.std() + video_latent.mean()
                alpha = min(1.0, noise_level / 10.0)
                noisy_embeddings = (1 - alpha) * video_latent + alpha * noise
                
                # Re-quantize the noisy embeddings
                h = noisy_embeddings.permute(0, 2, 3, 4, 1)  # [B, T, H, W, D]
                distances = torch.cdist(h.reshape(-1, h.shape[-1]), 
                                      self.vqvae.codebook.embeddings, p=2)
                noisy_encodings = distances.argmin(dim=-1).reshape(h.shape[:-1])
                noisy_video_latent = noisy_encodings
            else:
                # Continuous latents (temporal VAE)
                noise = torch.randn_like(video_latent) * video_latent.std() + video_latent.mean()
                alpha = min(1.0, noise_level / 10.0)
                noisy_video_latent = (1 - alpha) * video_latent + alpha * noise
            
            print(f"  Single noise affects entire video (alpha: {alpha:.2f})")
            
            # Decode entire video
            with torch.no_grad():
                if hasattr(self, 'temporal_vae'):
                    decoded_video = self.temporal_vae.decode_video(noisy_video_latent, len(frames))
                elif hasattr(self, 'vqvae'):
                    # VideoGPT decode expects discrete codes
                    decoded_video = self.vqvae.decode(noisy_video_latent)
                    if decoded_video.dim() == 5:  # [B, C, T, H, W]
                        decoded_video = decoded_video[0].permute(1, 0, 2, 3)  # [T, C, H, W]
                else:
                    # Fallback - just use noisy original
                    decoded_video = video_tensor
            
            # Convert to frames
            decoded_frames = []
            for t in range(len(frames)):
                if t < decoded_video.shape[0]:
                    frame = decoded_video[t]
                    frame = ((frame + 1) * 127.5).clamp(0, 255)
                    frame = frame.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    
                    # Resize if needed
                    if frame.shape[:2] != (512, 512):
                        img = Image.fromarray(frame)
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        frame = np.array(img)
                    
                    decoded_frames.append(frame)
                else:
                    # Pad with last frame if needed
                    decoded_frames.append(decoded_frames[-1] if decoded_frames else frames[t])
            
            results.append((noise_level, decoded_frames))
        
        return results

def create_model_report(model_results, output_dir):
    """Create a visual report for each model"""
    from PIL import Image, ImageDraw, ImageFont
    
    for model_name, data in model_results.items():
        if not data:
            continue
            
        print(f"\nCreating report for {model_name}...")
        
        # Create grid: rows = noise levels + pure noise, cols = sample frames
        noise_levels = [r[0] for r in data.get('uniform', [])]
        num_rows = len(noise_levels) + 2  # +1 for original, +1 for pure noise
        num_cols = 5  # Show 5 sample frames
        
        cell_size = 128
        padding = 5
        label_height = 30
        
        grid_width = num_cols * (cell_size + padding) + padding
        grid_height = num_rows * (cell_size + padding) + padding + label_height
        
        grid = Image.new('RGB', (grid_width, grid_height), color='black')
        draw = ImageDraw.Draw(grid)
        
        try:
            font = ImageFont.load_default()
        except:
            font = None
        
        # Add title
        if font:
            info = data.get('info', {})
            title = f"{info.get('name', model_name)} - {info.get('description', '')}"
            draw.text((10, 5), title, fill='white', font=font)
        
        y = label_height
        
        # Original frames
        if 'original' in data and data['original']:
            x = padding
            if font:
                draw.text((x, y + cell_size//2), "Original", fill='white', font=font)
            x += cell_size + padding
            
            frames = data['original']
            frame_indices = np.linspace(0, len(frames)-1, num_cols-1, dtype=int)
            for idx in frame_indices:
                if idx < len(frames):
                    img = Image.fromarray(frames[idx])
                    img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                    grid.paste(img, (x, y))
                x += cell_size + padding
            y += cell_size + padding
        
        # Uniform noise results
        if 'uniform' in data:
            for noise_level, frames in data['uniform']:
                x = padding
                if font:
                    draw.text((x, y + cell_size//2), f"Noise {noise_level}", fill='white', font=font)
                x += cell_size + padding
                
                frame_indices = np.linspace(0, len(frames)-1, num_cols-1, dtype=int)
                for idx in frame_indices:
                    if idx < len(frames):
                        img = Image.fromarray(frames[idx])
                        img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                        grid.paste(img, (x, y))
                    x += cell_size + padding
                y += cell_size + padding
        
        # Pure noise results
        if 'pure_noise' in data and data['pure_noise']:
            x = padding
            if font:
                draw.text((x, y + cell_size//2), "Pure Noise", fill='white', font=font)
            x += cell_size + padding
            
            frames = data['pure_noise']
            frame_indices = np.linspace(0, len(frames)-1, num_cols-1, dtype=int)
            for idx in frame_indices:
                if idx < len(frames):
                    img = Image.fromarray(frames[idx])
                    img = img.resize((cell_size, cell_size), Image.Resampling.LANCZOS)
                    grid.paste(img, (x, y))
                x += cell_size + padding
        
        # Save report
        report_path = output_dir / f"report_{model_name}.png"
        grid.save(str(report_path))
        print(f"✅ Saved report: {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--models', type=str, default='all', 
                        help='Comma-separated list or "all"')
    parser.add_argument('--noise', type=str, default='1,2,3,4,5,6,7,8,9', 
                        help='Comma-separated noise levels')
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--output', type=str, default='output_model_comparison')
    
    args = parser.parse_args()
    
    # Define available models
    available_models = [
        'sd_vae',
        'sd_vae_mse',
        'sd_vae_ema',
        'animatediff',
        'zeroscope',
        'modelscope',
        'svd',
        'cogvideo',  # Added: 2B model with compact latents
        'show1',     # Added: Compact latents
        'hotshot',   # Added: GIF-focused, compact
        'videogpt',  # Added: Video-level VQ-VAE
        'magvit',    # Added: 3D tokenizer
        'temporal_vae'  # Added: Custom temporal VAE
    ]
    
    # Parse models to test
    if args.models == 'all':
        models_to_test = available_models
    else:
        models_to_test = args.models.split(',')
    
    # Parse noise levels
    noise_levels = [float(x) for x in args.noise.split(',')]
    
    print("="*70)
    print("COMPREHENSIVE VIDEO MODEL COMPARISON")
    print("="*70)
    print(f"Video: {args.video}")
    print(f"Models to test: {models_to_test}")
    print(f"Noise levels: {noise_levels}")
    print(f"Frames: {args.frames}")
    print("="*70)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load video once with optimal sizing
    print("\nLoading video...")
    frames, input_size = load_video_frames(args.video, num_frames=args.frames)
    print(f"Loaded {len(frames)} frames")
    
    # Test each model
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING MODEL: {model_name}")
        print("="*60)
        
        try:
            # Initialize model
            tester = VideoModelTester(model_name)
            
            # Store model info
            model_results = {
                'info': tester.model_info,
                'original': frames
            }
            
            # Test uniform noise
            uniform_results = tester.test_uniform_noise(frames, input_size, noise_levels)
            model_results['uniform'] = uniform_results
            
            # Save uniform noise videos
            for noise_level, decoded_frames in uniform_results:
                path = output_dir / f"{model_name}_noise_{noise_level}.mp4"
                save_video_frames(decoded_frames, str(path), fps=8)
                print(f"✅ Saved: {path}")
            
            # Test pure noise decoding
            print(f"\n{'='*40}")
            print("Testing pure noise decoding...")
            # Get latent stats from the actual video if we have uniform results
            latent_stats = None
            if hasattr(tester, '_last_latent_stats'):
                latent_stats = tester._last_latent_stats
            pure_noise_frames = tester.test_pure_noise(num_frames=args.frames, latent_stats=latent_stats)
            model_results['pure_noise'] = pure_noise_frames
            
            # Save pure noise video
            path = output_dir / f"{model_name}_pure_noise.mp4"
            save_video_frames(pure_noise_frames, str(path), fps=8)
            print(f"✅ Saved: {path}")
            
            # Store results
            all_results[model_name] = model_results
            
        except Exception as e:
            print(f"❌ Failed to test {model_name}: {e}")
            continue
    
    # Create visual reports
    print(f"\n{'='*60}")
    print("Creating visual reports...")
    create_model_report(all_results, output_dir)
    
    # Save original video  
    save_video_frames(frames, str(output_dir / "original.mp4"), fps=8)
    
    print(f"\n{'='*70}")
    print("✅ MODEL COMPARISON COMPLETE!")
    print(f"Results saved to: {output_dir}/")
    print("\nWhat to look for:")
    print("1. Uniform noise: How structured are the glitches?")
    print("2. Pure noise: What does each model 'imagine' from random latents?")
    print("3. Model differences:")
    print("   - SD VAE: Basic pixel patterns")
    print("   - AnimateDiff: Motion-coherent patterns")
    print("   - ZeroScope/ModelScope: Video-specific patterns")
    print("   - SVD: High-quality video patterns")
    print("="*70)

if __name__ == "__main__":
    main()