#!/usr/bin/env python3
"""
Use actual pretrained video models from Hugging Face
"""
import torch
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import save_video, create_video_grid

def use_stable_diffusion_vae():
    """Use Stable Diffusion's VAE as a frame-by-frame video autoencoder"""
    try:
        from diffusers import AutoencoderKL
        
        print("Loading Stable Diffusion VAE...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae.eval()
        
        # Generate test frames
        frames = []
        for i in range(8):
            # Create colorful test pattern
            frame = torch.zeros(1, 3, 512, 512)
            # Add gradient
            x = torch.linspace(-1, 1, 512)
            y = torch.linspace(-1, 1, 512)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            
            frame[0, 0] = torch.sin(xx * 5 + i * 0.5) * 0.5
            frame[0, 1] = torch.sin(yy * 5 + i * 0.5) * 0.5
            frame[0, 2] = torch.sin((xx + yy) * 3 + i * 0.5) * 0.5
            
            frames.append(frame)
        
        # Encode and decode each frame
        latents = []
        reconstructed = []
        
        with torch.no_grad():
            for frame in frames:
                # Encode
                latent = vae.encode(frame).latent_dist.sample()
                latents.append(latent)
                
                # Decode
                recon = vae.decode(latent).sample
                reconstructed.append(recon)
        
        # Stack into video
        video = torch.cat(reconstructed, dim=0)  # (T, C, H, W)
        video = video.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        save_video(video.unsqueeze(0), "output/sd_vae_test.mp4", denormalize=False)
        print("Saved: output/sd_vae_test.mp4")
        
        # Test latent manipulation
        print("\nTesting latent manipulations...")
        base_latent = latents[0]
        
        manipulations = []
        for scale in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
            modified = base_latent * scale
            decoded = vae.decode(modified).sample
            manipulations.append(decoded[0])
        
        # Save manipulations
        for i, (scale, frame) in enumerate(zip([0.5, 0.8, 1.0, 1.2, 1.5, 2.0], manipulations)):
            save_video(frame.unsqueeze(0).unsqueeze(0), f"output/sd_vae_scale_{scale}.mp4", denormalize=False)
        
        return True
        
    except ImportError:
        print("Please install diffusers: pip install diffusers")
        return False


def create_simple_trained_autoencoder():
    """Create and quickly train a simple autoencoder on patterns"""
    from src.models import VideoAutoencoder
    
    print("Creating and training a simple autoencoder on patterns...")
    
    # Smaller model for faster training
    model = VideoAutoencoder(latent_dim=64, base_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate training data - simple patterns
    def generate_pattern_video(pattern_type, size=32, frames=8):
        video = torch.zeros(3, frames, size, size)
        
        for t in range(frames):
            if pattern_type == 'checkerboard':
                for i in range(0, size, 4):
                    for j in range(0, size, 4):
                        if ((i//4) + (j//4)) % 2 == 0:
                            video[:, t, i:i+4, j:j+4] = torch.rand(3, 1, 1) * 2 - 1
            elif pattern_type == 'stripes':
                for i in range(size):
                    if (i + t) % 8 < 4:
                        video[:, t, i, :] = torch.rand(3, 1) * 2 - 1
            elif pattern_type == 'circles':
                center = size // 2
                for i in range(size):
                    for j in range(size):
                        dist = ((i - center)**2 + (j - center)**2)**0.5
                        if int(dist + t*2) % 8 < 4:
                            video[:, t, i, j] = torch.rand(3) * 2 - 1
        
        return video
    
    # Train for a few iterations
    print("Training on pattern videos...")
    model.train()
    losses = []
    
    for epoch in range(50):
        epoch_loss = 0
        for _ in range(10):  # 10 batches per epoch
            # Generate random batch
            batch = []
            for _ in range(4):  # batch size 4
                pattern = np.random.choice(['checkerboard', 'stripes', 'circles'])
                video = generate_pattern_video(pattern)
                batch.append(video)
            
            batch = torch.stack(batch)
            
            # Forward pass
            recon, mu, log_var, z = model(batch)
            loss_dict = model.loss_function(recon, batch, mu, log_var)
            loss = loss_dict['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 10
        losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "output/simple_trained_autoencoder.pth")
    print("Saved trained model to output/simple_trained_autoencoder.pth")
    
    # Test the trained model
    model.eval()
    test_videos = []
    
    with torch.no_grad():
        # Generate test patterns
        for pattern in ['checkerboard', 'stripes', 'circles']:
            video = generate_pattern_video(pattern).unsqueeze(0)
            
            # Encode and decode
            recon, _, _, z = model(video)
            test_videos.append(recon[0])
            
            # Also test latent manipulation
            z_scaled = z * 1.5
            recon_scaled = model.decode(z_scaled)
            test_videos.append(recon_scaled[0])
            
            z_noisy = z + torch.randn_like(z) * 0.3
            recon_noisy = model.decode(z_noisy)
            test_videos.append(recon_noisy[0])
    
    # Create grid
    grid = create_video_grid(test_videos, grid_size=(3, 3))
    save_video(grid.unsqueeze(0), "output/trained_model_test_grid.mp4")
    print("Saved test grid to output/trained_model_test_grid.mp4")
    
    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['sd-vae', 'train-simple'], default='train-simple')
    args = parser.parse_args()
    
    if args.model == 'sd-vae':
        success = use_stable_diffusion_vae()
        if not success:
            print("\nFalling back to training simple model...")
            create_simple_trained_autoencoder()
    else:
        create_simple_trained_autoencoder()
    
    print("\nNow the autoencoder is trained and should produce meaningful results!")
    print("You can load and use it with:")
    print("  model = VideoAutoencoder(latent_dim=64, base_channels=16)")
    print("  model.load_state_dict(torch.load('output/simple_trained_autoencoder.pth'))")
    print("  model.eval()")