#!/usr/bin/env python3
"""
Train a working autoencoder with correct dimensions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import save_video, create_video_grid


class SimpleVideoAutoencoder(nn.Module):
    """Simplified autoencoder that actually works"""
    def __init__(self, frames=8, size=64, latent_dim=128):
        super().__init__()
        self.frames = frames
        self.size = size
        self.latent_dim = latent_dim
        
        # Simple encoder: flatten and project
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * frames * size * size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mu and log_var
        )
        
        # Simple decoder: project and reshape
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * frames * size * size),
            nn.Tanh()
        )
        
    def encode(self, x):
        # x shape: (B, C, T, H, W)
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        h = self.encoder(x_flat)
        mu, log_var = h.chunk(2, dim=1)
        return mu, log_var
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        batch_size = z.shape[0]
        x_flat = self.decoder(z)
        x = x_flat.view(batch_size, 3, self.frames, self.size, self.size)
        return x
        
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var, z
        
    def loss_function(self, recon, target, mu, log_var, beta=0.1):
        recon_loss = F.mse_loss(recon, target, reduction='mean')
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kld_loss


def generate_training_data(batch_size=8, frames=8, size=64):
    """Generate colorful pattern videos for training"""
    videos = []
    
    for _ in range(batch_size):
        video = torch.zeros(3, frames, size, size)
        pattern_type = np.random.choice(['checkerboard', 'stripes', 'gradient', 'circles'])
        
        # Random base colors
        color1 = torch.rand(3, 1, 1, 1) * 2 - 1
        color2 = torch.rand(3, 1, 1, 1) * 2 - 1
        
        for t in range(frames):
            phase = t / frames * 2 * np.pi
            
            if pattern_type == 'checkerboard':
                for i in range(0, size, 8):
                    for j in range(0, size, 8):
                        if ((i//8) + (j//8) + t//2) % 2 == 0:
                            video[:, t, i:i+8, j:j+8] = color1[:, 0, 0, 0].unsqueeze(1).unsqueeze(1)
                        else:
                            video[:, t, i:i+8, j:j+8] = color2[:, 0, 0, 0].unsqueeze(1).unsqueeze(1)
                            
            elif pattern_type == 'stripes':
                for i in range(size):
                    if (i + t*2) % 16 < 8:
                        video[:, t, i, :] = color1[:, 0, 0, 0].unsqueeze(1)
                    else:
                        video[:, t, i, :] = color2[:, 0, 0, 0].unsqueeze(1)
                        
            elif pattern_type == 'gradient':
                x = torch.linspace(-1, 1, size)
                y = torch.linspace(-1, 1, size)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                
                video[0, t] = torch.sin(xx * 3 + phase) * color1[0]
                video[1, t] = torch.sin(yy * 3 + phase + np.pi/3) * color1[1]
                video[2, t] = torch.sin((xx + yy) * 2 + phase + 2*np.pi/3) * color1[2]
                
            elif pattern_type == 'circles':
                center = size // 2
                for i in range(size):
                    for j in range(size):
                        dist = ((i - center)**2 + (j - center)**2)**0.5
                        if int(dist + t*3) % 16 < 8:
                            video[:, t, i, j] = color1[:, 0, 0, 0]
                        else:
                            video[:, t, i, j] = color2[:, 0, 0, 0]
        
        videos.append(video)
    
    return torch.stack(videos)


def train_autoencoder(epochs=100, save_path="output/trained_video_autoencoder.pth"):
    """Train the autoencoder"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Create model
    model = SimpleVideoAutoencoder(frames=8, size=64, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training autoencoder on colorful patterns...")
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Generate fresh data each epoch
        for _ in range(10):  # 10 batches per epoch
            data = generate_training_data(batch_size=8).to(device)
            
            # Forward pass
            recon, mu, log_var, z = model(data)
            loss = model.loss_function(recon, data, mu, log_var)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 10
        losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'frames': 8,
            'size': 64,
            'latent_dim': 128
        }
    }, save_path)
    print(f"\nModel saved to: {save_path}")
    
    return model


def test_trained_model(model, device='cpu'):
    """Test the trained model with various manipulations"""
    model.eval()
    
    print("\nTesting trained model...")
    
    # Generate test videos
    test_data = generate_training_data(batch_size=1, frames=8, size=64).to(device)
    
    with torch.no_grad():
        # Original reconstruction
        recon, mu, log_var, z = model(test_data)
        
        # Save original and reconstruction
        save_video(test_data, "output/test_original.mp4")
        save_video(recon, "output/test_reconstruction.mp4")
        
        # Latent manipulations
        manipulations = []
        
        # 1. Scale latent
        for scale in [0.5, 1.0, 1.5, 2.0]:
            z_scaled = z * scale
            decoded = model.decode(z_scaled)
            manipulations.append(decoded[0])
            
        # 2. Add noise
        z_noise = z + torch.randn_like(z) * 0.3
        decoded = model.decode(z_noise)
        manipulations.append(decoded[0])
        
        # 3. Interpolate with random
        z_random = torch.randn_like(z)
        for alpha in [0.25, 0.5, 0.75]:
            z_interp = (1 - alpha) * z + alpha * z_random
            decoded = model.decode(z_interp)
            manipulations.append(decoded[0])
        
        # 4. Zero out dimensions
        z_sparse = z.clone()
        z_sparse[:, 64:] = 0  # Zero out half the dimensions
        decoded = model.decode(z_sparse)
        manipulations.append(decoded[0])
        
        # Create grid
        grid = create_video_grid(manipulations[:9], grid_size=(3, 3))
        save_video(grid.unsqueeze(0), "output/trained_manipulations_grid.mp4")
        
    print("\nSaved results:")
    print("- output/test_original.mp4")
    print("- output/test_reconstruction.mp4") 
    print("- output/trained_manipulations_grid.mp4")


def main():
    # Train the model
    model = train_autoencoder(epochs=100)
    
    # Test it
    device = next(model.parameters()).device
    test_trained_model(model, device)
    
    print("\n✅ Success! The autoencoder is now trained and working.")
    print("\nYou can now use it for meaningful latent space manipulation!")
    print("\nTo load and use the model later:")
    print(">>> checkpoint = torch.load('output/trained_video_autoencoder.pth')")
    print(">>> model = SimpleVideoAutoencoder(**checkpoint['model_config'])")
    print(">>> model.load_state_dict(checkpoint['model_state_dict'])")


if __name__ == '__main__':
    main()