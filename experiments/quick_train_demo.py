#!/usr/bin/env python3
"""
Quick training demo - trains faster for immediate results
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import save_video, create_video_grid
from train_working_autoencoder import SimpleVideoAutoencoder, generate_training_data


def quick_train(epochs=20):
    """Quick training for demo purposes"""
    device = torch.device('cpu')  # CPU is fine for small model
    
    # Smaller model for faster training
    model = SimpleVideoAutoencoder(frames=8, size=32, latent_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Higher learning rate
    
    print("Quick training (20 epochs)...")
    
    for epoch in range(epochs):
        # Just 5 batches per epoch for speed
        epoch_loss = 0
        for _ in range(5):
            data = generate_training_data(batch_size=4, frames=8, size=32).to(device)
            
            recon, mu, log_var, z = model(data)
            loss = model.loss_function(recon, data, mu, log_var, beta=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}, Loss: {epoch_loss/5:.4f}")
    
    print("\nTraining complete! Testing...")
    
    # Test the model
    model.eval()
    
    # Generate test batch
    test_data = generate_training_data(batch_size=9, frames=8, size=32)
    
    with torch.no_grad():
        # Get reconstructions
        recon, mu, log_var, z = model(test_data)
        
        # Create comparison grid: original vs reconstruction
        comparison = []
        for i in range(9):
            comparison.append(test_data[i])  # Original
        for i in range(9):
            comparison.append(recon[i])  # Reconstruction
        
        grid = create_video_grid(comparison[:18], grid_size=(2, 9))
        save_video(grid.unsqueeze(0), "output/quick_train_comparison.mp4")
        
        # Test latent manipulations
        manipulations = []
        base_z = z[0:1]  # Take first sample
        
        # Different scales
        for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            z_scaled = base_z * scale
            decoded = model.decode(z_scaled)
            manipulations.append(decoded[0])
        
        # Interpolation between two samples
        z1, z2 = z[0:1], z[1:2]
        for alpha in [0.0, 0.33, 0.66, 1.0]:
            z_interp = (1 - alpha) * z1 + alpha * z2
            decoded = model.decode(z_interp)
            manipulations.append(decoded[0])
        
        # Random directions
        for _ in range(5):
            z_random = base_z + torch.randn_like(base_z) * 0.5
            decoded = model.decode(z_random)
            manipulations.append(decoded[0])
        
        manip_grid = create_video_grid(manipulations[:15], grid_size=(3, 5))
        save_video(manip_grid.unsqueeze(0), "output/quick_train_manipulations.mp4")
    
    print("\n✅ Success! Check these files:")
    print("- output/quick_train_comparison.mp4 (top row: original, bottom: reconstruction)")
    print("- output/quick_train_manipulations.mp4 (various latent manipulations)")
    
    # Save model
    torch.save(model.state_dict(), "output/quick_trained_model.pth")
    print("\nModel saved to: output/quick_trained_model.pth")
    
    return model


if __name__ == '__main__':
    model = quick_train(epochs=20)
    
    print("\n🎉 The autoencoder is now trained!")
    print("The grey videos should now show actual patterns and reconstructions.")
    print("\nTry opening the generated videos to see the results!")