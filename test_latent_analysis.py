#!/usr/bin/env python3
"""
Test latent space analysis capabilities
"""
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.models import VideoAutoencoder
from src.latent import LatentManipulator, LatentAnalyzer, interpolate_latents

def main():
    print("Testing Latent Space Analysis...")
    
    # Initialize model
    model = VideoAutoencoder(latent_dim=128, base_channels=32)
    model.eval()
    
    # Test LatentManipulator
    print("\n1. Testing LatentManipulator:")
    manipulator = LatentManipulator(model)
    
    # Get weight statistics
    stats = manipulator.compute_weight_statistics()
    print(f"   Found {len(stats)} parameter groups")
    for layer, layer_stats in list(stats.items())[:2]:
        print(f"   {layer}: {list(layer_stats.keys())}")
    
    # Test weight manipulation
    manipulator.save_original_params()
    manipulator.scale_weights('decoder_fc', 1.5)
    print("   ✓ Weight scaling applied")
    
    manipulator.restore_original_params()
    print("   ✓ Weights restored")
    
    # Test interpolation
    print("\n2. Testing Interpolation:")
    z1 = torch.randn(1, 128)
    z2 = torch.randn(1, 128)
    
    # Linear interpolation
    linear_path = interpolate_latents(z1[0], z2[0], steps=5, method='linear')
    print(f"   ✓ Linear interpolation: {linear_path.shape}")
    
    # Spherical interpolation
    spherical_path = interpolate_latents(z1[0], z2[0], steps=5, method='spherical')
    print(f"   ✓ Spherical interpolation: {spherical_path.shape}")
    
    # Test custom transformations
    print("\n3. Testing Custom Transformations:")
    
    # Pattern 1: Amplify specific dimensions
    z_test = torch.randn(1, 128)
    z_amplified = z_test.clone()
    z_amplified[:, :10] *= 3.0
    print(f"   ✓ Dimension amplification applied")
    
    # Pattern 2: Apply mask
    mask = torch.rand(128) > 0.5
    z_masked = z_test * mask.float()
    print(f"   ✓ Sparse mask applied ({mask.sum().item()} active dims)")
    
    # Pattern 3: Non-linear transform
    z_nonlinear = torch.tanh(z_test * 2.0)
    print(f"   ✓ Non-linear transform applied")
    
    print("\n4. Testing Programmatic Control Patterns:")
    
    # Test different activation patterns
    patterns = {
        'sine': lambda z: z * torch.sin(torch.linspace(0, 2*3.14159, z.shape[1])),
        'exponential': lambda z: z * torch.exp(-torch.linspace(0, 3, z.shape[1])),
        'step': lambda z: z * (torch.arange(z.shape[1]) % 10 < 5).float()
    }
    
    for name, pattern_fn in patterns.items():
        z_pattern = pattern_fn(z_test)
        print(f"   ✓ {name} pattern: mean={z_pattern.mean():.3f}, std={z_pattern.std():.3f}")
    
    print("\nAll tests completed successfully!")
    print("\nYou can now:")
    print("1. Load real videos and manipulate their latent representations")
    print("2. Train the model on your video dataset")
    print("3. Experiment with more complex latent space operations")

if __name__ == '__main__':
    main()