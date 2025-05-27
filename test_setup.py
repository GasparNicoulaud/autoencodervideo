#!/usr/bin/env python3
"""
Test script to verify the setup
"""
import sys
from pathlib import Path

print("Testing Video Autoencoder Setup...")
print(f"Python version: {sys.version}")

# Test imports
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} imported successfully")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ Failed to import torch: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} imported successfully")
except ImportError as e:
    print(f"✗ Failed to import cv2: {e}")

# Test local imports
sys.path.append(str(Path(__file__).parent))

try:
    from src.models import VideoAutoencoder
    print("✓ VideoAutoencoder imported successfully")
    
    # Test model creation
    model = VideoAutoencoder(latent_dim=256, base_channels=32)
    print("✓ Model created successfully")
    
    # Test forward pass with dummy data
    dummy_input = torch.randn(1, 3, 8, 64, 64)  # (B, C, T, H, W)
    recon, mu, log_var, z = model(dummy_input)
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Reconstruction shape: {recon.shape}")
    print(f"  Latent shape: {z.shape}")
    
except Exception as e:
    print(f"✗ Failed to test model: {e}")

try:
    from src.latent import LatentManipulator
    print("✓ LatentManipulator imported successfully")
except ImportError as e:
    print(f"✗ Failed to import LatentManipulator: {e}")

print("\nSetup test complete!")
print("\nTo run examples:")
print("1. Generate random samples:")
print("   python experiments/programmatic_control.py")
print("\n2. With video files:")
print("   python experiments/latent_manipulation_demo.py --video your_video.mp4")
print("   python experiments/interpolation_demo.py --video1 vid1.mp4 --video2 vid2.mp4")