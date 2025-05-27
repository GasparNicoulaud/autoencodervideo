# Video Autoencoder Project Status

## Current State (May 27, 2024)

### What Works ✅
- **Real VAE Integration**: Successfully using Stable Diffusion's VAE from HuggingFace
- **Video Processing**: Fixed video saving issues - videos now display correctly with proper value range conversion
- **Latent Manipulation**: All latent space operations work with real VAE
- **M1 Optimization**: Using MPS (Metal Performance Shaders) when available
- **Project Cleaned**: Removed all unused directories and files

### Working Example

#### Add Noise to Latent Space (Clean Demo)
```bash
python experiments/add_latent_noise.py
```
- Creates a simple test video with moving circle
- Encodes it with Stable Diffusion's VAE
- Saves three outputs:
  - `original.mp4` - Input video
  - `reconstructed.mp4` - VAE reconstruction (no modifications)
  - `reconstructed_with_noise.mp4` - VAE reconstruction with noise added to latent space
- Shows how latent space modifications affect output
- Uses CPU for stability on M1

### Project Structure (After Cleanup)
```
autoencodervideo/
├── CLAUDE.md           # This file - project status & notes
├── README.md           # User documentation
├── requirements.txt    # All dependencies
├── setup.py           # Package setup (updated)
├── src/               # Source code
│   ├── __init__.py
│   ├── models/        # Model architectures
│   │   ├── __init__.py
│   │   ├── video_autoencoder.py    # VAE implementation
│   │   ├── vqvae.py                # VQ-VAE implementation
│   │   └── huggingface_models.py   # Real model integration
│   ├── latent/        # Latent manipulation tools
│   │   ├── __init__.py
│   │   ├── manipulator.py          # Weight/bias control
│   │   ├── interpolation.py        # Interpolation methods
│   │   └── analysis.py             # Latent analysis tools
│   └── utils/         # Utilities
│       ├── __init__.py
│       ├── video_io.py             # Video loading/saving
│       └── training.py             # Training helpers
├── experiments/       # Single clean experiment
│   └── add_latent_noise.py             # Add noise to latent space demo
├── notebooks/         
│   └── autoencoder_tutorial.ipynb      # Jupyter tutorial
├── scripts/           
│   └── download_pretrained.py          # Model download helper
└── output/           # Generated videos go here
```

### M1 Mac Optimizations

#### Current Implementation:
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
vae = AutoencoderKL.from_pretrained("...", torch_dtype=torch.float32).to(device)
```

#### Performance Notes:
1. **MPS is faster for larger models** - Especially for VAE operations
2. **Float32 required** - Float16 can cause issues on MPS
3. **Batch processing** - M1 Max benefits from larger batch sizes (4-8)
4. **Memory efficient** - Unified memory allows large models without OOM

#### To Benchmark Your M1 Max:
```bash
python experiments/benchmark_mps.py
```

### Key Discoveries & Fixes

1. **Video Black Screen Issue**: 
   - Problem: `save_video()` function had incorrect value range conversion
   - Fix: Proper conversion from [-1,1] to [0,255] with clipping
   - Solution in: `test_vae_with_fixed_save.py`

2. **Real Model Integration**:
   - Stable Diffusion VAE works perfectly
   - Compresses 256x256 → 32x32x4 latents
   - Much better than toy models

3. **Memory Management on M1**:
   - MPS can run out of memory with many frames
   - Solution: Use CPU for stability or reduce batch size
   - Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to disable limits

4. **Movement in Latent Space**:
   - VAE successfully encodes motion
   - Interpolation in latent space creates smooth transitions
   - Can do arithmetic operations on motion

5. **Dependencies That Matter**:
   - `diffusers` - For real models
   - `torch` - Core framework
   - `imageio` - Video I/O
   - `einops` - Tensor operations

### Common Commands

```bash
# Activate environment
source venv/bin/activate

# Test real VAE
python experiments/test_vae_with_fixed_save.py

# Generate colorful patterns
python experiments/generate_colorful_samples.py --pattern rainbow

# Run M1 benchmark
python experiments/benchmark_mps.py

# Start Jupyter notebook
jupyter notebook notebooks/autoencoder_tutorial.ipynb
```

### TODO / Future Improvements
- [ ] Add config system if project grows
- [ ] Implement video dataset loading
- [ ] Add more pretrained model options
- [ ] Create web UI for easier experimentation
- [ ] Add unit tests

### Issues & Solutions Log

| Issue | Solution | File |
|-------|----------|------|
| Black videos | Fixed value range conversion | `test_vae_with_fixed_save.py` |
| NaN losses | Used stable architecture & clipping | `stable_autoencoder_demo.py` |
| Import errors | Made optional imports (cv2, wandb) | `video_io.py`, `training.py` |
| MPS compatibility | Use float32 instead of float16 | All model loading code |

### Notes
- Project started as framework demo with synthetic data
- Now fully integrated with real Stable Diffusion VAE
- Ready for real video experiments with proper models
- All unused files and directories have been cleaned up
- Removed `pretrained.py` as it only had placeholder URLs
- Cleaned experiments folder to contain only one focused demo
- Output folder contains only the essential outputs from the latest run