# Video Autoencoder Project Status

## Current State (May 27, 2024)

### What Works ✅
- **Real VAE Integration**: Successfully using Stable Diffusion's VAE from HuggingFace
- **Video Processing**: Fixed video saving issues - videos now display correctly with proper value range conversion
- **Latent Manipulation**: All latent space operations work with real VAE
- **M1 Optimization**: Using MPS (Metal Performance Shaders) when available
- **Project Cleaned**: Removed all unused directories and files

### Working Examples

#### 1. Add Noise to Latent Space (Simple Demo)
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

#### 2. Process iPhone Videos
```bash
python experiments/process_iphone_video.py --video your_video.mp4
```
- Handles any resolution (1080p, 4K, vertical videos)
- Automatically resizes to VAE-compatible dimensions
- Supports different aspect ratios
- Options:
  - `--size 512` - Target resolution (256, 512, 768)
  - `--frames 16` - Number of frames to process
  - `--noise 0.3` - Noise level for latent space

#### 3. List Available Models
```bash
python experiments/process_iphone_video.py --model list-models
```
Shows advanced models you can run locally on M1 Max

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
- Cleaned experiments folder to contain focused demos
- Output folder contains only the essential outputs from the latest run

### iPhone Video Support
- **Resolution Handling**: Automatically resizes any resolution (1080p, 4K, etc.)
- **Aspect Ratios**: Handles both landscape and portrait videos
- **Best Practices**:
  - Use 512x512 for optimal quality/speed balance
  - Reduce frame count if memory issues occur
  - Use CPU mode for very large resolutions

### Advanced Models for M1 Max
1. **Current (SD VAE)**: Stable, good quality, no temporal consistency
2. **AnimateDiff**: Better motion handling, ~4-6GB memory
3. **VideoMAE**: Lightweight, good for understanding, ~4GB memory
4. **FILM**: Frame interpolation, works with any resolution
5. **CogVideo**: State-of-art but needs optimization for M1

### NEW: Advanced Video Models (Better Motion & More Frames)

#### Stable Video Diffusion (SVD) - RECOMMENDED
- **Best for**: Motion quality and temporal consistency
- **Frames**: 14-25 native, extendable to 100+
- **Memory**: ~12-16GB (fits well in 32GB)
- **Install**: Already included with diffusers
- **Usage**:
  ```bash
  # Basic usage
  python experiments/use_stable_video_diffusion.py --video input.mov
  
  # With more motion
  python experiments/use_stable_video_diffusion.py --video input.mov --motion 200
  
  # Extended frames (50+)
  python experiments/use_stable_video_diffusion.py --video input.mov --extended 50
  ```

#### Other Notable Models:
- **I2VGen-XL**: Up to 128 frames! Great for long videos
- **ModelScope**: 64 frames, good motion
- **ZeroScope V2**: Efficient, 64 frames
- **CogVideoX**: State-of-art but needs 20-30GB

#### Model Comparison Script:
```bash
# List all advanced models
python experiments/advanced_video_models.py --list
```

### AnimateDiff Integration (NEW)
- **Supported Resolutions**: 256x256, 512x512, 768x768
- **Supported Frame Counts**: 8, 16, 24, or 32 frames (optimal)
- **File Format Support**: Now handles .mov files from iPhone!
- **Temporal Consistency**: Uses AnimateDiff's motion-aware VAE

#### Using AnimateDiff VAE:
```bash
# Process with AnimateDiff VAE (motion-aware)
python experiments/process_iphone_video.py --video your_iphone.mov --model animatediff

# Or use dedicated AnimateDiff script
python experiments/use_animatediff.py --video your_video.mov

# Generate new videos with AnimateDiff
python experiments/use_animatediff.py --generate "A spaceship flying through space"
```

#### Installation:
```bash
pip install diffusers[torch] transformers accelerate
```

#### Key Features:
- **Real AnimateDiff Model**: Uses guoyww/animatediff-motion-adapter-v1-5-2
- **Motion-Aware VAE**: Better temporal consistency than SD-VAE
- **Generation Mode**: Can create new videos from text prompts
- **Processing Mode**: Can process existing videos with motion awareness
- **.mov Support**: Handles iPhone videos directly
- **M1 Optimized**: Works on MPS with fallback to CPU