# Video Autoencoder with Programmable Latent Space Control

A PyTorch implementation of video autoencoders (VAE and VQ-VAE) with advanced programmatic control over latent space weights and biases.

## Features

- **Video Autoencoder Models**
  - Variational Autoencoder (VAE) with KL divergence
  - Vector Quantized VAE (VQ-VAE) with discrete latent codes
  - Pre-trained model support (placeholder for actual models)

- **Latent Space Manipulation**
  - Programmatic weight and bias control
  - Multiple interpolation methods (linear, spherical, circular)
  - Weight transformations (scaling, noise, quantization, pruning)
  - Custom activation patterns and frequency filtering

- **Analysis Tools**
  - PCA and t-SNE visualization
  - Latent dimension analysis
  - Interpolation quality metrics
  - Nearest neighbor search in latent space

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd autoencodervideo

# Run the setup script
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage

```python
from src.models import VideoAutoencoder
from src.latent import LatentManipulator
from src.utils import load_video, save_video

# Initialize model
model = VideoAutoencoder(latent_dim=512)
model.eval()

# Load and encode video
video = load_video("path/to/video.mp4", num_frames=32, size=(128, 128))
z, mu, log_var = model.encode(video.unsqueeze(0))

# Decode back
reconstructed = model.decode(z)
save_video(reconstructed, "output/reconstructed.mp4")
```

### 2. Latent Space Manipulation

```python
# Create manipulator
manipulator = LatentManipulator(model)
manipulator.save_original_params()

# Scale decoder weights
manipulator.scale_weights('decoder_fc', scale_factor=1.5)

# Add noise to encoder
manipulator.add_noise('encoder_fc', noise_std=0.1)

# Quantize weights
manipulator.quantize_weights('decoder_fc', num_levels=16)

# Restore original
manipulator.restore_original_params()
```

### 3. Interpolation

```python
from src.latent import interpolate_latents

# Linear interpolation
interp = interpolate_latents(z1, z2, steps=20, method='linear')

# Spherical interpolation
interp = interpolate_latents(z1, z2, steps=20, method='spherical')
```

## Example Scripts

### Latent Manipulation Demo
```bash
python experiments/latent_manipulation_demo.py \
    --video path/to/video.mp4 \
    --model vae_ucf101 \
    --output-dir output/manipulations
```

### Interpolation Demo
```bash
python experiments/interpolation_demo.py \
    --video1 path/to/video1.mp4 \
    --video2 path/to/video2.mp4 \
    --steps 20 \
    --output-dir output/interpolations
```

### Programmatic Control
```bash
python experiments/programmatic_control.py \
    --output-dir output/programmatic
```

## Advanced Programmatic Control

### Custom Weight Patterns

```python
controller = ProgrammaticLatentController(model)

# Apply gradient pattern
controller.create_custom_weight_pattern('decoder_fc', 'gradient')

# Apply frequency filter
controller.apply_frequency_filter('decoder_fc', cutoff_freq=0.1, filter_type='low')

# Create activation patterns
z_modulated = controller.create_activation_patterns(z, pattern='sine')
```

### Custom Transformations

```python
def custom_transform(z):
    z_new = z.clone()
    z_new[:, :100] = torch.tanh(z[:, :100] * 2)
    z_new[:, 100:200] = torch.sigmoid(z[:, 100:200])
    return z_new

z_transformed = custom_transform(z)
video = model.decode(z_transformed)
```

## Model Architecture

### VideoAutoencoder
- Encoder: 3D CNN with residual blocks
- Latent space: Continuous with reparameterization trick
- Decoder: 3D transposed CNN
- Loss: Reconstruction (MSE) + KL divergence

### VQ-VAE
- Encoder: 3D CNN
- Vector quantization layer with codebook
- Decoder: 3D transposed CNN
- Loss: Reconstruction + VQ loss + commitment loss

## Jupyter Notebook

See `notebooks/autoencoder_tutorial.ipynb` for an interactive tutorial with:
- Model initialization
- Latent space operations
- Interactive dimension exploration
- Batch processing examples

## Project Structure

```
autoencodervideo/
├── src/
│   ├── models/           # Model architectures
│   ├── latent/           # Latent space manipulation tools
│   └── utils/            # Video I/O and training utilities
├── experiments/          # Example scripts
├── notebooks/            # Jupyter tutorials
├── configs/              # Configuration files
└── scripts/              # Utility scripts
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)
- See `requirements.txt` for full list

## Citation

If you use this code in your research, please cite:
```bibtex
@software{video_autoencoder_2024,
  title = {Video Autoencoder with Programmable Latent Space Control},
  year = {2024},
  author = {Your Name}
}
```

## License

MIT License