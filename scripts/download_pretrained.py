#!/usr/bin/env python3
"""
Download actual pretrained video autoencoder models
"""
import torch
import requests
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Some actual video autoencoder models available
REAL_MODELS = {
    # VideoGPT VQ-VAE model
    'videogpt_vqvae': {
        'url': 'https://huggingface.co/spaces/akhaliq/VideoGPT/resolve/main/bair_gpt.pt',
        'info': 'VideoGPT VQ-VAE trained on BAIR robot dataset',
        'type': 'vqvae'
    },
    
    # TATS (Time-Agnostic VQGAN) - video tokenizer
    'tats_vqgan': {
        'url': 'https://huggingface.co/sayakpaul/tats-vqgan/resolve/main/pytorch_model.bin',
        'info': 'TATS VQ-GAN for video tokenization',
        'type': 'vqgan'
    },
    
    # You can also use image autoencoders frame-by-frame
    'vqgan_imagenet': {
        'url': 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1',
        'info': 'VQ-GAN trained on ImageNet (can be applied to video frames)',
        'type': 'vqgan'
    }
}

def download_model(model_name, save_dir='models/pretrained'):
    """Download a pretrained model"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if model_name not in REAL_MODELS:
        print(f"Available models: {list(REAL_MODELS.keys())}")
        return None
        
    model_info = REAL_MODELS[model_name]
    save_path = Path(save_dir) / f"{model_name}.pt"
    
    if save_path.exists():
        print(f"Model already downloaded: {save_path}")
        return save_path
        
    print(f"Downloading {model_name}: {model_info['info']}")
    print(f"From: {model_info['url']}")
    
    try:
        response = requests.get(model_info['url'], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='')
                    
        print(f"\nSaved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


if __name__ == '__main__':
    print("Available pretrained models:")
    for name, info in REAL_MODELS.items():
        print(f"- {name}: {info['info']}")
    
    # Example: download VideoGPT model
    # download_model('videogpt_vqvae')