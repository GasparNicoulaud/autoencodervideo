import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional, Callable

try:
    import wandb
except ImportError:
    wandb = None


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int,
                log_interval: int = 10,
                use_wandb: bool = False) -> Dict[str, float]:
    """
    Train for one epoch
    
    Returns:
        Dictionary of average metrics
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, (videos, _) in enumerate(pbar):
        videos = videos.to(device)
        
        optimizer.zero_grad()
        
        if hasattr(model, 'loss_function'):
            recon, mu, log_var, z = model(videos)
            losses = model.loss_function(recon, videos, mu, log_var)
            loss = losses['loss']
            
            total_recon_loss += losses['recon_loss'].item()
            total_kld_loss += losses['kld_loss'].item()
        else:
            recon, vq_loss, perplexity, _ = model(videos)
            recon_loss = nn.functional.mse_loss(recon, videos)
            loss = recon_loss + vq_loss
            
            total_recon_loss += recon_loss.item()
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': losses.get('recon_loss', recon_loss).item() if 'losses' in locals() else recon_loss.item()
            })
            
            if use_wandb and wandb is not None:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/recon_loss': losses.get('recon_loss', recon_loss).item() if 'losses' in locals() else recon_loss.item(),
                    'train/step': epoch * len(dataloader) + batch_idx
                })
                
    n_batches = len(dataloader)
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kld_loss': total_kld_loss / n_batches if total_kld_loss > 0 else 0
    }
    
    return metrics


def validate_epoch(model: nn.Module,
                  dataloader: DataLoader,
                  device: torch.device,
                  epoch: int,
                  use_wandb: bool = False) -> Dict[str, float]:
    """
    Validate for one epoch
    
    Returns:
        Dictionary of average metrics
    """
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kld_loss = 0
    
    with torch.no_grad():
        for videos, _ in tqdm(dataloader, desc=f'Val {epoch}'):
            videos = videos.to(device)
            
            if hasattr(model, 'loss_function'):
                recon, mu, log_var, z = model(videos)
                losses = model.loss_function(recon, videos, mu, log_var)
                loss = losses['loss']
                
                total_recon_loss += losses['recon_loss'].item()
                total_kld_loss += losses['kld_loss'].item()
            else:
                recon, vq_loss, perplexity, _ = model(videos)
                recon_loss = nn.functional.mse_loss(recon, videos)
                loss = recon_loss + vq_loss
                
                total_recon_loss += recon_loss.item()
                
            total_loss += loss.item()
            
    n_batches = len(dataloader)
    metrics = {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'kld_loss': total_kld_loss / n_batches if total_kld_loss > 0 else 0
    }
    
    if use_wandb and wandb is not None:
        wandb.log({
            'val/loss': metrics['loss'],
            'val/recon_loss': metrics['recon_loss'],
            'val/kld_loss': metrics['kld_loss'],
            'val/epoch': epoch
        })
        
    return metrics