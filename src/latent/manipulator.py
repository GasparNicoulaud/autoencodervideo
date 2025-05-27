import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Dict, List, Tuple


class LatentManipulator:
    """
    Class for programmatically manipulating latent space weights and biases
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.original_params = None
        self.manipulation_history = []
        
    def save_original_params(self):
        """Save the original parameters before manipulation"""
        self.original_params = self.model.get_latent_params()
        
    def restore_original_params(self):
        """Restore the original parameters"""
        if self.original_params is not None:
            self.model.set_latent_params(self.original_params)
            
    def manipulate_weights(self, 
                          layer_name: str,
                          weight_fn: Callable[[torch.Tensor], torch.Tensor],
                          param_type: str = 'weight'):
        """
        Apply a function to manipulate weights
        
        Args:
            layer_name: Name of the layer to manipulate
            weight_fn: Function to apply to the weights
            param_type: 'weight' or 'bias'
        """
        params = self.model.get_latent_params()
        
        if layer_name in params:
            layer_params = params[layer_name]
            if param_type in layer_params:
                original = layer_params[param_type].clone()
                layer_params[param_type] = weight_fn(layer_params[param_type])
                
                self.manipulation_history.append({
                    'layer': layer_name,
                    'param_type': param_type,
                    'original': original,
                    'new': layer_params[param_type].clone()
                })
                
        self.model.set_latent_params(params)
        
    def scale_weights(self, layer_name: str, scale_factor: float, param_type: str = 'weight'):
        """Scale weights by a factor"""
        self.manipulate_weights(
            layer_name,
            lambda w: w * scale_factor,
            param_type
        )
        
    def add_noise(self, layer_name: str, noise_std: float, param_type: str = 'weight'):
        """Add Gaussian noise to weights"""
        def add_noise_fn(w):
            noise = torch.randn_like(w) * noise_std
            return w + noise
            
        self.manipulate_weights(layer_name, add_noise_fn, param_type)
        
    def prune_weights(self, layer_name: str, threshold: float, param_type: str = 'weight'):
        """Prune weights below a threshold"""
        def prune_fn(w):
            mask = torch.abs(w) > threshold
            return w * mask.float()
            
        self.manipulate_weights(layer_name, prune_fn, param_type)
        
    def quantize_weights(self, layer_name: str, num_levels: int, param_type: str = 'weight'):
        """Quantize weights to a fixed number of levels"""
        def quantize_fn(w):
            w_min, w_max = w.min(), w.max()
            w_normalized = (w - w_min) / (w_max - w_min)
            w_quantized = torch.round(w_normalized * (num_levels - 1)) / (num_levels - 1)
            return w_quantized * (w_max - w_min) + w_min
            
        self.manipulate_weights(layer_name, quantize_fn, param_type)
        
    def apply_mask(self, layer_name: str, mask: torch.Tensor, param_type: str = 'weight'):
        """Apply a binary mask to weights"""
        def mask_fn(w):
            return w * mask
            
        self.manipulate_weights(layer_name, mask_fn, param_type)
        
    def interpolate_with_model(self, other_model: nn.Module, alpha: float):
        """Interpolate latent parameters with another model"""
        params1 = self.model.get_latent_params()
        params2 = other_model.get_latent_params()
        
        interpolated_params = {}
        for key in params1:
            if key in params2:
                interpolated_params[key] = {}
                for param_name in params1[key]:
                    if param_name in params2[key]:
                        interpolated_params[key][param_name] = (
                            (1 - alpha) * params1[key][param_name] + 
                            alpha * params2[key][param_name]
                        )
                        
        self.model.set_latent_params(interpolated_params)
        
    def apply_pca_direction(self, layer_name: str, direction: torch.Tensor, 
                           magnitude: float, param_type: str = 'weight'):
        """Apply movement in a PCA direction"""
        def pca_fn(w):
            w_flat = w.view(w.size(0), -1)
            w_flat = w_flat + magnitude * direction.unsqueeze(0)
            return w_flat.view_as(w)
            
        self.manipulate_weights(layer_name, pca_fn, param_type)
        
    def get_manipulation_history(self) -> List[Dict]:
        """Get the history of manipulations"""
        return self.manipulation_history
        
    def clear_history(self):
        """Clear manipulation history"""
        self.manipulation_history = []
        
    def compute_weight_statistics(self) -> Dict:
        """Compute statistics about the current weights"""
        params = self.model.get_latent_params()
        stats = {}
        
        for layer_name, layer_params in params.items():
            stats[layer_name] = {}
            for param_type, param_tensor in layer_params.items():
                stats[layer_name][param_type] = {
                    'mean': param_tensor.mean().item(),
                    'std': param_tensor.std().item(),
                    'min': param_tensor.min().item(),
                    'max': param_tensor.max().item(),
                    'sparsity': (param_tensor == 0).float().mean().item(),
                    'shape': list(param_tensor.shape)
                }
                
        return stats