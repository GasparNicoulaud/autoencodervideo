import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple


class LatentAnalyzer:
    """
    Analyze and visualize latent space representations
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.latent_codes = []
        self.labels = []
        
    def collect_latents(self, dataloader, max_samples: Optional[int] = None):
        """Collect latent codes from a dataloader"""
        self.model.eval()
        collected = 0
        
        with torch.no_grad():
            for batch_idx, (videos, labels) in enumerate(dataloader):
                if max_samples and collected >= max_samples:
                    break
                    
                if hasattr(self.model, 'encode'):
                    z, _, _ = self.model.encode(videos)
                else:
                    _, _, _, z = self.model(videos)
                    
                self.latent_codes.append(z.cpu())
                self.labels.extend(labels.cpu().numpy())
                
                collected += z.shape[0]
                
        self.latent_codes = torch.cat(self.latent_codes, dim=0)
        self.labels = np.array(self.labels)
        
    def compute_pca(self, n_components: int = 50) -> Dict:
        """Compute PCA on latent codes"""
        latents_np = self.latent_codes.numpy()
        
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(latents_np)
        
        return {
            'pca': pca,
            'transformed': transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'components': pca.components_
        }
        
    def compute_tsne(self, n_components: int = 2, perplexity: float = 30.0) -> np.ndarray:
        """Compute t-SNE on latent codes"""
        latents_np = self.latent_codes.numpy()
        
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        transformed = tsne.fit_transform(latents_np)
        
        return transformed
        
    def find_nearest_neighbors(self, query_latent: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find k nearest neighbors in latent space"""
        distances = torch.cdist(query_latent.unsqueeze(0), self.latent_codes)
        values, indices = torch.topk(distances, k, largest=False)
        
        return indices.squeeze(), values.squeeze()
        
    def interpolation_quality(self, z1: torch.Tensor, z2: torch.Tensor, 
                            steps: int = 10) -> Dict[str, float]:
        """Measure quality of interpolation between two latents"""
        from .interpolation import interpolate_latents
        
        linear_path = interpolate_latents(z1, z2, steps, method='linear')
        spherical_path = interpolate_latents(z1, z2, steps, method='spherical')
        
        linear_distances = []
        spherical_distances = []
        
        for i in range(1, steps):
            linear_distances.append(
                torch.norm(linear_path[i] - linear_path[i-1]).item()
            )
            spherical_distances.append(
                torch.norm(spherical_path[i] - spherical_path[i-1]).item()
            )
            
        return {
            'linear_variance': np.var(linear_distances),
            'spherical_variance': np.var(spherical_distances),
            'linear_mean_step': np.mean(linear_distances),
            'spherical_mean_step': np.mean(spherical_distances)
        }
        
    def visualize_latent_space(self, method: str = 'pca', save_path: Optional[str] = None):
        """Visualize the latent space using PCA or t-SNE"""
        plt.figure(figsize=(10, 8))
        
        if method == 'pca':
            pca_result = self.compute_pca(n_components=2)
            coords = pca_result['transformed'][:, :2]
            xlabel = f'PC1 ({pca_result["explained_variance_ratio"][0]:.2%})'
            ylabel = f'PC2 ({pca_result["explained_variance_ratio"][1]:.2%})'
        elif method == 'tsne':
            coords = self.compute_tsne(n_components=2)
            xlabel = 't-SNE 1'
            ylabel = 't-SNE 2'
        else:
            raise ValueError(f"Unknown method: {method}")
            
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=self.labels, cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Latent Space Visualization ({method.upper()})')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_dimensions(self, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """Analyze the importance and variation of each latent dimension"""
        if len(self.latent_codes) < num_samples:
            samples = self.latent_codes
        else:
            indices = torch.randperm(len(self.latent_codes))[:num_samples]
            samples = self.latent_codes[indices]
            
        mean = samples.mean(dim=0).numpy()
        std = samples.std(dim=0).numpy()
        
        correlations = np.corrcoef(samples.T)
        
        return {
            'mean': mean,
            'std': std,
            'correlations': correlations,
            'active_dims': np.where(std > 0.1)[0]
        }
        
    def generate_latent_traversal(self, base_latent: torch.Tensor,
                                 dimension: int,
                                 range_vals: Tuple[float, float] = (-3, 3),
                                 steps: int = 10) -> torch.Tensor:
        """Generate a traversal along a single latent dimension"""
        values = torch.linspace(range_vals[0], range_vals[1], steps)
        traversal = base_latent.unsqueeze(0).repeat(steps, 1)
        traversal[:, dimension] = values
        
        return traversal