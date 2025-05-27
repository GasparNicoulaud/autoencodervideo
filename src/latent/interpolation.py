import torch
import numpy as np
from typing import List, Optional, Union


def interpolate_latents(z1: torch.Tensor, z2: torch.Tensor, 
                       steps: int = 10, method: str = 'linear') -> torch.Tensor:
    """
    Interpolate between two latent vectors
    
    Args:
        z1: Starting latent vector
        z2: Ending latent vector
        steps: Number of interpolation steps
        method: Interpolation method ('linear' or 'spherical')
        
    Returns:
        Tensor of interpolated latent vectors
    """
    if method == 'linear':
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolated = []
        
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            interpolated.append(z_interp)
            
        return torch.stack(interpolated)
        
    elif method == 'spherical':
        return slerp(z1, z2, steps)
        
    else:
        raise ValueError(f"Unknown interpolation method: {method}")


def slerp(z1: torch.Tensor, z2: torch.Tensor, steps: int = 10) -> torch.Tensor:
    """
    Spherical linear interpolation between latent vectors
    """
    z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
    
    omega = torch.acos(torch.clamp(torch.sum(z1_norm * z2_norm, dim=-1), -1, 1))
    
    alphas = torch.linspace(0, 1, steps, device=z1.device)
    interpolated = []
    
    for alpha in alphas:
        z_interp = (torch.sin((1 - alpha) * omega) / torch.sin(omega)).unsqueeze(-1) * z1 + \
                   (torch.sin(alpha * omega) / torch.sin(omega)).unsqueeze(-1) * z2
        interpolated.append(z_interp)
        
    return torch.stack(interpolated)


def interpolate_along_path(latents: List[torch.Tensor], 
                          steps_per_segment: int = 10,
                          method: str = 'linear',
                          loop: bool = False) -> torch.Tensor:
    """
    Interpolate along a path defined by multiple latent vectors
    
    Args:
        latents: List of latent vectors defining the path
        steps_per_segment: Number of steps between each pair of latents
        method: Interpolation method
        loop: Whether to create a loop by connecting last to first
        
    Returns:
        Tensor of interpolated latent vectors
    """
    interpolated_segments = []
    
    for i in range(len(latents) - 1):
        segment = interpolate_latents(
            latents[i], latents[i + 1], 
            steps_per_segment, method
        )
        interpolated_segments.append(segment[:-1])
        
    interpolated_segments.append(latents[-1].unsqueeze(0))
    
    if loop and len(latents) > 2:
        loop_segment = interpolate_latents(
            latents[-1], latents[0],
            steps_per_segment, method
        )
        interpolated_segments.append(loop_segment[1:])
        
    return torch.cat(interpolated_segments, dim=0)


def radial_interpolation(center: torch.Tensor,
                        radius: float,
                        num_points: int,
                        dimensions: Optional[List[int]] = None) -> torch.Tensor:
    """
    Create a circular interpolation around a center point in latent space
    
    Args:
        center: Center point in latent space
        radius: Radius of the circle
        num_points: Number of points on the circle
        dimensions: Which dimensions to interpolate (default: first 2)
        
    Returns:
        Tensor of latent vectors forming a circle
    """
    if dimensions is None:
        dimensions = [0, 1]
        
    angles = torch.linspace(0, 2 * np.pi, num_points, device=center.device)
    interpolated = center.unsqueeze(0).repeat(num_points, 1)
    
    interpolated[:, dimensions[0]] += radius * torch.cos(angles)
    interpolated[:, dimensions[1]] += radius * torch.sin(angles)
    
    return interpolated


def grid_interpolation(corners: List[torch.Tensor],
                      grid_size: tuple = (10, 10)) -> torch.Tensor:
    """
    Create a 2D grid interpolation between 4 corner points
    
    Args:
        corners: List of 4 latent vectors [top_left, top_right, bottom_left, bottom_right]
        grid_size: (height, width) of the grid
        
    Returns:
        Tensor of interpolated latent vectors in grid shape
    """
    if len(corners) != 4:
        raise ValueError("Need exactly 4 corner points")
        
    h, w = grid_size
    grid = torch.zeros(h, w, corners[0].shape[-1], device=corners[0].device)
    
    for i in range(h):
        for j in range(w):
            alpha_h = i / (h - 1)
            alpha_w = j / (w - 1)
            
            top = (1 - alpha_w) * corners[0] + alpha_w * corners[1]
            bottom = (1 - alpha_w) * corners[2] + alpha_w * corners[3]
            
            grid[i, j] = (1 - alpha_h) * top + alpha_h * bottom
            
    return grid.reshape(-1, corners[0].shape[-1])