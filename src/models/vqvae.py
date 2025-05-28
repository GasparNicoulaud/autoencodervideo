import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        z = rearrange(z, 'b c h w d -> b h w d c')
        z_flattened = z.reshape(-1, self.embedding_dim)
        
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z + (quantized - z).detach()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        quantized = rearrange(quantized, 'b h w d c -> b c h w d')
        
        return quantized, loss, perplexity, encodings, encoding_indices.view(z.shape[:-1])


class VQVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_embeddings=512, base_channels=64):
        super().__init__()
        from .video_autoencoder import VideoEncoder, VideoDecoder
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 7, 2, 3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 4, latent_dim, 4, 2, 1),
        )
        
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, base_channels * 4, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 4, 2, 1),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels * 2, base_channels, 4, 2, 1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(base_channels, in_channels, 7, 2, 3, output_padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss, perplexity, _, indices = self.vq(z)
        recon = self.decoder(quantized)
        
        return recon, vq_loss, perplexity, indices
        
    def encode(self, x):
        z = self.encoder(x)
        quantized, _, _, _, indices = self.vq(z)
        return quantized, indices
        
    def decode(self, quantized):
        return self.decoder(quantized)
        
    def decode_from_indices(self, indices):
        quantized = self.vq.embedding(indices)
        quantized = rearrange(quantized, 'b h w d c -> b c h w d')
        return self.decode(quantized)
        
    def get_codebook(self):
        return self.vq.embedding.weight.data.clone()
        
    def set_codebook(self, new_codebook):
        with torch.no_grad():
            self.vq.embedding.weight.data = new_codebook