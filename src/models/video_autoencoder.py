import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Dict


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm3d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class VideoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.initial = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 7, 2, 3),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, latent_dim * 2)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        mu, log_var = torch.chunk(x, 2, dim=1)
        return mu, log_var


class VideoDecoder(nn.Module):
    def __init__(self, latent_dim=512, out_channels=3, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4 * 4)
        
        self.layer1 = self._make_layer(base_channels * 8, base_channels * 4, 2)
        self.layer2 = self._make_layer(base_channels * 4, base_channels * 2, 2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels, 2)
        
        self.final = nn.Sequential(
            nn.ConvTranspose3d(base_channels, out_channels, 7, 2, 3, output_padding=1),
            nn.Tanh()
        )
        
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_channels * 8, 4, 4, 4)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final(x)
        
        return x


class VideoAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, base_channels=64, beta=1.0):
        super().__init__()
        self.encoder = VideoEncoder(in_channels, latent_dim, base_channels)
        self.decoder = VideoDecoder(latent_dim, in_channels, base_channels)
        self.beta = beta
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var, z
        
    def encode(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
        
    def decode(self, z):
        return self.decoder(z)
        
    def sample(self, batch_size, device):
        z = torch.randn(batch_size, self.latent_dim).to(device)
        return self.decode(z)
        
    def loss_function(self, recon, target, mu, log_var):
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        total_loss = recon_loss + self.beta * kld_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kld_loss': kld_loss
        }
        
    def get_latent_params(self):
        """Get all parameters related to latent space for manipulation"""
        params = {
            'encoder_fc': self.encoder.fc.state_dict(),
            'decoder_fc': self.decoder.fc.state_dict(),
        }
        return params
        
    def set_latent_params(self, params):
        """Set latent space parameters programmatically"""
        if 'encoder_fc' in params:
            self.encoder.fc.load_state_dict(params['encoder_fc'])
        if 'decoder_fc' in params:
            self.decoder.fc.load_state_dict(params['decoder_fc'])