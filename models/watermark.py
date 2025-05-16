# models/watermark.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkEmbedder(nn.Module):
    def __init__(self, embed_dim=512, alpha=0.3):
        """
        embed_dim: watermark dimension
        alpha: （from0-1，more near 0 more visible）
        """
        super(WatermarkEmbedder, self).__init__()
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, z, w):
        """
        z: latent vec， (B, C, H, W) or (B, L, D)
        w: watermark vec， (B, D)
        """
        encoded_w = self.encoder(w).unsqueeze(-1).unsqueeze(-1)
        if z.dim() == 3:
            return z + self.alpha * self.encoder(w).unsqueeze(1)
        elif z.dim() == 4:
            return z + self.alpha * encoded_w.expand_as(z)
        else:
            raise ValueError("Unsupported latent shape")

class WatermarkExtractor(nn.Module):
    def __init__(self, embed_dim=512):
        super(WatermarkExtractor, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, z):
        """
        get vec from z
        """
        if z.dim() == 4:
            z = torch.mean(z, dim=[2, 3])  # (B, D)
        elif z.dim() == 3:
            z = torch.mean(z, dim=1)       # (B, D)
        return self.decoder(z)

def watermark_loss(w_orig, w_extracted, threshold=0.1):
    """
    calculate L1 
    """
    return F.l1_loss(w_extracted, w_orig)
