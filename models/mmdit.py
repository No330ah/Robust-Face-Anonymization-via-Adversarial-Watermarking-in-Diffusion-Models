# models/mm_dit.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MMDiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, condition):
        """
        x: (B, N, D), condition: (B, M, D)
        """
        x_norm = self.norm1(x)
        cond_norm = self.norm1(condition)
        x_attn, _ = self.attn(x_norm, cond_norm, cond_norm)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

class SingleDiTBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class MM_DiT(nn.Module):
    def __init__(self, dim=512, num_heads=8, depth_mm=2, depth_single=2):
        super().__init__()
        self.mm_blocks = nn.ModuleList([
            MMDiTBlock(dim, num_heads) for _ in range(depth_mm)
        ])
        self.single_blocks = nn.ModuleList([
            SingleDiTBlock(dim, num_heads) for _ in range(depth_single)
        ])

    def forward(self, z, id_embed, image_embed):
        """
        z: (B, N, D)
        id_embed: (B, 1, D), image_embed: (B, 1, D)
        """
        condition = torch.cat([id_embed, image_embed], dim=1)  # (B, 2, D)
        for blk in self.mm_blocks:
            z = blk(z, condition)
        for blk in self.single_blocks:
            z = blk(z)
        return z

# Example usage
if __name__ == '__main__':
    model = MM_DiT(dim=512)
    z = torch.randn(2, 256, 512)  # e.g. 16x16 patch latent
    id_embed = torch.randn(2, 1, 512)
    img_embed = torch.randn(2, 1, 512)
    out = model(z, id_embed, img_embed)
    print(out.shape)
