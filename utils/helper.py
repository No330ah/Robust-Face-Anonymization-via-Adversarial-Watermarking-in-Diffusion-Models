import torch
import math

def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    grid_h = torch.arange(grid_size_h, dtype=torch.float32)
    grid_w = torch.arange(grid_size_w, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape(2, 1, grid_size_h, grid_size_w)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)

def get_1d_sincos_pos_embed(embed_dim, pos):
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / (10000 ** omega)
    pos = pos.reshape(-1)
    out = torch.einsum('p,d->pd', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb

def flatten_patch(x, ph=2, pw=2):
    # x: (B, C, H, W) -> (B, N, patch_dim)
    B, C, H, W = x.shape
    x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H//ph, W//pw, ph, pw)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H//ph, W//pw, C, ph, pw)
    x = x.view(B, -1, C * ph * pw)
    return x

def unflatten_patch(x, ph=2, pw=2, h=16, w=16):
    # x: (B, N, patch_dim) -> (B, C, H, W)
    B, N, D = x.shape
    C = D // (ph * pw)
    x = x.view(B, h, w, C, ph, pw)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, C, h * ph, w * pw)
    return x

def visualize_mask(tensor_mask, save_path="mask_vis.png"):
    from torchvision.utils import save_image
    if tensor_mask.dim() == 4:
        tensor_mask = tensor_mask[0]
    save_image(tensor_mask, save_path)

def apply_rope(x, rope_cache):
    # x: (B, N, D)
    # rope_cache: (1, N, D)
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = rope_cache[..., ::2], rope_cache[..., 1::2]
    x_rope = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return x_rope