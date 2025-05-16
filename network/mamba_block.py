import torch
import torch.nn as nn

class MambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim or dim * 4)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim or dim * 4, dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x
