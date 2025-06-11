import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import FourierBlock1d, FourierBlock2d

class FNO1d(nn.Module):
    def __init__(self, modes, width, input_dim=2, output_dim=None, layers=4):
        super(FNO1d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.layers = layers

        self.lift = nn.Linear(input_dim, width)

        self.fourier = nn.Sequential(
            [FourierBlock1d(width, width, modes) for i in range(layers)]
        )
        self.proj = nn.Sequential(
            nn.Linear(width, width//2),
            nn.GELU(),
            nn.Linear(width//2, self.output_dim)
        )

    def forward(self, x):
        # x shape: (batch_size, grid_points, input_dim)
        x = self.lift(x)
        x = x.permute(0, 2, 1)
        x = self.fourier(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x
    
class FNO2d(nn.Module):
    def __init__(self, modes, width, input_dim=2, output_dim=None, layers=4):
        super(FNO2d, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.layers = layers

        self.lift = nn.Conv2d(input_dim, width, 1)

        self.fourier = nn.Sequential(
            [FourierBlock2d(width, width, modes) for i in range(layers)]
        )
        self.proj = nn.Sequential(
            nn.Conv2d(width, width//2, 1),
            nn.GELU(),
            nn.Conv2d(width//2, self.output_dim, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, input_dim, height, width)
        x = self.lift(x)
        x = self.fourier(x)
        x = self.proj(x)
        return x