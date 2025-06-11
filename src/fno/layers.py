import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels * out_channels))

        # Complex weights because we are applying in Fourier space
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes)/self.scale, dtype=torch.cfloat)

    def weight_multiply(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        # Needs to be independent of x size 
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        # x shape: (batch_size, in_channels, number of gridpoints)

        batchsize = x.shape[0]
        length = x.shape[-1]
        # Compute Fourier coefficients 
        # rfft returns only the non-redundant positive Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, length//2+1, 2, device=x.device)
        out_ft[:, :, :self.modes] = self.weight_multiply(x_ft[:, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=length)
        return x

class FourierConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / (in_channels * out_channels))

        # Complex weights because we are applying in Fourier space
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, 2)/self.scale, dtype=torch.cfloat)

    def weight_multiply(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        # Needs to be independent of x size 
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        # x shape: (batch_size, in_channels, x, y)

        batchsize = x.shape[0]
        xlen = x.shape[-2]
        ylen = x.shape[-1]
        # Compute Fourier coefficients 
        # rfft returns only the non-redundant positive Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, xlen, ylen//2+1, 2, device=x.device)
        out_ft[:, :, :, :self.modes] = self.weight_multiply(x_ft[:, :, :, :self.modes], self.weights)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(xlen, ylen), dim=[-2, -1])
        return x

class FourierBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierBlock1d, self).__init__()
        self.fconv = FourierConv1d(in_channels, out_channels, modes) # Fourier global transformation
        self.lin = nn.Conv1d(in_channels, out_channels, 1) # Linear local transformation
        self.activation = nn.GELU()
    def forward(self, x):
        x1 = self.fconv(x)
        x2 = self.lin(x)
        return self.activation(x1 + x2)

class FourierBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(FourierBlock2d, self).__init__()
        self.fconv = FourierConv2d(in_channels, out_channels, modes) # Fourier global transformation
        self.lin = nn.Conv2d(in_channels, out_channels, 1) # Linear local transformation
        self.activation = nn.GELU()

    def forward(self, x):
        x1 = self.fconv(x)
        x2 = self.lin(x)
        return self.activation(x1 + x2)
