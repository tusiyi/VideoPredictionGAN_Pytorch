import torch
import torch.nn as nn


class PixelShuffleSC(nn.Module):
    def __init__(self, scale):
        super(PixelShuffleSC, self).__init__()
        self.scale = scale

    def forward(self, x):
        """
        Upscale an input array along its channel dimension from interspersed spatial layer by subpixel array sampling
         with the up-scaling factor scale
        :param x: shape-(N, C, D, H, W)
        :return: shape-(N, out_C, D, out_H, out_W)
        """
        N, C, D, H, W = x.shape
        out_C = C * self.scale ** 2
        out_H = H // self.scale
        out_W = W // self.scale
        return x.reshape(N, out_C, D, out_H, out_W)


class PixelShuffleTC(nn.Module):
    def __init__(self, scale):
        super(PixelShuffleTC, self).__init__()
        self.scale = scale

    def forward(self, x):
        """
        Upscale an input array along its channel dimension from interspersed time layer by subpixel array sampling
         with the up-scaling factor scale
        :param x: shape-(N, C, D, H, W)
        :return: shape-(N, out_C, out_D, H, W)
        """
        N, C, D, H, W = x.shape
        out_C = C * self.scale
        out_D = D // self.scale
        return x.reshape(N, out_C, out_D, H, W)


class PixelShuffleCS(nn.Module):
    def __init__(self, scale, spatial_dim):
        super(PixelShuffleCS, self).__init__()
        self.scale = scale
        self.spatial_dim = spatial_dim
        # self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        """
        Upscale an input array along its spatial dimension from interspersed channel layer by subpixel array sampling
         with the up-scaling factor scale
        :param x: shape-(N, C, H, W)  # D==1
        :return: shape-(N, C1, H1, W1)
        """
        N, C, H, W = x.shape  # dimension before pixel shuffling

        if self.spatial_dim == 'H':
            out_C = C // self.scale
            out_H = H * self.scale
            x = x.reshape(N, out_C, out_H, W)
        else:
            # 'W'
            out_C = C // self.scale
            out_W = W * self.scale
            x = x.reshape(N, out_C, H, out_W)

        return x
