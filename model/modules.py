import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.pixel_shuffle import *


class BottomUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(BottomUpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 4, 4), stride=(1, 2, 2),
                               padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.activation2 = nn.ReLU()
        # residual connection
        self.conv3 = nn.Conv3d(in_channels, out_channels // 4, kernel_size=1, stride=1)
        self.ps = PixelShuffleSC(scale)

    def forward(self, x):

        x1 = x
        # straight forward path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        # residual connection path
        x1 = self.conv3(x1)
        x1 = self.ps(x1)
        # print(x.shape, x1.shape)
        out = x + x1
        return out


class LateralBlock(nn.Module):
    def __init__(self, channels, scale=2):
        super(LateralBlock, self).__init__()
        self.channels = channels
        self.scale = scale

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 7, 7), stride=(2, 1, 1),
                               padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3, 7, 7), stride=(1, 1, 1),
                               padding=(1, 3, 3))
        self.bn2 = nn.BatchNorm3d(channels)
        self.activation2 = nn.ReLU()
        # residual connection
        self.conv3 = nn.Conv3d(channels, channels // 2, kernel_size=1, stride=1, padding=0)
        self.ps = PixelShuffleTC(scale)

    def forward(self, x):

        x1 = x
        # straight forward path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        # residual connection path
        x1 = self.conv3(x1)
        x1 = self.ps(x1)

        out = x + x1
        return out


class TopDownBlock(nn.Module):
    def __init__(self, channels, scale=2, spatial_dim='H'):
        super(TopDownBlock, self).__init__()
        self.channels = channels
        self.scale = scale
        self.spatial_dim = spatial_dim

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation2 = nn.ReLU()
        self.ps1 = PixelShuffleCS(scale, spatial_dim)
        # residual connection
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=1, stride=1)
        self.ps2 = PixelShuffleCS(scale, spatial_dim)

    def forward(self, x):
        x1 = x
        # straight forward path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.ps1(x)
        # residual connection path
        x1 = self.conv3(x1)
        x1 = self.ps2(x1)

        out = x + x1
        return out


class RBD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RBD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 原论文里RBD的直连通道上的Conv，要是没有padding的信息，默认为padding=0,那么两条通路产生的结果维度不相同是无法直接相加的
        # 因此这里实现的时候把直连通路上的conv1和conv2都加上padding=1(conv3只改变通道数)
        self.activation1 = nn.LeakyReLU()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # residual connection
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        # # spectral_norm: 对某一层的权重做标准化，输入参数是module（forward中怎么用）
        # self.norm1 = nn.utils.spectral_norm(self.conv1)
        # self.norm2 = nn.utils.spectral_norm(self.conv2)
        # self.norm3 = nn.utils.spectral_norm(self.conv3)
        # 具体用法参考：https://blog.csdn.net/qq_37950002/article/details/115592633，即模型实例化之后对所需的Layer
        # 使用nn.utils.spectral_norm

    def forward(self, x):
        """

        :param x: (N,C,T,H,W)
        :return:
        """
        x1 = x
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        # residual connection
        x1 = self.conv3(x1)

        out = x + x1
        return out


class RBD_2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RBD_2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation1 = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation2 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # residual
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        """
        :param x: (N,C,H,W)
        :return:
        """
        x1 = x
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.activation2(x)
        x = self.conv2(x)
        # residual
        x1 = self.conv3(x1)
        out = x + x1
        return out


# https://blog.csdn.net/qq_37950002/article/details/115592633
def add_spectral_norm(module):
    for name, layer in module.named_children():
        module.add_module(name, add_spectral_norm(layer))  # 递归调用
    if isinstance(module, nn.Conv3d):
        return nn.utils.spectral_norm(module)
    else:
        return module
