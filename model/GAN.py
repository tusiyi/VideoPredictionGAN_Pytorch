import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

from model.modules import RBD, TopDownBlock, LateralBlock, BottomUpBlock, add_spectral_norm


class Discriminator(nn.Module):
    def __init__(self, num_blocks=5, num_rbds=[1, 2, 2, 2, 2], num_channels=[64, 128, 512, 1024, 2048],
                 img_channel=3, seq_len=9, img_size=(128, 160), avg_size=16):
        super(Discriminator, self).__init__()
        self.num_blocks = num_blocks
        self.num_rbds = num_rbds
        self.num_channels = num_channels
        self.img_channel = img_channel
        self.seq_len = seq_len
        self.h, self.w = img_size[0], img_size[1]
        self.avg_size = avg_size  # average pooling kernel size spatially

        self.model_list = []
        for i in range(num_blocks):
            in_chan = img_channel if i == 0 else num_channels[i - 1]
            out_chan = num_channels[i]
            for j in range(num_rbds[i]):
                in_c = in_chan if j == 0 else out_chan  # block中第一个RBD的输入通道为in_chan，其他为out_chan
                out_c = out_chan
                rbd = RBD(in_channels=in_c, out_channels=out_c)
                self.model_list.append(add_spectral_norm(rbd))

        # 论文里对discriminator的描述：The last spatial average pooling
        # is followed by temporal average pooling, and the discriminator finally tries to
        # classify if each N × M patch in a frame is real or predicted
        # 最后为什么是N × M patch in a frame？？？
        # 找文献来源：https://arxiv.org/pdf/1611.07004.pdf里的PatchGAN

        # spatial average pooling & temporal average pooling
        self.model_list.append(nn.AvgPool3d(kernel_size=(seq_len, self.avg_size, self.avg_size),
                                            stride=(1, self.avg_size, self.avg_size)))  # output :(N, 1, C=2048, 8, 10)
        self.model = nn.Sequential(*self.model_list)  # output :(N, 1, C=2048, 8, 10)
        # 需要算一个logits(batch中每个样本对应一个)
        self.conv = nn.Conv2d(num_channels[-1], 1, kernel_size=1, stride=1)  # 1x1卷积，将通道数变为1

    def forward(self, x):
        """

        :param x: shape—(N,T,C,H,W)
        :return:
        """
        N, T, C, H, W = x.shape
        x = x.reshape(N, C, T, H, W)  # 3d Conv requires
        x = self.model(x)  # output shape :(N, C=2048, 1, 8, 10)
        output = self.conv(x.squeeze(dim=2))  # # shape :(N, C=1, 8, 10)
        return output.squeeze()  # shape :(N, 8, 10)


class Generator(nn.Module):
    def __init__(self, seq_len=8, img_channel=3, img_size=(128, 160), num_channels=[64, 128, 256, 512]):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        self.img_channel = img_channel
        self.img_size = img_size
        self.num_channels = num_channels
        # module 1
        self.bu1 = BottomUpBlock(in_channels=img_channel, out_channels=num_channels[0], scale=2)
        self.lat1_1 = LateralBlock(channels=num_channels[0], scale=2)
        self.lat1_2 = LateralBlock(channels=num_channels[0], scale=2)
        self.lat1_3 = LateralBlock(channels=num_channels[0], scale=2)
        self.td1_1 = TopDownBlock(channels=num_channels[0] * 2, scale=2, spatial_dim='H')
        self.td1_2 = TopDownBlock(channels=num_channels[0], scale=2, spatial_dim='W')
        # module 2
        self.bu2 = BottomUpBlock(in_channels=num_channels[0], out_channels=num_channels[1], scale=2)
        self.lat2_1 = LateralBlock(channels=num_channels[1], scale=2)
        self.lat2_2 = LateralBlock(channels=num_channels[1], scale=2)
        self.lat2_3 = LateralBlock(channels=num_channels[1], scale=2)
        self.td2_1 = TopDownBlock(channels=num_channels[1] * 2, scale=2, spatial_dim='H')
        self.td2_2 = TopDownBlock(channels=num_channels[1], scale=2, spatial_dim='W')
        # module 3
        self.bu3 = BottomUpBlock(in_channels=num_channels[1], out_channels=num_channels[2], scale=2)
        self.lat3_1 = LateralBlock(channels=num_channels[2], scale=2)
        self.lat3_2 = LateralBlock(channels=num_channels[2], scale=2)
        self.lat3_3 = LateralBlock(channels=num_channels[2], scale=2)
        self.td3_1 = TopDownBlock(channels=num_channels[2] * 2, scale=2, spatial_dim='H')
        self.td3_2 = TopDownBlock(channels=num_channels[2], scale=2, spatial_dim='W')
        # module 4
        self.bu4 = BottomUpBlock(in_channels=num_channels[2], out_channels=num_channels[3], scale=2)
        self.lat4_1 = LateralBlock(channels=num_channels[3], scale=2)
        self.lat4_2 = LateralBlock(channels=num_channels[3], scale=2)
        self.lat4_3 = LateralBlock(channels=num_channels[3], scale=2)
        self.td4_1 = TopDownBlock(channels=num_channels[3] * 2, scale=2, spatial_dim='H')
        self.td4_2 = TopDownBlock(channels=num_channels[3], scale=2, spatial_dim='W')
        # last conv layer
        self.conv = nn.Conv2d(in_channels=num_channels[0] // 2, out_channels=img_channel,
                              kernel_size=3, stride=1, padding=1)
        self.out = nn.Sigmoid()

    def forward(self, x, d5):
        """
        :param x: frame sequence, (N, T, C, H, W)
        :param d5: random noise, (N, C4, H//16, W//16)
        :return:
        """
        N, T, C, H, W = x.shape
        x = x.reshape(N, C, T, H, W)  # 3d Conv requires
        # Bottom up
        u1 = self.bu1(x)
        u2 = self.bu2(u1)
        u3 = self.bu3(u2)
        u4 = self.bu4(u3)
        # Lateral, T=1
        lat1 = self.lat1_3(self.lat1_2(self.lat1_1(u1)))
        lat2 = self.lat2_3(self.lat2_2(self.lat2_1(u2)))
        lat3 = self.lat3_3(self.lat3_2(self.lat3_1(u3)))
        lat4 = self.lat4_3(self.lat4_2(self.lat4_1(u4)))
        # Top down
        # d5 = torch.randn(lat4.shape, dtype=torch.float32)
        d4 = self.td4_2(self.td4_1(torch.cat([lat4.squeeze(dim=2), d5], dim=1)))
        # print(self.td4_1(torch.cat([lat4.squeeze(), d5], dim=1)).shape, d4.shape)
        # print(lat3.shape, d4.shape)
        # squeeze(dim=2), top down input should be 4-dimensional(note that only squeeze C dim)
        d3 = self.td3_2(self.td3_1(torch.cat([lat3.squeeze(dim=2), d4], dim=1)))
        d2 = self.td2_2(self.td2_1(torch.cat([lat2.squeeze(dim=2), d3], dim=1)))
        d1 = self.td1_2(self.td1_1(torch.cat([lat1.squeeze(dim=2), d2], dim=1)))  # shape:(N, 32, 128, 160)
        # output layer
        output = self.out(self.conv(d1))
        return output


if __name__ == '__main__':
    gen = Generator()
    num_param = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f'Total parameters: {num_param}')  # result = 497694435

    num_lat_param = 0
    num_bu_param = 0
    num_td_param = 0
    for name, param in gen.named_parameters():
        # print(name, param.requires_grad)
        if name[:3] == 'lat' and param.requires_grad:
            num_lat_param += param.numel()
        elif name[:2] == 'bu' and param.requires_grad:
            num_bu_param += param.numel()
        elif name[:2] == 'td' and param.requires_grad:
            num_td_param += param.numel()

    print(f'BU parameters: {num_bu_param}\nLAT parameters: {num_lat_param}\nTD parameters: {num_td_param}')
