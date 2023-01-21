""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class InitBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.apply(weights_init)

    def forward(self, _x):
        return self.double_conv(_x)


class DownBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(DownBlock, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.apply(weights_init)

    def forward(self, _x):
        return self.double_conv(_x)


class BottleBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(BottleBlock, self).__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.bottle_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.apply(weights_init)

    def forward(self, _x):
        _x = self.down_conv(_x)
        _x = self.bottle_conv(_x)
        _x = self.up_conv(_x)
        return _x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )
        self.apply(weights_init)

    def forward(self, x_current, x_previous):
        x_current = self.up_conv1(x_current)

        _x = torch.cat([x_previous, x_current], dim=1)

        _x = self.up_conv2(_x)
        return _x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class FingerGAN(nn.Module):
    def __init__(self, in_channels):
        super(FingerGAN, self).__init__()

        # x = 192 x 192
        self.init_block = InitBlock(in_channels, 64) # 190 x 190, 188 x 188

        self.down1 = DownBlock(64, 128)     # 94 x 94,   92 x 92
        self.down2 = DownBlock(128, 256)    # 46 x 46,   44 x 44
        self.down3 = DownBlock(256, 512)    # 22 x 22,   20 x 20

        self.bottle = BottleBlock(512, 1024)   # 10 x 10,   8  x 8,  10 x 10

        self.up1 = UpBlock(1024, 512)       # 20 x 20, 22 x 22
        self.up2 = UpBlock(512, 256)        # 44 x 44, 46 x 46
        self.up3 = UpBlock(256, 128)        # 92 x 92, 94 x 94
        self.up4 = UpBlock(128, 64)         # 188 x 188, 190 x 190

        self.out_blcok = OutConv(64, in_channels)     # 192 x 192

    def forward(self, _x):
        x1 = self.init_block(_x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottle(x4)

        _x = self.up1(x5, x4)
        _x = self.up2(_x, x3)
        _x = self.up3(_x, x2)
        _x = self.up4(_x, x1)
        logits_sigmoid = self.out_blcok(_x)
        return logits_sigmoid


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()

        self.f1 = nn.Sequential(
            # input is (in_channel) x 192 x 192
            nn.Conv2d(in_channel, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.f2 = nn.Sequential(
            # state size. (64) x 96 x 96
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))
        self.f3 = nn.Sequential(
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2))
        self.f4 = nn.Sequential(
            # state size. (64*2) x 24 x 24
            nn.Conv2d(64 * 2, 64 * 2, 4, 2, 1),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2))

        self.f5 = nn.Sequential(
            # state size. (64*2) x 12 x 12
            nn.Conv2d(64*2, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2))

        self.f6 = nn.Sequential(
            # state size. (64*4) x 6 x 6
            nn.Conv2d(64 * 4, 64 * 4, 4, 2, 1),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2))

        self.f7 = nn.Sequential(
            # state size. (64*4) x 3 x 3
            nn.Conv2d(64 * 4, 1, 3),
            nn.Sigmoid())

        self.apply(weights_init)

    def forward(self, _input):
        x1 = self.f1(_input)
        x2 = self.f2(x1)
        x3 = self.f3(x2)
        x4 = self.f4(x3)
        x5 = self.f5(x4)
        x6 = self.f6(x5)
        x7 = self.f7(x6)
        x8 = x7.view(-1)
        return x8


if __name__ == "__main__":
    x = torch.randn(1, 2, 192, 192)
    fG = FingerGAN(2)

    fD = Discriminator(2)

    x_ = fG(x)
    y = fD(x_)

    pass
