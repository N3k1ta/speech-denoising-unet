import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu_2 = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        return x
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = ConvBlock(1, 32)
        self.pool1  = nn.MaxPool2d(kernel_size=2)
        self.block2 = ConvBlock(32, 64)
        self.pool2  = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConvBlock(64, 128)

        self.upsample1 = nn.Upsample(scale_factor=2)
        self.decoder_block1 = ConvBlock(192, 64)
        self.upsample2 = nn.Upsample(size=(1025, 32))
        self.decoder_block2 = ConvBlock(96, 32)

        self.final_conv = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.pool1(x1)
        x3 = self.block2(x2)
        x4 = self.pool2(x3)
        x5 = self.bottleneck(x4)
        x6 = self.upsample1(x5)
        x7 = torch.cat([x6, x3], dim=1)
        x8 = self.decoder_block1(x7)
        x9  = self.upsample2(x8)
        x10 = torch.cat([x9, x1], dim=1)
        x11 = self.decoder_block2(x10)

        return self.final_conv(x11)