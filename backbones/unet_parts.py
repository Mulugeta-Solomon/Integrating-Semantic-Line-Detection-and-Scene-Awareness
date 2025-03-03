import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Inconv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
       # Resize x1 to match the spatial dimensions of x2
        if isinstance(self.up, nn.Upsample):
           x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)
        # Concatenate x2 and x1 along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x