import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, Up, Inconv, Outconv

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inconv = Inconv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outconv = Outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inconv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outconv(x)

        segmentation_logits = logits[:, :6, :, :] # 6 classes for segmentation
        classification_logits = logits[:, 6:, :, :] # last channel for classification

        # # pool the classication logits to get a single value for each class
        # classification_logits = F.adaptive_avg_pool2d(classification_logits, (1, 1))   #.squeeze()
        # classification_logits = torch.sigmoid(classification_logits).view(-1)

        # pool the classification logits to get a single value for each class
        classification_logits = F.adaptive_avg_pool2d(classification_logits, (1, 1)).squeeze()
        classification_logits = torch.sigmoid(classification_logits)

        
        return segmentation_logits, classification_logits

# Path: model/models.py



