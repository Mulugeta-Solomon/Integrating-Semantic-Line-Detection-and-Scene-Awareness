from .registry import BACKBONES
from .unet import UNet
import torch

@BACKBONES.register('unet')
def build_unet(cfg):
    model = UNet(n_channels=cfg.MODEL.UNET.INPUT_CHANNELS, n_classes=cfg.MODEL.UNET.NUM_CLASSES)
    return model

def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE in BACKBONES, \
        f"Backbone '{cfg.MODEL.BACKBONE}' is not registered"
    
    return BACKBONES[cfg.MODEL.BACKBONE](cfg)