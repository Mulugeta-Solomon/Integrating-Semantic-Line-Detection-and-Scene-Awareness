import torch
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('/home/malab/Desktop/Research/FINAL')


from backbones import build_backbone
from collections import defaultdict

from .losses import classification_loss, segmentation_loss
from .mask_encoder import MaskEncoder

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.mask_encoder = MaskEncoder(cfg)
        self.classification_loss = classification_loss
        self.segmentation_loss = segmentation_loss
        self.train_step = 0
    
    def forward(self, images, annotations = None, target = None):
        if self.training:
            return self.forward_train(images, annotations = annotations)
        else:
            return self.forward_test(images, annotations = annotations)

    def forward_train(self, images, annotations=None):
        device = images.device

        targets = self.mask_encoder(annotations)
        # print('Targets:', targets["seg_mask"].shape, targets["env_class"].shape)

        self.train_step += 1

        segmentation_logits, classification_logits = self.backbone(images)
        # print('Segmentation Logits:', len(segmentation_logits), segmentation_logits[0].shape)

        loss_dict = {'loss_seg': 0, 'loss_cls': 0}

        # print('####################################################################')
        # print('Segmentation Logits:', segmentation_logits[0].shape)
        # print('Classification Logits:', classification_logits[0])
        # print(targets)

        # if targets is not None:
        #     for nstack, output in enumerate(segmentation_logits):
        #         loss_dict['seg_loss'] += self.segmentation_loss(segmentation_logits[nstack], targets['seg_mask'])
        #         loss_dict['cls_loss'] += self.classification_loss(classification_logits[nstack], targets['env_class'])

        if targets is not None:
            # print(f"Original Seg Mask Shape: {targets['seg_mask'].shape}")
            # print(f"Original Env Class Shape: {targets['env_class'].shape}")

            seg_masks = targets['seg_mask'].squeeze(1).to(device)
            env_classes = targets['env_class'].to(device).float()

            for i in range(len(targets['seg_mask'])):
                # print(targets['seg_mask'][i].shape, targets['env_class'][i].shape)
                # print(segmentation_logits[i].shape, classification_logits[i].shape)
                
                # print(f'Segmentation Logits[{i}] Shape:', segmentation_logits[i].shape)
                # print(f'Target Seg Mask[{i}] Shape:', seg_masks[i].shape)
                # print(f'Classification Logits[{i}] Shape:', classification_logits[i].shape)
                # print(f'Target Env Class[{i}] Shape:', env_classes[i].shape)

                class_logit = classification_logits[i].unsqueeze(0).to(device)
                env_class = env_classes[i].unsqueeze(0).to(device)

                seg_loss = self.segmentation_loss(segmentation_logits[i].unsqueeze(0), targets['seg_mask'][i].unsqueeze(0).squeeze(0).long())
                cls_loss = self.classification_loss(class_logit, env_class)
                loss_dict['loss_seg'] += seg_loss
                loss_dict['loss_cls'] += cls_loss
            
            # print(f"Segmentation Loss: {loss_dict}")
        
        return loss_dict

    def forward_test(self, images, annotations=None):
        device = images.device

        segmentation_logits, classification_logits = self.backbone(images)

        return segmentation_logits, classification_logits


def build_model(pretrained=False):
    from config import cfg
    import os

    return Model(cfg)


