import torch
import torch.nn as nn
import torch.nn.functional as F


def segmentation_loss(logits, target):
    loss = F.cross_entropy(logits, target)
    return loss

def classification_loss(logits, target):
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss