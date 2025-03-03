import torch
import numpy as np
import random
import os
import os.path as osp
import argparse
import logging
from tqdm import tqdm
import json
import sys
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.append('/home/malab/Desktop/Research/FINAL')

from config import cfg
from base.utils.comm import to_device
from base.utils.logger import setup_logger
from base.utils.checkpoint import DetectronCheckpointer
from base.utils.metric_logger import MetricLogger
from base.utils.miscellaneous import save_config

from model import build_model
from dataset import build_transform
from solver import make_lr_scheduler, make_optimizer
from config.paths_catalog import DatasetCatalog
from model.mask_encoder import MaskEncoder
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

class ImageList(IterableDataset):
    def __init__(self, image_paths, transform):
        super().__init__()
        self.image_paths = image_paths
        self.transform = transform

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError("Single worker only.")
        for image_path in self.image_paths:
            im = Image.open(image_path)
            w, h = im.size
            meta = {
                "filename": image_path,
                "height": h,
                "width": w,
            }
            yield self.transform(np.array(im)), meta

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def plot_result(image, segmentation_mask_pred, classification_logits_pred, size):
    custom_colors = ['#B91C1C', '#34D399', '#60A5FA', '#FBBF24', '#C084FC', '#F97316']
    custom_cmap = ListedColormap(custom_colors)

    segmentation_mask_pred = Image.fromarray(segmentation_mask_pred.astype(np.uint8))
    segmentation_mask_pred = segmentation_mask_pred.resize(size, Image.NEAREST)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image)
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(segmentation_mask_pred, cmap=custom_cmap)
    axs[1].set_title(f'Segmentation Mask\nClassification: {"Indoor" if classification_logits_pred else "Outdoor"}')
    axs[1].axis('off')

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Semantic Segmentation prediction')
    parser.add_argument('--config-file', default='', help='path to config file')
    parser.add_argument('--image-path', required=True, help='path to the image directory')
    parser.add_argument('--ckpt', required=True, type=str, help='path to the checkpoint file')
    parser.add_argument('--output', required=True, type=str, help='path to the output directory')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')

    args = parser.parse_args()

    config_path = args.config_file
    cfg.merge_from_file(config_path)

    root = args.output
    logger = setup_logger('Semantic Segmentation Prediction', root, out_file='predict.log')
    logger.info(args)
    logger.info('Loaded configuration file {}'.format(config_path))

    set_random_seed(args.seed)

    device = cfg.MODEL.DEVICE
    logger.info('Running on device: {}'.format(device))

    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    transform = build_transform(cfg)
    image_paths = [osp.join(args.image_path, fname) for fname in os.listdir(args.image_path) if fname.endswith(('png', 'jpg'))]
    dataset = ImageList(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)
    
    pred_masks = []

    for image, targets in dataloader:
        with torch.no_grad():   
            image = to_device(image, device)
            segmentation_logits, classification_logits = model(image)
            segmentation_mask = torch.argmax(segmentation_logits, dim=1).squeeze().cpu().numpy()
            
            image_path = targets['filename'][0]
            image = Image.open(image_path).convert('RGB')
            size = image.size
            logger.info(f'Predicting on {image_path}')
            plot_result(image, segmentation_mask, classification_logits, size)
            image_path = image_path.split('/')[-1]
        pred_masks.append({'seg_pred':segmentation_mask.tolist(),
                            'filename': image_path})

    logger.info("Prediction has finished!")

    try:
        with open('curr_pred.json', 'w') as f:
            json.dump(pred_masks, f)
            logger.info('Prediction saved to curr_pred.json')
    except Exception as e:
        logger.error(f"Error saving prediction: {e}")
    

if __name__ == '__main__':
    main()
