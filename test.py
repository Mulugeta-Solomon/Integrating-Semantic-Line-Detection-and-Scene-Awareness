import torch
import numpy as np
import random
import time 
import datetime
import os 
import os.path as osp
import argparse
import json
from tqdm import tqdm
import sys
sys.path.append('/home/malab/Desktop/Research/FINAL')
from config import cfg
from base.utils.comm import to_device
from base.utils.logger import setup_logger
from model import build_model
from dataset import build_transform
from config import cfg
from model.mask_encoder import MaskEncoder
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from model.mask_encoder import MaskEncoder
from base import _C
from collections import defaultdict


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

def get_data(ann, filename):

    for i in range(len(ann)):
        if ann[i]['filename'] == filename:
            return ann[i]
    return None

def plot_pr_curve(precision_dict, recall_dict, fscore_dict, num_classes, output_path, n_gt = None):
    custom_colors = ['#B91C1C', '#34D399', '#60A5FA', '#FBBF24', '#C084FC', '#F97316']
    cmap = ListedColormap(custom_colors)
    class_names = [
        'UpperEdge', 
        'wallEdge', 
        'LowerEdge', 
        'doorEdge', 
        'windowEdge', 
        'miscellaneous'
    ]
    
    plt.figure()
    f_scores = np.linspace(0.2, 0.9, num=8).tolist()
    
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        plt.plot(x[y >= 0], y[y >= 0], color=[0, 0.5, 0], alpha=0.3)
        plt.annotate(f"f={f_score:0.1f}", xy=(0.9, y[45] + 0.02), alpha=0.4, fontsize=10)
    
    plt.rc('legend', fontsize=10)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))

    for class_id in range(num_classes):
        precision = np.array(precision_dict[class_id])
        recall = np.array(recall_dict[class_id])
        f_score = np.array(fscore_dict[class_id])
        
        sorted_indices = np.argsort(f_score)[::-1]
        precision = precision[sorted_indices]
        sorted_indices = np.argsort(-f_score)[::-1]
        recall = recall[sorted_indices]

        color = cmap(class_id)

        # plt.plot(recall, precision, marker='.', color=color, linestyle='-', label=f'{class_names[class_id]} (F-score={np.mean(f_score):.2f})')

        cumulative_precision = np.cumsum(precision) / np.arange(1, len(precision) + 1)
        cumulative_recall = np.cumsum(recall) / np.arange(1, len(recall) + 1)

        # Debugging statements to ensure data validity
        print(f"Class ID: {class_id}, Precision: {np.mean(cumulative_precision)}, Recall: {np.mean(cumulative_recall)}, F-score: {np.mean(f_score)}")
        plt.plot(cumulative_recall, cumulative_precision, marker= '.', color=color, linestyle='-', label=f'{class_names[class_id]} (F-score={np.mean(f_score):.2f})')

   
    plt.legend(loc='lower right')
    plt.grid(True)
    # plt.savefig(output_path)
    plt.show()


def sAPEval(predictions, targets, num_classes):
    precision_dict = defaultdict(list)
    recall_dict = defaultdict(list)
    fscore_dict = defaultdict(list)

    for filename in predictions.keys():
        pred_mask = predictions[filename]
        target_mask = targets[filename]

        for class_id in range(num_classes):
            binary_predictions = (pred_mask == class_id).astype(int)
            binary_targets = (target_mask == class_id).astype(int)

            tp = np.sum((binary_predictions == 1) & (binary_targets == 1))
            fp = np.sum((binary_predictions == 1) & (binary_targets == 0))
            fn = np.sum((binary_predictions == 0) & (binary_targets == 1))

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f_score = 2 * (precision * recall) / (precision + recall + 1e-8)

            precision_dict[class_id].append(precision)
            recall_dict[class_id].append(recall)
            fscore_dict[class_id].append(f_score)

            # print(f"Filename: {filename}, Class ID: {class_id}, TP: {tp}, FP: {fp}, FN: {fn}, Precision: {precision}, Recall: {recall}, F-score: {f_score}")

    # for key in precision_dict.keys():
    #     print(f'{key}:{precision_dict[key]}')


    # avg_precision_dict = {class_id: np.mean(precision_dict[class_id]) for class_id in range(num_classes)}
    # avg_recall_dict = {class_id: np.mean(recall_dict[class_id]) for class_id in range(num_classes)}
    # avg_fscore_dict = {class_id: np.mean([fscore_dict[class_id]]) for class_id in range(num_classes)}
    # print(f"Average Precision: {avg_precision_dict}, Average Recall: {avg_recall_dict}, Average F-score: {avg_fscore_dict}")
    return precision_dict, recall_dict, fscore_dict


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Semantic Segmentation Benchmark')
    args.add_argument('--config-file', default='', help='path to config file')
    args.add_argument('--image-path', required=True, help='path to the image directory')
    args.add_argument('--ckpt', required=True, type=str, help='path to the checkpoint file')
    args.add_argument('--output', required=True, type=str, help='path to the output directory')
    args.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    args.add_argument('--annotations', required=True, help='path to the annotation file for evaluation')

    args = args.parse_args()

    config_path = args.config_file
    cfg.merge_from_file(config_path)

    root = args.output
    
    logger = setup_logger('Semantic Segmentation Benchmark', root, out_file='predict.log')
    logger.info(args)
    logger.info('Loaded configuration file {}'.format(config_path))

    set_random_seed(args.seed)

    device = cfg.MODEL.DEVICE
    logger.info('Running on device: {}'.format(device))

    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    logger.info('Running on device: {}'.format(device))

    model = build_model(cfg).to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model = model.eval()

    with open(args.annotations, 'r') as f:
        ann = json.load(f)
    
    filenames = {ann[i]['filename'] for i in range(len(ann))}

    transform = build_transform(cfg)
    
    image_paths = [osp.join(args.image_path, fname) for fname in os.listdir(args.image_path) if fname.endswith('png') and fname in filenames]
    dataset = ImageList(image_paths, transform)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)

    mask_encoder = MaskEncoder(cfg)

    predictions = defaultdict()
    targets = defaultdict()
    pred_masks = []


    for image, meta in tqdm(dataloader, total=len(ann), desc='Predicting'):
        with torch.no_grad():
            image = to_device(image, device)
            segmentation_logits, classification_logits = model(image)
            segmentation_mask = torch.argmax(segmentation_logits, dim=1).squeeze().cpu().numpy()
            
            image_path = meta['filename'][0]
            image = Image.open(image_path).convert('RGB')
            size = image.size
            segmentation_mask_pred = Image.fromarray(segmentation_mask.astype(np.uint8))
            segmentation_mask_pred = segmentation_mask_pred.resize(size, Image.NEAREST)
            segmentation_mask_pred = np.array(segmentation_mask_pred)
            image_path = image_path.split('/')[-1]

        predictions[image_path] = segmentation_mask_pred
        pred_masks.append({'seg_pred':segmentation_mask_pred.tolist(),
                            'filename': image_path})
    
    # a = pred_masks[1]
    # for key, val in a.items():
    #     print(key)
    #     print(len(val), len(val[0]))
    # # print(len(pred_masks[0][0]),len(pred_masks[0]))
    with open('predictions.json', 'w') as f:
        json.dump(pred_masks, f)
        logger.info("Prediction has been saved to predictions.json")

    logger.info("Prediction has finished!")


    for i in tqdm(range(len(ann)), desc='Loading Targets'):
        junctions = torch.tensor(ann[i]['junctions'], dtype=torch.float32)
        junctions = junctions.to(device)
        height, width = ann[i]['height'], ann[i]['width']
        edges_positive = torch.tensor(ann[i]['edges_positive']) 
        line_annotations = torch.tensor(ann[i]["line_annotations"], dtype=torch.float32, device=device)
        lines = torch.cat((junctions[edges_positive[:,0]], junctions[edges_positive[:,1]]),dim=-1)
        lmap, annotation, _, _ = _C.encodels(lines,height,width,height,width,lines.size(0), line_annotations)
        annotation = annotation - 1
        targets[ann[i]['filename']] = annotation.squeeze().cpu().numpy()

    logger.info("Targets have been loaded!")

    num_classes = cfg.MODEL.NUM_CLASSES - 1
    precion_dict, recall_dict, fscore_dict = sAPEval(predictions, targets, num_classes)
    plot_pr_curve(precion_dict, recall_dict, fscore_dict, num_classes, os.path.join(root, 'pr_curve.png'))
    logger.info("Precision-Recall Curve has been plotted!")

