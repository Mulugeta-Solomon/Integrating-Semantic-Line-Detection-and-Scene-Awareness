import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
import sys
sys.path.append('/home/malab/Desktop/Research/FINAL')

from base import _C

class MaskEncoder(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_classes = 6

    def __call__(self, annotation):
        targets = []

        for ann in annotation:
            target = self._process_per_image(ann)
            targets.append(target)
        
        return default_collate(targets)
    
    def _process_per_image(self, ann):
        junctions = ann['junctions']
        device = junctions.device
        height, width = ann['height'], ann['width']

        edges_positive = ann['edges_positive']
        # edges_negative = ann['edges_negative']

        # if not isinstance(edges_positive, np.ndarray):
        #     edges_positive = np.array(edges_positive)
        
        lines = torch.cat((junctions[edges_positive[:,0]],junctions[edges_positive[:,1]]), dim=-1)
        
        line_annotations = torch.tensor(ann['line_annotations'], dtype=torch.float32, device=device)
        environment_class = 1 if ann['environment_annotation'] else 0

        # print(lines.shape, height, width, height, width, lines.size(0), line_annotations)

        lmap, segmentation_mask, _, _ = _C.encodels(lines, height, width, height, width, lines.size(0), line_annotations)
        # print(segmentation_mask.shape, environment_class)
        segmentation_mask = segmentation_mask - 1

        

        return {'seg_mask': segmentation_mask, 'env_class': environment_class}
    
