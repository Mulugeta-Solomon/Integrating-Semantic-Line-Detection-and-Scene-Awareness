import torch
import numpy as np
import random
import time 
import datetime
import os 
import os.path as osp
import argparse
import logging
import json
from tqdm import tqdm
import sys
sys.path.append('/home/malab/Desktop/Research/FINAL')

from config import cfg
from base.utils.comm import to_device
from base.utils.logger import setup_logger
from base.utils.checkpoint import DetectronCheckpointer
from base.utils.metric_logger import MetricLogger
from base.utils.miscellaneous import save_config

from model import build_model
from dataset import build_train_dataset
from solver import make_lr_scheduler, make_optimizer
from config import cfg
from config.paths_catalog import DatasetCatalog


AVAILABLE_DATASETS = ('wireframe_test', 'york_test')

def get_output_dir(root, basename):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return os.path.join(root, basename, timestamp)

class LossReducer(object):
    def __init__(self, cfg):
        try:
            self.lose_weights = dict(cfg.MODEL.LOSS_WEIGHTS)
            logger.info('Loss weights: {}'.format(self.lose_weights))
        except AttributeError:
            raise AttributeError('The config file must contain the LOSS_WEIGHTS')

    def __call__(self, loss_dict):
        total_loss = 0
        for key, weight in self.lose_weights.items():
            total_loss += loss_dict[key] * weight
        return total_loss


def train(cfg, model, train_dataset, optimizer, scheduler, loss_reducer, checkpointer, arguments):
    logger = logging.getLogger('Semantic Segmentation Training')
    device = cfg.MODEL.DEVICE
    model = model.to(device)
    start_training_time = time.time()
    end = time.time()

    start_epoch = arguments['epoch']
    num_epochs = arguments['max_epoch'] - start_epoch
    epoch_size = len(train_dataset)

    epoch = arguments['epoch'] + 1

    total_iterations = num_epochs * epoch_size
    step = 0

    logger.info('Start training')

    for epoch in tqdm(range(start_epoch+1, start_epoch+num_epochs+1), total=start_epoch+num_epochs+1, desc='Epochs'):
        model.train()
        loss_meters = MetricLogger(' ')
        aux_meters = MetricLogger(' ')
        sys_meters = MetricLogger(' ')

        for itr, (images, annotations) in enumerate(train_dataset):
            date_time = time.time() - end
            
            images = to_device(images, device)
            annotations = to_device(annotations, device)

            loss_dict = model(images, annotations)
            loss = loss_reducer(loss_dict)

            with torch.no_grad():
                loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
                loss_reduced = loss.item()
                loss_meters.update(loss = loss_reduced, **loss_dict_reduced)
                #aux_meters.update(date_time = date_time)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            sys_meters.update(time=batch_time, data_time=date_time)

            total_iterations -= 1
            step += 1

            eta_seconds = sys_meters.time.global_avg * total_iterations
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if itr % 20 == 0:
                logger.info(
                    ' '.join(
                        [ 
                            'Epoch: [{epoch}/{total_epoch}]',
                            'ETA: {eta}',
                            'ITER: [{iter}/{total_iterations}]',
                            'LOSS: {loss_meter}',
                            'LR: {lr:.6f}',
                            'RUNTIME: {sys_meter}'

                        ]
                        ).format(
                            epoch=epoch,
                            total_epoch=start_epoch+num_epochs,
                            eta=eta_string,
                            iter=itr,
                            total_iterations=total_iterations,
                            loss_meter=loss_meters,
                            lr=optimizer.param_groups[0]['lr'],
                            sys_meter=sys_meters

                        )
                    )
        scheduler.step()

        checkpointer.save('model_{:07d}'.format(epoch))
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=int(total_training_time)))
    
    logger.info(
        'Total training time: {} ({:.4f} s / epoch)'.format(
            total_time_str, total_training_time / max_epoch
            )
        )

    logger.info('Training has finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Segmentation for Wireframe Classification Training')

    parser.add_argument('--config', type=str , help='path to config file')  # metavar='FILE'
    parser.add_argument('--logdir', required=True, type=str, help='path to log directory')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--seed', default=42, type=int, help='set seed for reproducibility')
    parser.add_argument('--tf32', default=False, action='store_true', help='toggle on the TF32 of pytorch')
    parser.add_argument('--dtm', default=True, choices=[True, False], help='toggle the deterministic option of CUDNN. This option will affect the replication of experiments')

    args = parser.parse_args()
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.deterministic = args.dtm

    assert args.config.endswith('.yaml') or args.config.endswith('.yml'), 'The config file must be a yaml file'
    config_basename = os.path.basename(args.config)
    if config_basename.endswith('.yaml'):
        config_basename = config_basename[:-5]
    else:
        config_basename = config_basename[:-4]

    cfg.merge_from_file(args.config)

    output_dir = get_output_dir(args.logdir, config_basename)

    cfg.OUTPUT_DIR = output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger('Semantic Segmentation Training', output_dir, out_file = 'train.log')

    logger.info(args)
    logger.info('Loaded configuration file {}'.format(args.config))

    with open(args.config, 'r') as f:
        config_str = '\n' + f.read()
        logger.info(config_str)

    logger.info('Running with config:\n{}'.format(cfg))
    output_config_path = os.path.join(output_dir, 'config.yaml')
    logger.info('Saving config into: {}'.format(output_config_path))
    save_config(cfg, output_config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    model = build_model(cfg)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    loss_reducer = LossReducer(cfg)

    arguments = {}
    arguments['epoch'] = 0
    max_epoch = cfg.SOLVER.MAX_EPOCH
    arguments['max_epoch'] = max_epoch

    checkpointer = DetectronCheckpointer(cfg, model, optimizer, save_dir=cfg.OUTPUT_DIR, save_to_disk=True, logger=logger)

    if args.resume:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)
        logger.info('Loading the pretrained model from {}'.format(args.resume))
    
    train_dataset = build_train_dataset(cfg)
    logger.info('Train dataset: {}'.format(train_dataset))
  
    logger.info('Epoch size = {}'.format(len(train_dataset)))

    train(cfg, model, train_dataset, optimizer, scheduler, loss_reducer, checkpointer, arguments)


    # import pdb; pdb.set_trace()



