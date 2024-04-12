import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
from .__base_dataset__ import BaseDataset


class UASOLDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(UASOLDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        return depth
    
    def load_rgb_depth(self, rgb_path: str, depth_path: str) -> (np.array, np.array):
        """
        Load the rgb and depth map with the paths.
        """
        rgb = self.load_data(rgb_path, is_rgb_img=True)
        if rgb is None:
            self.logger.info(f'>>>>{rgb_path} has errors.')
       
        depth = self.load_data(depth_path)
        if depth is None:
            self.logger.info(f'{depth_path} has errors.')
        
        depth = depth.astype(np.float)
        
        depth  = self.process_depth(depth, rgb)
        depth = depth[1:-1, ...]
        return rgb, depth



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = UASOLDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    