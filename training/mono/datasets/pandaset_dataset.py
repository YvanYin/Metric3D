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


class PandasetDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(PandasetDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        # depth[(depth>self.cap_range[1]) | (depth<self.cap_range[0])] = -1
        # depth /= self.cap_range[1]
        return depth



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = PandasetDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    