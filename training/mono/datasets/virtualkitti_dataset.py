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


class VKITTIDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(VKITTIDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    

    
    def process_depth(self, depth, rgb):
        depth[depth>(150 * self.metric_scale)] = 0
        depth /= self.metric_scale

        return depth
    
    def load_sem_label(self, sem_path, depth=None, sky_id=142) -> np.array:
        """
            Category r g b
            Terrain 210 0 200
            Sky     90 200 255
            Tree     0 199 0
            Vegetation 90 240 0
            Building 140 140 140
            Road 100 60 100
            GuardRail 250 100 255
            TrafficSign 255 255 0
            TrafficLight 200 200 0
            Pole 255 130 0
            Misc 80 80 80
            Truck 160 60 60
            Car 255 127 80
            Van 0 139 139
        """
        H, W = depth.shape
        sem_label = np.ones((H, W), dtype=np.int) * -1
        sem = cv2.imread(sem_path)[:, :, ::-1]
        if sem is None:
            return sem_label
        
        sky_color = [90, 200, 255]
        sky_mask = (sem == sky_color).all(axis=2)
        sem_label[sky_mask] = 142 # set sky region to 142
        return sem_label



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = ApolloscapeDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    