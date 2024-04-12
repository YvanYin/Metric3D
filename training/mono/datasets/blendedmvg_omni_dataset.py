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


class BlendedMVGOmniDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(BlendedMVGOmniDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        #self.cap_range = self.depth_range # in meter
    
    # def __getitem__(self, idx: int) -> dict:
    #     if self.phase == 'test':
    #         return self.get_data_for_test(idx)
    #     else:
    #         return self.get_data_for_trainval(idx)

    
    def process_depth(self, depth: np.array, rgb: np.array) -> np.array:
        depth[depth>60000] = 0
        depth = depth / self.metric_scale
        return depth
