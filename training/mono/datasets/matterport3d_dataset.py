import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset
import random
from .__base_dataset__ import BaseDataset


class Matterport3DDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(Matterport3DDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        #self.cap_range = self.depth_range # in meter

    def load_norm_label(self, norm_path, H, W):
        normal_x = cv2.imread(norm_path['x'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        normal_y = cv2.imread(norm_path['y'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        normal_z = cv2.imread(norm_path['z'], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        raw_normal = np.array([normal_x, normal_y, normal_z])
        invalid_mask = np.all(raw_normal == 0, axis=0)

        ego_normal = raw_normal.astype(np.float64) / 32768.0 - 1
        ego2cam = np.array([[1,0,0],
                            [0,-1,0],
                            [0,0,-1]])
        normal = (ego2cam @ ego_normal.reshape(3,-1)).reshape(ego_normal.shape)
        normal[:,invalid_mask] = 0
        normal = normal.transpose((1,2,0))
        if normal.shape[0] != H or normal.shape[1] != W:
            normal = cv2.resize(normal, [W,H], interpolation=cv2.INTER_NEAREST)
        return normal
    
    def process_depth(self, depth: np.array, rgb: np.array) -> np.array:
        depth[depth>65500] = 0
        depth = depth / self.metric_scale
        return depth
