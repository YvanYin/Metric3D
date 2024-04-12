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


class HM3DDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(HM3DDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        #self.cap_range = self.depth_range # in meter

    def load_norm_label(self, norm_path, H, W):
        with open(norm_path, 'rb') as f:
            normal = Image.open(f)
            normal = np.array(normal.convert(normal.mode), dtype=np.uint8)
        invalid_mask = np.all(normal == 128, axis=2)
        normal = normal.astype(np.float64) / 255.0 * 2 - 1
        normal[invalid_mask, :] = 0
        return normal

    def process_depth(self, depth: np.array, rgb: np.array) -> np.array:
        depth[depth>60000] = 0
        depth = depth / self.metric_scale
        return depth
