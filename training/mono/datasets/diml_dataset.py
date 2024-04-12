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
import pickle

class DIMLDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(DIMLDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def load_meta_data(self, anno: dict) -> dict:
        """
        Load meta data information.
        """
        if self.meta_data_root is not None and ('meta_data' in anno or 'meta' in anno):
            meta_data_path = os.path.join(self.meta_data_root, anno['meta_data']) if 'meta_data' in anno else os.path.join(self.meta_data_root, anno['meta'])
            with open(meta_data_path, 'rb') as f:
                meta_data = pickle.load(f)
            meta_data.update(anno)
        else:
            meta_data = anno
        
        # DIML_indoor has no cam_in
        if 'cam_in' not in meta_data:
            meta_data['cam_in'] = [1081, 1081, 704, 396]
        return meta_data
   
    def process_depth(self, depth, rgb):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        h, w, _ = rgb.shape # to rgb size
        depth_resize = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth_resize




if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = DIMLDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    