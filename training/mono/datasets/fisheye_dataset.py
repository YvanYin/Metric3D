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


class FisheyeDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(FisheyeDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def load_data(self, path: str, is_rgb_img: bool=False):
        if not os.path.exists(path):
            self.logger.info(f'>>>>{path} does not exist.')
            # raise RuntimeError(f'{path} does not exist.')

        data_type = os.path.splitext(path)[-1]
        if data_type in self.img_file_type:
            if is_rgb_img:
                data = cv2.imread(path)
            else:
                data = cv2.imread(path, -1)
                data[data>65500] = 0
                data &= 0x7FFF

        elif data_type in self.np_file_type:
            data = np.load(path)
        else:
            raise RuntimeError(f'{data_type} is not supported in current version.')
        
        return data.squeeze()

    def load_batch(self, meta_data, data_path):
        curr_intrinsic = meta_data['cam_in']
        # load rgb/depth
        curr_rgb, curr_depth = self.load_rgb_depth(data_path['rgb_path'], data_path['depth_path'])
        # get semantic labels
        curr_sem = self.load_sem_label(data_path['sem_path'], curr_depth)
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)       
        # get normal labels
        curr_normal = self.load_norm_label(data_path['normal_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1]) 
        # get depth mask
        depth_mask = self.load_depth_valid_mask(data_path['depth_mask_path'])[:, :, :]
         
        # with masks from andy
        curr_depth[~(depth_mask[:, :, 0])] = -1
        curr_rgb[~(depth_mask[:, :, :])] = 0
        
        # get stereo depth
        curr_stereo_depth = self.load_stereo_depth_label(data_path['disp_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1]) 

        data_batch = dict(
            curr_rgb = curr_rgb,
            curr_depth = curr_depth,
            curr_sem = curr_sem,
            curr_normal = curr_normal,
            curr_cam_model=curr_cam_model,
            curr_stereo_depth=curr_stereo_depth,
        )
        return data_batch

    
    def process_depth(self, depth, rgb):

        depth /= self.metric_scale
        return depth
