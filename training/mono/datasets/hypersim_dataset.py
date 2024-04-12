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
import h5py

def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float), np.arange(0, W, dtype=np.float), indexing='ij')
    meshgrid = np.stack((x,y))
    ones = np.ones((1,H*W), dtype=np.float)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

class HypersimDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(HypersimDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        #self.cap_range = self.depth_range # in meter
                # init uv

        # meshgrid for depth reprojection
        self.xy = creat_uv_mesh(768, 1024)
        
    def load_batch(self, meta_data, data_path):
        curr_intrinsic = meta_data['cam_in']
        # load rgb/depth
        curr_rgb, curr_depth = self.load_rgb_depth(data_path['rgb_path'], data_path['depth_path'])
        # get semantic labels
        curr_sem = self.load_sem_label(data_path['sem_path'], curr_depth)
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)       
        # get normal labels
        curr_normal = self.load_norm_label(data_path['normal_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1], depth=curr_depth, K=curr_intrinsic) # !!! this is diff of BaseDataset
        # get depth mask
        depth_mask = self.load_depth_valid_mask(data_path['depth_mask_path'])
        curr_depth[~depth_mask] = -1
        data_batch = dict(
            curr_rgb = curr_rgb,
            curr_depth = curr_depth,
            curr_sem = curr_sem,
            curr_normal = curr_normal,
            curr_cam_model=curr_cam_model,
        )
        return data_batch

    def load_data_path(self, meta_data):
        # 'rgbs': {'rgb_color': 'Hypersim/data/ai_001_001/images/scene_cam_00_final_preview/frame.0008.color.jpg', 
        #          'rgb_gamma': 'Hypersim/data/ai_001_001/images/scene_cam_00_final_preview/frame.0008.gamma.jpg', 
        #          'rgb_tonemap': 'Hypersim/data/ai_001_001/images/scene_cam_00_final_preview/frame.0008.tonemap.jpg', 
        #          'rgb_raw': 'Hypersim/data/ai_001_001/images/scene_cam_00_final_hdf5/frame.0008.color.hdf5'}
        meta_data['rgb'] = meta_data['rgbs']['rgb_color'] # this is diff of BaseDataset
        curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
        curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
        curr_sem_path = os.path.join(self.sem_root, meta_data['sem']) \
            if self.sem_root is not None and ('sem' in meta_data) and (meta_data['sem'] is not None)  \
            else None
        curr_norm_path = os.path.join(self.norm_root, meta_data['normal']) \
            if ('normal' in meta_data) and (meta_data['normal'] is not None) and (self.norm_root is not None) \
            else None
        curr_depth_mask_path = os.path.join(self.depth_mask_root, meta_data['depth_mask']) \
            if self.depth_mask_root is not None and ('depth_mask' in meta_data) and (meta_data['depth_mask'] is not None)  \
            else None

        data_path=dict(
            rgb_path=curr_rgb_path,
            depth_path=curr_depth_path,
            sem_path=curr_sem_path,
            normal_path=curr_norm_path,
            depth_mask_path=curr_depth_mask_path,
            )
        return data_path

    def load_rgb_depth(self, rgb_path: str, depth_path: str):
        """
        Load the rgb and depth map with the paths.
        """
        rgb = self.load_data(rgb_path, is_rgb_img=True)
        if rgb is None:
            self.logger.info(f'>>>>{rgb_path} has errors.')
       
        # depth = self.load_data(depth_path)
        with h5py.File(depth_path, "r") as f: depth = f["dataset"][:]
        np.nan_to_num(depth, copy=False, nan=0) # fill nan in gt
        if depth is None:
            self.logger.info(f'{depth_path} has errors.')
        
        depth = depth.astype(np.float)
        
        depth  = self.process_depth(depth, rgb)
        return rgb, depth


    def load_norm_label(self, norm_path, H, W, depth, K):
        with h5py.File(norm_path, "r") as f: 
            normal = f["dataset"][:]
        np.nan_to_num(normal, copy=False, nan=0)
        normal[:,:,1:] *= -1
        normal = normal.astype(np.float)

        return self.align_normal(normal, depth, K, H, W)

    def process_depth(self, depth: np.array, rgb: np.array) -> np.array:
        depth[depth>60000] = 0
        depth = depth / self.metric_scale
        return depth
    
    def align_normal(self, normal, depth, K, H, W):
        '''
        Orientation of surface normals in hypersim is not always consistent
        see https://github.com/apple/ml-hypersim/issues/26
        '''
        # inv K
        K = np.array([[K[0], 0 ,K[2]], 
                      [0, K[1], K[3]], 
                      [0, 0, 1]])
        inv_K = np.linalg.inv(K)
        # reprojection depth to camera points
        if H == 768 and W == 1024:
            xy = self.xy
        else:
            print('img size no-equal 768x1024')
            xy = creat_uv_mesh(H, W)
        points = np.matmul(inv_K[:3, :3], xy).reshape(3, H, W)
        points = depth * points
        points = points.transpose((1,2,0))

        # align normal
        orient_mask = np.sum(normal * points, axis=2) > 0
        normal[orient_mask] *= -1

        return normal