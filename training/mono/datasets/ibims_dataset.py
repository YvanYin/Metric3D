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


class IBIMSDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(IBIMSDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale

        self.avg = torch.nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.unfold = torch.nn.Unfold(kernel_size=7, dilation=1, padding=0, stride=1)
        self.pad = torch.nn.ZeroPad2d(3)
    
    
    def process_depth(self, depth, rgb):
        depth[depth>50000] = 0
        depth /= self.metric_scale
        return depth

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

    def load_norm_label(self, norm_path, H, W, depth, K):
        depth = torch.from_numpy(depth).squeeze()
        K = torch.Tensor([[K[0], 0 ,K[2]], 
                      [0, K[1], K[3]], 
                      [0, 0, 1]])
        K_inv = K.inverse()

        y, x = torch.meshgrid([torch.arange(0, 480, dtype=torch.float32),
                            torch.arange(0, 640, dtype=torch.float32)], indexing='ij')
        x = x.reshape(1, 480*640)
        y = y.reshape(1, 480*640)
        ones = torch.ones_like(x)
        coord_2d = torch.cat((x, y, ones), dim=0)

        coord_3d = torch.matmul(K_inv, coord_2d).view(3, 480, 640)
        coord_3d = (coord_3d * depth[None, :])[None, :]
        coord_3d_mean = self.avg(coord_3d)

        uf_coord_3d = self.unfold(coord_3d.permute(1, 0, 2, 3))
        coord_3d_decenter = uf_coord_3d - coord_3d_mean.view(3, 1, (480-6)*(640-6))
        coord_3d_decenter = coord_3d_decenter.permute(2, 0, 1)
        cov = torch.bmm(coord_3d_decenter, coord_3d_decenter.permute(0, 2, 1))
        
        eig = torch.linalg.eigh(cov)
        #svd = torch.linalg.svd(coord_3d_decenter)
        normal = (eig[1])[:, :, 0].float()
        #normal = (svd[1])[:, 2, :]
        normal = self.pad(normal.permute(1, 0).view(1, 3, (480-6), (640-6)))
        
        orient_mask = (torch.sum(normal * coord_3d, axis=1) < 0).unsqueeze(1)
        normal = normal * orient_mask - normal * (~orient_mask)
        gt_normal = normal.squeeze().permute(1, 2, 0).numpy()
        return gt_normal

if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = IBIMSDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    