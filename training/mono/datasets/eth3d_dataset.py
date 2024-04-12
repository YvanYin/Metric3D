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


class ETH3DDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(ETH3DDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    def __getitem__(self, idx):
        anno = self.annotations['files'][idx]
        curr_rgb_path = os.path.join(self.data_root, anno['rgb_path'])
        curr_depth_path = os.path.join(self.depth_root, anno['depth_path'])
        meta_data = self.load_meta_data(anno)
        ori_curr_intrinsic = [2000, 2000, 3024, 2016] #meta_data['cam_in']
        
        curr_rgb = cv2.imread(curr_rgb_path) # [r, g, b]
        with open(curr_depth_path, 'r') as f:
            imgfile = np.fromfile(f, np.float32)
            curr_depth = imgfile.reshape((4032, 6048))
            curr_depth[curr_depth>100] = 0
        
        #curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        # curr_rgb = cv2.resize(curr_rgb, dsize=(3024, 2016), interpolation=cv2.INTER_LINEAR)
        # curr_depth = cv2.resize(curr_depth, dsize=(3024, 2016), interpolation=cv2.INTER_LINEAR)
        # ori_curr_intrinsic = [i//2 for i in ori_curr_intrinsic]
        
        ori_h, ori_w, _ = curr_rgb.shape
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)
        # load tmpl rgb info
        # tmpl_annos = self.load_tmpl_annos(anno, curr_rgb, meta_data)
        # tmpl_rgb = tmpl_annos['tmpl_rgb_list'] # list of reference rgbs

        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb, ], 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic,], 
                                                                   cam_models=[curr_cam_model, ],
                                                                   transform_paras=transform_paras)
        # depth in original size
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1]

        filename = os.path.basename(anno['rgb_path'])
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]    
        raw_rgb = torch.from_numpy(curr_rgb)
        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    ref_input=rgbs[1:],
                    tmpl_flg=False,
                    pad=pad,
                    scale=scale_ratio,
                    raw_rgb=raw_rgb,
                    normal = np.zeros_like(curr_rgb.transpose((2,0,1))),
                    #stereo_depth=torch.zeros_like(depth_out)
                    ) 
        return data
    
    def process_depth(self, depth):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        return depth



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = NYUDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    