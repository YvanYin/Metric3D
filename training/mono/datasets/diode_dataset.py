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


def creat_uv_mesh(H, W):
    y, x = np.meshgrid(np.arange(0, H, dtype=np.float), np.arange(0, W, dtype=np.float), indexing='ij')
    meshgrid = np.stack((x,y))
    ones = np.ones((1,H*W), dtype=np.float)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)

class DIODEDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(DIODEDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        
        # meshgrid for depth reprojection
        self.xy = creat_uv_mesh(768, 1024)

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # load data
        curr_rgb, curr_depth, curr_normal, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_cam_model']
        ori_curr_intrinsic = meta_data['cam_in']

        # get crop size
        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb,],  #+ tmpl_rgbs, 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic, ], # * (len(tmpl_rgbs) + 1), 
                                                                   cam_models=[curr_cam_model, ],
                                                                   transform_paras=transform_paras)
        # depth in original size and orignial metric***
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1] # self.clip_depth(depths[0]) #
        inv_depth = self.depth2invdepth(depth_out, np.zeros_like(depth_out, dtype=np.bool))
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        ori_curr_intrinsic_mat = self.intrinsics_list2mat(ori_curr_intrinsic)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]    
        raw_rgb = torch.from_numpy(curr_rgb)
        curr_normal = torch.from_numpy(curr_normal.transpose((2,0,1)))
        

        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=pad,
                    scale=scale_ratio,
                    raw_rgb=raw_rgb,
                    sample_id=idx,
                    data_path=meta_data['rgb'],
                    inv_depth=inv_depth,
                    normal=curr_normal,
                    )
        return data


    # def get_data_for_trainval(self, idx: int):
    #     anno = self.annotations['files'][idx]
    #     meta_data = self.load_meta_data(anno)
        
    #     # curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
    #     # curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
    #     # curr_sem_path = os.path.join(self.sem_root, meta_data['sem']) if self.sem_root is not None and ('sem' in meta_data) and (meta_data['sem'] is not None)  else None
    #     # curr_depth_mask_path = os.path.join(self.depth_mask_root, meta_data['depth_mask']) if self.depth_mask_root is not None and ('depth_mask' in meta_data) and (meta_data['depth_mask'] is not None)  else None
    #     data_path = self.load_data_path(meta_data)
    #     data_batch = self.load_batch(meta_data, data_path)

    #     curr_rgb, curr_depth, curr_normal, curr_sem, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_sem'], data_batch['curr_cam_model']

    #     # load data
    #     # curr_intrinsic = meta_data['cam_in']
    #     # curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        
    #     # # mask the depth
    #     # curr_depth = curr_depth.squeeze()
    #     # depth_mask = self.load_depth_valid_mask(curr_depth_mask_path, curr_depth)
    #     # curr_depth[~depth_mask] = -1
        
        
    #     # # get semantic labels
    #     # curr_sem = self.load_sem_label(curr_sem_path, curr_depth)
    #     # # create camera model
    #     # curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)        
        
    #     # get crop size
    #     transform_paras = dict(random_crop_size = self.random_crop_size)
    #     rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
    #                                                                images=[curr_rgb, ], 
    #                                                                labels=[curr_depth, ], 
    #                                                                intrinsics=[curr_intrinsic,], 
    #                                                                cam_models=[curr_cam_model, ],
    #                                                                other_labels=[curr_sem, ],
    #                                                                transform_paras=transform_paras)
    #     # process sky masks
    #     sem_mask = other_labels[0].int()
        
    #     # clip depth map 
    #     depth_out = self.normalize_depth(depths[0])
    #     # set the depth in sky region to the maximum depth
    #     depth_out[sem_mask==142] = -1 #self.depth_normalize[1] - 1e-6
    #     filename = os.path.basename(meta_data['rgb'])
    #     curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
    #     cam_models_stacks = [
    #         torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
    #         for i in [2, 4, 8, 16, 32] 
    #         ]
    #     pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]        
    #     data = dict(input=rgbs[0],
    #                 target=depth_out,
    #                 intrinsic=curr_intrinsic_mat,
    #                 filename=filename,
    #                 dataset=self.data_name,
    #                 cam_model=cam_models_stacks,
    #                 #ref_input=rgbs[1:],
    #                 # tmpl_flg=tmpl_annos['w_tmpl'],
    #                 pad=torch.tensor(pad),
    #                 data_type=[self.data_type, ],
    #                 sem_mask=sem_mask.int())
    #     return data
    
    # def get_data_for_test(self, idx: int):
    #     anno = self.annotations['files'][idx]
    #     meta_data = self.load_meta_data(anno)
    #     curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
    #     curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
    #     curr_depth_mask_path = os.path.join(self.depth_mask_root, meta_data['depth_mask']) if self.depth_mask_root is not None and ('depth_mask' in meta_data) and (meta_data['depth_mask'] is not None)  else None

    #     # load data
    #     ori_curr_intrinsic = meta_data['cam_in']
    #     curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        
    #     # mask the depth
    #     curr_depth = curr_depth.squeeze()
    #     depth_mask = self.load_depth_valid_mask(curr_depth_mask_path, curr_depth)
    #     curr_depth[~depth_mask] = -1
        
    #     ori_h, ori_w, _ = curr_rgb.shape
    #     # create camera model
    #     curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)

    #     # get crop size
    #     transform_paras = dict()
    #     rgbs, depths, intrinsics, cam_models, _,  other_labels, transform_paras = self.img_transforms(
    #                                                                images=[curr_rgb,],  #+ tmpl_rgbs, 
    #                                                                labels=[curr_depth, ], 
    #                                                                intrinsics=[ori_curr_intrinsic, ], # * (len(tmpl_rgbs) + 1), 
    #                                                                cam_models=[curr_cam_model, ],
    #                                                                transform_paras=transform_paras)
    #     # depth in original size and orignial metric***
    #     depth_out = self.clip_depth(curr_depth) * self.depth_range[1] # self.clip_depth(depths[0]) #

    #     filename = os.path.basename(meta_data['rgb'])
    #     curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])

    #     pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
    #     scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
    #     cam_models_stacks = [
    #         torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
    #         for i in [2, 4, 8, 16, 32] 
    #         ]    
    #     raw_rgb = torch.from_numpy(curr_rgb)
    #     # rel_pose = torch.from_numpy(tmpl_annos['tmpl_pose_list'][0])

    #     data = dict(input=rgbs[0],
    #                 target=depth_out,
    #                 intrinsic=curr_intrinsic_mat,
    #                 filename=filename,
    #                 dataset=self.data_name,
    #                 cam_model=cam_models_stacks,
    #                 pad=pad,
    #                 scale=scale_ratio,
    #                 raw_rgb=raw_rgb,
    #                 sample_id=idx,
    #                 data_path=meta_data['rgb'],
    #                 )
    #     return data
    

    def load_batch(self, meta_data, data_path):
        curr_intrinsic = meta_data['cam_in']
        # load rgb/depth
        curr_rgb, curr_depth = self.load_rgb_depth(data_path['rgb_path'], data_path['depth_path'])
        # get semantic labels
        curr_sem = self.load_sem_label(data_path['sem_path'], curr_depth)
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)       
        # get normal labels
        
        try:
            curr_normal = self.load_norm_label(data_path['normal_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1], depth=curr_depth, K=curr_intrinsic) # !!! this is diff of BaseDataset
        except:
            curr_normal = np.zeros_like(curr_rgb)
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
        normal = np.load(norm_path)
        normal[:,:,1:] *= -1
        normal = self.align_normal(normal, depth, K, H, W)

        return normal


    def process_depth(self, depth, rgb):
        depth[depth>150] = 0
        depth[depth<0.1] = 0
        depth /= self.metric_scale
        return depth

    def align_normal(self, normal, depth, K, H, W):
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


if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = DIODEDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    