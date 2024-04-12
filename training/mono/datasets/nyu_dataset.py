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


class NYUDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(NYUDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        # if data_path['sem_path'] is not None:
        #     print(self.data_name)

        curr_rgb, curr_depth, curr_normal, curr_sem, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_sem'], data_batch['curr_cam_model']
        #curr_stereo_depth = data_batch['curr_stereo_depth']
        new_rgb = np.zeros_like(curr_rgb)
        new_rgb[6:-6, 6:-6, :] = curr_rgb[6:-6, 6:-6, :]
        curr_rgb = new_rgb

        # A patch for stereo depth dataloader (no need to modify specific datasets)
        if 'curr_stereo_depth' in data_batch.keys():
            curr_stereo_depth = data_batch['curr_stereo_depth']
        else:
            curr_stereo_depth = self.load_stereo_depth_label(None, H=curr_rgb.shape[0], W=curr_rgb.shape[1]) 

        curr_intrinsic = meta_data['cam_in']
        # data augmentation
        transform_paras = dict(random_crop_size = self.random_crop_size) # dict() 
        assert curr_rgb.shape[:2] == curr_depth.shape == curr_normal.shape[:2] == curr_sem.shape
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb, ], 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[curr_intrinsic,], 
                                                                   cam_models=[curr_cam_model, ],
                                                                   normals = [curr_normal, ],
                                                                   other_labels=[curr_sem, curr_stereo_depth],
                                                                   transform_paras=transform_paras)
        # process sky masks
        sem_mask = other_labels[0].int()
        # clip depth map 
        depth_out = self.normalize_depth(depths[0])
        # set the depth of sky region to the invalid
        depth_out[sem_mask==142] = -1 # self.depth_normalize[1] - 1e-6
        # get inverse depth
        inv_depth = self.depth2invdepth(depth_out, sem_mask==142)
        filename = os.path.basename(meta_data['rgb'])[:-4] + '.jpg'
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]

        # stereo_depth 
        stereo_depth_pre_trans = other_labels[1] * (other_labels[1] > 0.3) * (other_labels[1] < 200)
        stereo_depth = stereo_depth_pre_trans * transform_paras['label_scale_factor']
        stereo_depth = self.normalize_depth(stereo_depth)

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]        
        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=torch.tensor(pad),
                    data_type=[self.data_type, ],
                    sem_mask=sem_mask.int(),
                    stereo_depth= stereo_depth,
                    normal=normals[0],
                    inv_depth=inv_depth,
                    scale=transform_paras['label_scale_factor'])
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
        curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
        # load data
        ori_curr_intrinsic = meta_data['cam_in']
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        # crop rgb/depth
        new_rgb = np.zeros_like(curr_rgb)
        new_rgb[6:-6, 6:-6, :] = curr_rgb[6:-6, 6:-6, :]
        curr_rgb = new_rgb
        
        ori_h, ori_w, _ = curr_rgb.shape
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)

        if 'normal' in meta_data.keys():
            normal_path = os.path.join(self.data_root, meta_data['normal'])
        else:
            normal_path = None

        curr_normal = self.load_norm_label(normal_path, H=curr_rgb.shape[0], W=curr_rgb.shape[1]) 
        # load tmpl rgb info
        # tmpl_annos = self.load_tmpl_image_pose(curr_rgb, meta_data)
        # tmpl_rgbs = tmpl_annos['tmpl_rgb_list'] # list of reference rgbs

        # get crop size
        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb,],  #+ tmpl_rgbs, 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic, ], # * (len(tmpl_rgbs) + 1), 
                                                                   cam_models=[curr_cam_model, ],
                                                                   normals = [curr_normal, ],
                                                                   transform_paras=transform_paras)
        # depth in original size and orignial metric***
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1] # self.clip_depth(depths[0]) #
        
        filename = os.path.basename(meta_data['rgb'])
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])

        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
        scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32] 
            ]    
        raw_rgb = torch.from_numpy(curr_rgb)
        # rel_pose = torch.from_numpy(tmpl_annos['tmpl_pose_list'][0])
        curr_normal = torch.from_numpy(curr_normal.transpose((2,0,1)))

        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    # ref_input=rgbs[1:],
                    # tmpl_flg=tmpl_annos['w_tmpl'],
                    pad=pad,
                    scale=scale_ratio,
                    raw_rgb=raw_rgb,
                    # rel_pose=rel_pose,
                    normal=curr_normal
                    #normal=np.zeros_like(curr_rgb.transpose((2,0,1))),
                    )
        return data

    def load_norm_label(self, norm_path, H, W):
        if norm_path is None:
            norm_gt = np.zeros((H, W, 3)).astype(np.float32)
        else:
            norm_gt = cv2.imread(norm_path)

            norm_gt = np.array(norm_gt).astype(np.uint8)
            norm_valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
                    norm_gt[:, :, 2] == 0))
            norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm_gt = norm_gt * norm_valid_mask * -1
            
        return norm_gt   

    def process_depth(self, depth, rgb):
        # eign crop
        new_depth = np.zeros_like(depth)
        new_depth[45:471, 41:601] = depth[45:471, 41:601]
        
        new_depth[new_depth>65500] = 0
        new_depth /= self.metric_scale
        return new_depth




if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = NYUDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    