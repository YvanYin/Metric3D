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


class ScanNetDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(ScanNetDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
    
    # def get_data_for_test(self, idx):
    #     anno = self.annotations['files'][idx]
    #     curr_rgb_path = os.path.join(self.data_root, anno['rgb'])
    #     curr_depth_path = os.path.join(self.depth_root, anno['depth'])
    #     meta_data = self.load_meta_data(anno)
    #     ori_curr_intrinsic = meta_data['cam_in']
        
    #     curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
    #     # curr_rgb = cv2.resize(curr_rgb, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
    #     ori_h, ori_w, _ = curr_rgb.shape
    #     # create camera model
    #     curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)
    #     # load tmpl rgb info
    #     # tmpl_annos = self.load_tmpl_annos(anno, curr_rgb, meta_data)
    #     # tmpl_rgb = tmpl_annos['tmpl_rgb_list'] # list of reference rgbs

    #     transform_paras = dict()
    #     rgbs, depths, intrinsics, cam_models, _,  other_labels, transform_paras = self.img_transforms(
    #                                                                images=[curr_rgb, ], 
    #                                                                labels=[curr_depth, ], 
    #                                                                intrinsics=[ori_curr_intrinsic,], 
    #                                                                cam_models=[curr_cam_model, ],
    #                                                                transform_paras=transform_paras)
    #     # depth in original size
    #     depth_out = self.clip_depth(curr_depth) * self.depth_range[1]

    #     filename = os.path.basename(anno['rgb'])
    #     curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])

    #     pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]
    #     scale_ratio = transform_paras['label_scale_factor'] if 'label_scale_factor' in transform_paras else 1.0
    #     cam_models_stacks = [
    #         torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
    #         for i in [2, 4, 8, 16, 32] 
    #         ]    
    #     raw_rgb = torch.from_numpy(curr_rgb)
    #     data = dict(input=rgbs[0],
    #                 target=depth_out,
    #                 intrinsic=curr_intrinsic_mat,
    #                 filename=filename,
    #                 dataset=self.data_name,
    #                 cam_model=cam_models_stacks,
    #                 ref_input=rgbs[1:],
    #                 tmpl_flg=False,
    #                 pad=pad,
    #                 scale=scale_ratio,
    #                 raw_rgb=raw_rgb,
    #                 normal =np.zeros_like(curr_rgb.transpose((2,0,1))),
    #     ) 
    #     return data

    def get_data_for_test(self, idx: int, test_mode=True):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path, test_mode)
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

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path, test_mode=False)

        # if data_path['sem_path'] is not None:
        #     print(self.data_name)

        curr_rgb, curr_depth, curr_normal, curr_sem, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_sem'], data_batch['curr_cam_model']
        #curr_stereo_depth = data_batch['curr_stereo_depth']
        
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

    def load_batch(self, meta_data, data_path, test_mode):

        # print('############')
        # print(data_path['rgb_path'])
        # print(data_path['normal_path'])
        # print('############')

        curr_intrinsic = meta_data['cam_in']
        # load rgb/depth
        curr_rgb, curr_depth = self.load_rgb_depth(data_path['rgb_path'], data_path['depth_path'], test_mode)
        # get semantic labels
        curr_sem = self.load_sem_label(data_path['sem_path'], curr_depth)
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], curr_intrinsic)       
        # get normal labels
        curr_normal = self.load_norm_label(data_path['normal_path'], H=curr_rgb.shape[0], W=curr_rgb.shape[1], test_mode=test_mode) 
        # get depth mask
        depth_mask = self.load_depth_valid_mask(data_path['depth_mask_path'])
        curr_depth[~depth_mask] = -1
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

    def load_rgb_depth(self, rgb_path: str, depth_path: str, test_mode: bool):
        """
        Load the rgb and depth map with the paths.
        """
        rgb = self.load_data(rgb_path, is_rgb_img=True)
        if rgb is None:
            self.logger.info(f'>>>>{rgb_path} has errors.')
       
        depth = self.load_data(depth_path)
        if depth is None:
            self.logger.info(f'{depth_path} has errors.')
        
        # self.check_data(dict(
        #     rgb_path=rgb,
        #     depth_path=depth,
        # ))
        depth = depth.astype(np.float)
        # if depth.shape != rgb.shape[:2]:
        #     print(f'no-equal in {self.data_name}')
        #     depth = cv2.resize(depth, rgb.shape[::-1][1:])
        
        depth  = self.process_depth(depth, rgb, test_mode)
        return rgb, depth
    
    def process_depth(self, depth, rgb, test_mode=False):
        depth[depth>65500] = 0
        depth /= self.metric_scale
        h, w, _ = rgb.shape # to rgb size
        if test_mode==False:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth

    def load_norm_label(self, norm_path, H, W, test_mode):
        
        if norm_path is None:
            norm_gt = np.zeros((H, W, 3)).astype(np.float32)
        else:
            norm_gt = cv2.imread(norm_path)
            norm_gt = cv2.cvtColor(norm_gt, cv2.COLOR_BGR2RGB) 

            norm_gt = np.array(norm_gt).astype(np.uint8)

            mask_path = 'orient-mask'.join(norm_path.rsplit('normal', 1))
            mask_gt = cv2.imread(mask_path)
            mask_gt = np.array(mask_gt).astype(np.uint8)
            valid_mask = np.logical_not(
                np.logical_and(
                    np.logical_and(
                        mask_gt[:, :, 0] == 0, mask_gt[:, :, 1] == 0),
                    mask_gt[:, :, 2] == 0))
            valid_mask = valid_mask[:, :, np.newaxis]

            # norm_valid_mask = np.logical_not(
            #     np.logical_and(
            #         np.logical_and(
            #             norm_gt[:, :, 0] == 0, norm_gt[:, :, 1] == 0),
            #         norm_gt[:, :, 2] == 0))
            # norm_valid_mask = norm_valid_mask[:, :, np.newaxis]

            norm_gt = ((norm_gt.astype(np.float32) / 255.0) * 2.0) - 1.0
            norm_valid_mask = (np.linalg.norm(norm_gt, axis=2, keepdims=True) > 0.5) * valid_mask
            norm_gt = norm_gt * norm_valid_mask

            if test_mode==False:
                norm_gt = cv2.resize(norm_gt, (W, H), interpolation=cv2.INTER_NEAREST)
            
        return norm_gt   



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = NYUDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    