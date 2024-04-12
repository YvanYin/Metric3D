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
import pickle


class TaskonomyDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):
        super(TaskonomyDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)
        self.metric_scale = cfg.metric_scale
        #self.cap_range = self.depth_range # in meter
    
    def __getitem__(self, idx: int) -> dict:
        if self.phase == 'test':
            return self.get_data_for_test(idx)
        else:
            return self.get_data_for_trainval(idx)

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
        u0, v0, fx, fy = meta_data['cam_in']
        meta_data['cam_in'] = [fx, fy, u0, v0] # fix data bugs
        return meta_data

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
        curr_rgb, curr_depth, curr_normal, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_cam_model']
        curr_intrinsic = meta_data['cam_in']

        ins_planes_path = os.path.join(self.data_root, meta_data['ins_planes']) if ('ins_planes' in meta_data) and (meta_data['ins_planes'] is not None) else None
        # get instance planes
        ins_planes = self.load_ins_planes(curr_depth, ins_planes_path)

        # load data
        # u0, v0, fx, fy = meta_data['cam_in'] # this is 
        # ori_curr_intrinsic = [fx, fy, u0, v0]        
        # curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        
        # get crop size
        # transform_paras = dict()
        transform_paras = dict(random_crop_size = self.random_crop_size)
        rgbs, depths, intrinsics, cam_models, normals, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb, ], 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[curr_intrinsic,], 
                                                                   cam_models=[curr_cam_model, ],
                                                                   normals = [curr_normal, ],
                                                                   other_labels=[ins_planes, ],
                                                                   transform_paras=transform_paras)
        # process instance planes
        ins_planes = other_labels[0].int()
        
        # clip depth map 
        depth_out = self.normalize_depth(depths[0])
        # get inverse depth
        inv_depth = self.depth2invdepth(depth_out, torch.zeros_like(depth_out, dtype=torch.bool))
        filename = os.path.basename(meta_data['rgb'])
        curr_intrinsic_mat = self.intrinsics_list2mat(intrinsics[0])
        cam_models_stacks = [
            torch.nn.functional.interpolate(cam_models[0][None, :, :, :], size=(cam_models[0].shape[1]//i, cam_models[0].shape[2]//i), mode='bilinear', align_corners=False).squeeze()
            for i in [2, 4, 8, 16, 32]
            ]
        pad = transform_paras['pad'] if 'pad' in transform_paras else [0,0,0,0]        
        data = dict(input=rgbs[0],
                    target=depth_out,
                    intrinsic=curr_intrinsic_mat,
                    filename=filename,
                    dataset=self.data_name,
                    cam_model=cam_models_stacks,
                    pad=torch.tensor(pad),
                    data_type=[self.data_type, ],
                    sem_mask=ins_planes,
                    normal=normals[0],
                    inv_depth=inv_depth,
                    stereo_depth=torch.zeros_like(inv_depth),
                    scale= transform_paras['label_scale_factor'])
        return data

    def get_data_for_test(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)

        curr_rgb, curr_depth, curr_normal, curr_cam_model = data_batch['curr_rgb'], data_batch['curr_depth'], data_batch['curr_normal'], data_batch['curr_cam_model']
        ori_curr_intrinsic = meta_data['cam_in']

        # curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
        # curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])

        # curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        # ori_h, ori_w, _ = curr_rgb.shape
        # # create camera model
        # curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)
        # load tmpl rgb info
        # tmpl_annos = self.load_tmpl_image_pose(curr_rgb, meta_data)
        # tmpl_rgbs = tmpl_annos['tmpl_rgb_list'] # list of reference rgbs

        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, _, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb,], #  + tmpl_rgbs, 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic, ], # * (len(tmpl_rgbs) + 1), 
                                                                   cam_models=[curr_cam_model, ],
                                                                   transform_paras=transform_paras)
        # depth in original size and orignial metric***
        depth_out = self.clip_depth(curr_depth) * self.depth_range[1]
        inv_depth = self.depth2invdepth(depth_out, np.zeros_like(depth_out, dtype=np.bool))

        filename = os.path.basename(meta_data['rgb'])
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
    
    def load_ins_planes(self, depth: np.array, ins_planes_path: str) -> np.array:
        if ins_planes_path is not None:
            ins_planes = cv2.imread(ins_planes_path, -1)
        else:
            ins_planes = np.zeros_like(depth)
        return ins_planes



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = ApolloscapeDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    