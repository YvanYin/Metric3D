import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
import copy
from .__base_dataset__ import BaseDataset
import mono.utils.transform as img_transform


class AnyDataset(BaseDataset):
    def __init__(self, cfg, phase, **kwargs):        
        super(AnyDataset, self).__init__(
            cfg=cfg,
            phase=phase,
            **kwargs)

        self.cfg = cfg
        self.phase = phase
        self.mldb_info = kwargs['mldb_info']
        
        # root dir for data
        self.data_root = os.path.join(self.mldb_info['mldb_root'], self.mldb_info['data_root'])
        # depth/disp data root
        disp_root = self.mldb_info['disp_root'] if 'disp_root' in self.mldb_info else None
        self.disp_root = os.path.join(self.mldb_info['mldb_root'], disp_root) if disp_root is not None else None
        depth_root = self.mldb_info['depth_root'] if 'depth_root' in self.mldb_info else None
        self.depth_root = os.path.join(self.mldb_info['mldb_root'], depth_root) if depth_root is not None \
            else self.data_root
        # meta data root
        meta_data_root = self.mldb_info['meta_data_root'] if 'meta_data_root' in self.mldb_info else None
        self.meta_data_root = os.path.join(self.mldb_info['mldb_root'], meta_data_root) if meta_data_root is not None \
            else None
        # semantic segmentation labels root
        sem_root = self.mldb_info['semantic_root'] if 'semantic_root' in self.mldb_info else None
        self.sem_root = os.path.join(self.mldb_info['mldb_root'], sem_root) if sem_root is not None \
            else None

        # data annotations path
        self.data_annos_path = '/yvan1/data/NuScenes/NuScenes/annotations/train_ring_annotations.json'  # fill this 

        # load annotations
        annotations = self.load_annotations()
        whole_data_size = len(annotations['files']) 
               
        cfg_sample_ratio = cfg.data[phase].sample_ratio 
        cfg_sample_size = int(cfg.data[phase].sample_size)
        self.sample_size = int(whole_data_size * cfg_sample_ratio) if cfg_sample_size == -1 \
                           else (cfg_sample_size if cfg_sample_size < whole_data_size else whole_data_size)
        sample_list_of_whole_data = list(range(whole_data_size))[:self.sample_size]
        self.data_size = self.sample_size
        sample_list_of_whole_data = random.sample(list(range(whole_data_size)), whole_data_size)
        self.annotations = {'files': [annotations['files'][i] for i in sample_list_of_whole_data]}
        self.sample_list = list(range(self.data_size))
        
        # config transforms for the input and label
        self.transforms_cfg = cfg.data[phase]['pipeline']
        self.transforms_lib = 'mono.utils.transform.'

        self.img_file_type = ['.png', '.jpg', '.jpeg', '.bmp', '.tif']
        self.np_file_type = ['.npz', '.npy']

        # update canonical sparce information
        self.data_basic = copy.deepcopy(kwargs)
        canonical = self.data_basic.pop('canonical_space')
        self.data_basic.update(canonical)
        self.depth_range = kwargs['depth_range'] # predefined depth range for the network
        self.clip_depth_range = kwargs['clip_depth_range'] # predefined depth range for data processing
        self.depth_normalize = kwargs['depth_normalize']
        
        self.img_transforms = img_transform.Compose(self.build_data_transforms())
        self.EPS = 1e-8

        self.tmpl_info = ['rgb_sr', 'rgb_pre', 'rgb_next']

        # dataset info
        self.data_name = cfg.data_name
        self.data_type = cfg.data_type # there are mainly four types, i.e. ['rel', 'sfm', 'stereo', 'lidar']

    def __getitem__(self, idx: int) -> dict:
        return self.get_data_for_test(idx)

    def get_data_for_test(self, idx: int):
        # basic info
        anno = self.annotations['files'][idx]
        curr_rgb_path = os.path.join(self.data_root, anno['CAM_FRONT_RIGHT']['rgb']) # Lyft: CAM_FRONT_LEFT
        curr_depth_path = os.path.join(self.depth_root, anno['CAM_FRONT_RIGHT']['depth'])
        meta_data = self.load_meta_data(anno['CAM_FRONT_RIGHT'])
        ori_curr_intrinsic = meta_data['cam_in']
        
        curr_rgb, curr_depth = self.load_rgb_depth(curr_rgb_path, curr_depth_path)
        ori_h, ori_w, _ = curr_rgb.shape
        # create camera model
        curr_cam_model = self.create_cam_model(curr_rgb.shape[0], curr_rgb.shape[1], ori_curr_intrinsic)
        # load tmpl rgb info
        # tmpl_annos = self.load_tmpl_annos(anno, curr_rgb, meta_data)
        # tmpl_rgb = tmpl_annos['tmpl_rgb_list'] # list of reference rgbs

        transform_paras = dict()
        rgbs, depths, intrinsics, cam_models, other_labels, transform_paras = self.img_transforms(
                                                                   images=[curr_rgb, ], 
                                                                   labels=[curr_depth, ], 
                                                                   intrinsics=[ori_curr_intrinsic,], 
                                                                   cam_models=[curr_cam_model, ],
                                                                   transform_paras=transform_paras)
        # depth in augmented size
        # depth_out = self.clip_depth(depths[0])
        # depth in original size
        #depth_out = self.clip_depth(curr_depth)
        depth_out = curr_depth

        filename = os.path.basename(curr_rgb_path)
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
                    # ref_input=rgbs[1:],
                    # tmpl_flg=tmpl_annos['w_tmpl'],
                    pad=pad,
                    scale=scale_ratio,
                    raw_rgb=raw_rgb) 
        return data


    def process_depth(self, depth):
        depth[depth>65500] = 0
        depth /= 200.0
        return depth



if __name__ == '__main__':
    from mmcv.utils import Config 
    cfg = Config.fromfile('mono/configs/Apolloscape_DDAD/convnext_base.cascade.1m.sgd.mae.py')
    dataset_i = ApolloscapeDataset(cfg['Apolloscape'], 'train', **cfg.data_basic)
    print(dataset_i)
    