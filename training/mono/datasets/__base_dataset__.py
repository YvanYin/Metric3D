import os
import json
import torch
import torchvision.transforms as transforms
import os.path
import numpy as np
import cv2
from torch.utils.data import Dataset
import random
import mono.utils.transform as img_transform
import copy 
from mono.utils.comm import get_func
import pickle
import logging
import multiprocessing as mp
import ctypes
"""
Dataset annotations are saved in a Json file. All data, including rgb, depth, pose, and so on, captured within the same frame are saved in the same dict.
All frames are organized in a list. In each frame, it may contains the some or all of following data format. 

# Annotations for the current central RGB/depth cameras.

'rgb':          rgb image in the current frame.
'depth':        depth map in the current frame.
'sem':          semantic mask in the current frame.
'cam_in':       camera intrinsic parameters of the current rgb camera. 
'cam_ex':       camera extrinsic parameters of the current rgb camera.
'cam_ex_path':  path to the extrinsic parameters.
'pose':         pose in current frame.
'timestamp_rgb':    time stamp of current rgb image.

# Annotations for the left hand RGB/depth cameras.

'rgb_l':          rgb image of the left hand camera in the current frame.
'depth_l':        depth map of the left hand camera in the current frame.
'sem_l':          semantic mask of the left hand camera in the current frame.
'cam_in_l':       camera intrinsic parameters of the left hand rgb camera in the current frame.
'cam_ex_l':       camera extrinsic parameters of the left hand rgb camera in the current frame.
'cam_ex_path':    path to the extrinsic parameters.
'pose_l':         pose of the left hand camera  in the incurrent frame.
'timestamp_rgb_l':    time stamp of the rgb img captured by the left hand camera.

# Annotations for the right RGB/depth cameras, which is on the left hand of the current central cameras.

'rgb_r':          rgb image of the right hand camera in the current frame.
'depth_r':        depth map of the right hand camera in the current frame.
'sem_r':          semantic mask of the right hand camera in the current frame.
'cam_in_r':       camera intrinsic parameters of the right hand rgb camera in the current frame.
'cam_ex_r':       camera extrinsic parameters of the right hand rgb camera in the current frame.
'cam_ex_path_r':  path to the extrinsic parameters.
'pose_r':         pose of the right hand camera  in the incurrent frame.
'timestamp_rgb_r':    time stamp of the rgb img captured by the right hand camera.

# Annotations for the central RGB/depth cameras in the last frame.

'rgb_pre':          rgb image of the central camera in the last frame.
'depth_pre':        depth map of the central camera in the last frame.
'sem_pre':          semantic mask of the central camera in the last frame.
'cam_in_pre':       camera intrinsic parameters of the central rgb camera in the last frame.
'cam_ex_pre':       camera extrinsic parameters of the central rgb camera in the last frame.
'cam_ex_path_pre':  path to the extrinsic parameters.
'pose_pre':         pose of the central camera  in the last frame.
'timestamp_rgb_pre':    time stamp of the rgb img captured by the central camera.

# Annotations for the central RGB/depth cameras in the next frame.

'rgb_next':          rgb image of the central camera in the next frame.
'depth_next':        depth map of the central camera in the next frame.
'sem_next':          semantic mask of the central camera in the next frame.
'cam_in_next':       camera intrinsic parameters of the central rgb camera in the next frame.
'cam_ex_next':       camera extrinsic parameters of the central rgb camera in the next frame.
'cam_ex_path_next':  path to the extrinsic parameters.
'pose_next':         pose of the central camera  in the next frame.
'timestamp_rgb_next':    time stamp of the rgb img captured by the central camera.
"""

class BaseDataset(Dataset):
    def __init__(self, cfg, phase, **kwargs):
        super(BaseDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.db_info = kwargs['db_info']
        
        # root dir for data
        self.data_root = os.path.join(self.db_info['db_root'], self.db_info['data_root']) 
        # depth/disp data root
        disp_root = self.db_info['disp_root'] if 'disp_root' in self.db_info else None
        self.disp_root = os.path.join(self.db_info['db_root'], disp_root) if disp_root is not None else None
        depth_root = self.db_info['depth_root'] if 'depth_root' in self.db_info else None
        self.depth_root = os.path.join(self.db_info['db_root'], depth_root) if depth_root is not None \
            else self.data_root
        # meta data root
        meta_data_root = self.db_info['meta_data_root'] if 'meta_data_root' in self.db_info else None
        self.meta_data_root = os.path.join(self.db_info['db_root'], meta_data_root) if meta_data_root is not None \
            else None
        # semantic segmentation labels root
        sem_root = self.db_info['semantic_root'] if 'semantic_root' in self.db_info else None
        self.sem_root = os.path.join(self.db_info['db_root'], sem_root) if sem_root is not None \
            else None
        # depth valid mask labels root
        depth_mask_root = self.db_info['depth_mask_root'] if 'depth_mask_root' in self.db_info else None
        self.depth_mask_root = os.path.join(self.db_info['db_root'], depth_mask_root) if depth_mask_root is not None \
            else None
        # surface normal labels root
        norm_root = self.db_info['normal_root'] if 'normal_root' in self.db_info else None
        self.norm_root = os.path.join(self.db_info['db_root'], norm_root) if norm_root is not None \
            else None
        # data annotations path
        self.data_annos_path = os.path.join(self.db_info['db_root'], self.db_info['%s_annotations_path' % phase])

        # load annotations
        self.data_info = self.load_annotations()
        whole_data_size = len(self.data_info['files']) 
        
        # sample a subset for training/validation/testing
        # such method is deprecated, each training may get different sample list
        
        cfg_sample_ratio = cfg.data[phase].sample_ratio 
        cfg_sample_size = int(cfg.data[phase].sample_size)
        self.sample_size = int(whole_data_size * cfg_sample_ratio) if cfg_sample_size == -1 \
                           else (cfg_sample_size if cfg_sample_size < whole_data_size else whole_data_size)
        random.seed(100) # set the random seed
        sample_list_of_whole_data = random.sample(list(range(whole_data_size)), self.sample_size)

        self.data_size = self.sample_size
        self.annotations = {'files': [self.data_info['files'][i] for i in sample_list_of_whole_data]}
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
        self.disp_scale = 10.0
        self.depth_range = kwargs['depth_range'] # predefined depth range for the network
        self.clip_depth_range = kwargs['clip_depth_range'] # predefined depth range for data processing
        self.depth_normalize = kwargs['depth_normalize']
        
        self.img_transforms = img_transform.Compose(self.build_data_transforms())
        self.EPS = 1e-6

        # self.tmpl_info = ['rgb_sr', 'rgb_pre', 'rgb_next']
        # self.tgt2ref_pose_lookup = {'rgb_sr': 'cam_ex', 'rgb_pre': 'pose_pre', 'rgb_next': 'pose_next'}

        # dataset info
        self.data_name = cfg.data_name
        self.data_type = cfg.data_type # there are mainly four types, i.e. ['rel', 'sfm', 'stereo', 'lidar']
        self.logger = logging.getLogger()
        self.logger.info(f'{self.data_name} in {self.phase} whole data size: {whole_data_size}')
        
        # random crop size for training
        crop_size = kwargs['crop_size']
        shared_array_base = mp.Array(ctypes.c_int32, 2)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array[0] = crop_size[0]
        shared_array[1] = crop_size[1]
        # self.random_crop_size = torch.from_numpy(np.array([0,0])) #torch.from_numpy(shared_array)
        self.random_crop_size = torch.from_numpy(shared_array)

        
    def __name__(self):
        return self.data_name

    def __len__(self):
        return self.data_size

    def load_annotations(self):
        if not os.path.exists(self.data_annos_path):
            raise RuntimeError(f'Cannot find {self.data_annos_path} annotations.')

        with open(self.data_annos_path, 'r')  as f:
            annos = json.load(f)
        return annos
    
    def build_data_transforms(self):
        transforms_list = []
        for transform in self.transforms_cfg:
            args = copy.deepcopy(transform)
            # insert the canonical space configs
            args.update(self.data_basic)
      
            obj_name = args.pop('type')
            obj_path = self.transforms_lib + obj_name
            obj_cls = get_func(obj_path)
            
            obj = obj_cls(**args)
            transforms_list.append(obj)
        return transforms_list
    
        
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
        elif data_type in self.np_file_type:
            data = np.load(path)
        else:
            raise RuntimeError(f'{data_type} is not supported in current version.')
        
        try:
            return data.squeeze()
        except:
            temp = 1
            raise RuntimeError(f'{path} is not successfully loaded.')
    
    def __getitem__(self, idx: int) -> dict:
        if self.phase == 'test':
            return self.get_data_for_test(idx)
        else:
            return self.get_data_for_trainval(idx)

    def get_data_for_trainval(self, idx: int):
        anno = self.annotations['files'][idx]
        meta_data = self.load_meta_data(anno)
        
        data_path = self.load_data_path(meta_data)
        data_batch = self.load_batch(meta_data, data_path)
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
        if 'label_scale_factor' not in transform_paras.keys():
            transform_paras['label_scale_factor'] = 1
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
    
    def load_data_path(self, meta_data):
        curr_rgb_path = os.path.join(self.data_root, meta_data['rgb'])
        curr_depth_path = os.path.join(self.depth_root, meta_data['depth'])
        curr_sem_path = os.path.join(self.sem_root, meta_data['sem']) \
            if self.sem_root is not None and ('sem' in meta_data) and (meta_data['sem'] is not None)  \
            else None
        # matterport3d separates xyz into three images
        if ('normal' in meta_data) and (meta_data['normal'] is not None) and (self.norm_root is not None):
            if isinstance(meta_data['normal'], dict):
                curr_norm_path = {}
                for k,v in meta_data['normal'].items():
                    curr_norm_path[k] = os.path.join(self.norm_root, v)
            else:
                curr_norm_path = os.path.join(self.norm_root, meta_data['normal'])
        else:
            curr_norm_path = None
        curr_depth_mask_path = os.path.join(self.depth_mask_root, meta_data['depth_mask']) \
            if self.depth_mask_root is not None and ('depth_mask' in meta_data) and (meta_data['depth_mask'] is not None)  \
            else None

        if ('disp' in meta_data) and (meta_data['disp'] is not None) and (self.disp_root is not None):
            if isinstance(meta_data['disp'], dict):
                curr_disp_path = {}
                for k,v in meta_data['disp'].items():
                    curr_disp_path[k] = os.path.join(self.disp_root, v)
            else:
                curr_disp_path = os.path.join(self.disp_root, meta_data['disp'])
        else:
            curr_disp_path = None

        data_path=dict(
            rgb_path=curr_rgb_path,
            depth_path=curr_depth_path,
            sem_path=curr_sem_path,
            normal_path=curr_norm_path,
            disp_path=curr_disp_path,
            depth_mask_path=curr_depth_mask_path,
            )
        return data_path
    
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


    def clip_depth(self, depth: np.array) -> np.array:
        depth[(depth>self.clip_depth_range[1]) | (depth<self.clip_depth_range[0])] = -1
        depth /= self.depth_range[1]
        depth[depth<self.EPS] = -1
        return depth
    
    def normalize_depth(self, depth: np.array) -> np.array:
        depth /= self.depth_range[1]
        depth[depth<self.EPS] = -1
        return depth
    
    def process_depth(self, depth: np.array, rgb:np.array=None):
        return depth
    
    def create_cam_model(self, H : int, W : int, intrinsics : list) -> np.array:
        """
        Encode the camera model (focal length and principle point) to a 4-channel map. 
        """
        fx, fy, u0, v0 = intrinsics
        f = (fx + fy) / 2.0
        # principle point location
        x_row = np.arange(0, W).astype(np.float32)
        x_row_center_norm = (x_row - u0) / W
        x_center = np.tile(x_row_center_norm, (H, 1)) # [H, W]

        y_col = np.arange(0, H).astype(np.float32) 
        y_col_center_norm = (y_col - v0) / H
        y_center = np.tile(y_col_center_norm, (W, 1)).T

        # FoV
        fov_x = np.arctan(x_center / (f / W))
        fov_y =  np.arctan(y_center/ (f / H))

        cam_model = np.stack([x_center, y_center, fov_x, fov_y], axis=2)
        return cam_model
    
    def check_data(self, data_dict : dict):
        for k, v in data_dict.items():
            if v is None:
                # print(f'{self.data_name}, {k} cannot be read!')
                self.logger.info(f'{self.data_name}, {k} cannot be read!')
    
    def intrinsics_list2mat(self, intrinsics: torch.tensor) -> torch.tensor:
        """
        Create camera intrinsic matrix.
        Args:
            intrinsics (torch.tensor, [4,]): list of camera intrinsic parameters.
        returns:
            intrinsics_mat (torch.tensor, [3x3]): camera intrinsic parameters matrix.
        """
        intrinsics_mat = torch.zeros((3,3)).float()
        intrinsics_mat[0, 0] = intrinsics[0]
        intrinsics_mat[1, 1] = intrinsics[1]
        intrinsics_mat[0, 2] = intrinsics[2]
        intrinsics_mat[1, 2] = intrinsics[3]
        intrinsics_mat[2, 2] = 1.0
        return intrinsics_mat
        
    # def load_tmpl_image(self, curr_rgb: np.array, meta_data: dict) -> dict:
    #     """
    #     Load  consecutive RGB frames.
    #     Args:
    #         anno: the annotation for this group.
    #         curr_rgb: rgb image of the current frame.
    #         meta_data: meta data information.
    #     Returns:
    #         tmpl_annos: temporal rgbs.
    #     """
    #     w_tmpl = False
        
    #     tmpl_list = []
    #     # organize temporal annotations 
    #     for i in self.tmpl_info:
    #         if (i in meta_data) and (meta_data[i] is not None) and os.path.exists(os.path.join(self.data_root, meta_data[i])):
    #             tmpl_list.append(os.path.join(self.data_root, meta_data[i]))

    #     if len(tmpl_list) == 0:
    #         rgb_tmpl = curr_rgb.copy()
    #     else:
    #         id = np.random.randint(len(tmpl_list))
    #         rgb_tmpl = self.load_data(tmpl_list[id], is_rgb_img=True)
    #         w_tmpl = True

    #     tmpl_annos = dict(
    #         tmpl_rgb_list = [rgb_tmpl,],
    #         w_tmpl = w_tmpl
    #     )
    #     return tmpl_annos

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
        return meta_data
    
    def load_rgb_depth(self, rgb_path: str, depth_path: str):
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
        
        depth  = self.process_depth(depth, rgb)
        return rgb, depth
    
    def load_sem_label(self, sem_path, depth=None, sky_id=142) -> np.array:
        H, W = depth.shape
        # if sem_path is not None:
        #     print(self.data_name)
        sem_label = cv2.imread(sem_path, 0) if sem_path is not None \
            else np.ones((H, W), dtype=np.int) * -1
        if sem_label is None:
            sem_label = np.ones((H, W), dtype=np.int) * -1
        # set dtype to int before 
        sem_label = sem_label.astype(np.int) 
        sem_label[sem_label==255] = -1
        
        # mask invalid sky region
        mask_depth_valid = depth > 1e-8
        invalid_sky_region = (sem_label==142) & (mask_depth_valid)
        if self.data_type in ['lidar', 'sfm', 'denselidar', 'denselidar_nometric']:
            sem_label[invalid_sky_region] = -1
        return sem_label
    
    def load_depth_valid_mask(self, depth_mask_path, depth=None) -> np.array:
        if depth_mask_path is None:
            return np.ones_like(depth, dtype=np.bool)
        data_type = os.path.splitext(depth_mask_path)[-1]
        if data_type in self.img_file_type:
            data = cv2.imread(depth_mask_path, -1)
        elif data_type in self.np_file_type:
            data = np.load(depth_mask_path)
        else:
            raise RuntimeError(f'{data_type} is not supported in current version.')
        data = data.astype(np.bool)
        return data
        
    def load_norm_label(self, norm_path, H, W):
        norm_gt = np.zeros((H, W, 3)).astype(np.float32)
        return norm_gt

    def load_stereo_depth_label(self, disp_path, H, W):
        stereo_depth_gt = np.zeros((H, W, 1)).astype(np.float32)
        return stereo_depth_gt

    def depth2invdepth(self, depth, sky_mask):
        inv_depth = 1.0 / depth * self.disp_scale
        inv_depth[depth<1e-6] = -1.0
        inv_depth[inv_depth < 0] = -1.0
        inv_depth[sky_mask] = 0
        return inv_depth


    def set_random_crop_size(self, random_crop_size):
        self.random_crop_size[0] = random_crop_size[0]
        self.random_crop_size[1] = random_crop_size[1]
