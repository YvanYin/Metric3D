import numpy as np
import logging
import torch.distributed as dist
import math
import os
from mono.utils.comm import get_func, main_process
from torch.utils.data import ConcatDataset, DataLoader
import random
import copy
import torch
import logging


def build_dataset_n_sampler_with_cfg(cfg, phase):
    # build data array, similar datasets are organized in the same group
    datasets_array = build_data_array(cfg, phase)
    # concatenate datasets with torch.utils.data.ConcatDataset methods
    dataset_merge = concatenate_datasets(datasets_array)
    # customerize sampler
    custom_sampler = CustomerMultiDataSampler(cfg, dataset_merge, phase)
    return dataset_merge, custom_sampler

class CustomerMultiDataSampler(torch.utils.data.Sampler):
    """
    Customerize a sampler method.  During this process, the size of some datasets will be tailored or expanded. 
    Such process aims to ensure each group has the same data size. 
    e.g. dataset_list: [[A, B, C], [E, F], M], then group 'A,B,C' (Size(A) + Size(B) + Size(C)) has the same size 
    as to group 'E,F' (Size(E) + Size(F)), so as to 'M'.
    args:
        @ cfg: configs for each dataset.
        @ dataset_merge: merged multiple datasets with the torch.utils.data.ConcatDataset method.
        @ phase: train/val/test phase.
    """

    def __init__(self, cfg, dataset_merge, phase):
        self.cfg = cfg
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.phase = phase
        self.global_rank = cfg.dist_params.global_rank
        self.dataset_merge = dataset_merge  
        self.logger = logging.getLogger()
        if main_process():
            self.logger.info(f'Initilized CustomerMultiDataSampler for {phase}.')
        self.random_seed = 136
        self.random_seed_cp = 639

    def __iter__(self):
        self.create_samplers() 
        self.logger.info("Sample list of {} in rank {} is: {}".format(self.phase, self.global_rank, ' '.join(map(str, self.sample_indices_array[-20: -10]))))
        # subsample, each rank sample a subset for training.
        rank_offset = self.each_gpu_size * self.global_rank
        rank_indices = self.sample_indices_array[rank_offset : rank_offset + self.each_gpu_size]
        
        assert rank_indices.size == self.each_gpu_size
        
        for id in rank_indices:
            yield id

    def __len__(self):
        return self.total_dist_size

    def create_samplers(self):       
        # sample idx for each dataset, idx value should not exceed the size of data, 
        # i.e. 0 <= idx < len(data_size)
        #self.samples_mat = []
        self.indices_mat = []
        # size expanded, idx cumulative aggregrated for calling
        self.indices_expand_mat = []
        
        # max group size, each group may consists of multiple similar datasets
        max_group_size = max([len(i) for i in self.dataset_merge.datasets])
        
        dataset_cumulative_sizes = [0] + self.dataset_merge.cumulative_sizes
        
        for gi, dataset_group in enumerate(self.dataset_merge.datasets):
            # the merged dataset consists of multiple grouped datasets
            samples_group = []
            indices_expand_group = []
            indices_group = []

            # to ensure each group has the same size, group with less data has to duplicate its sample list for 'cp_times' times
            cp_times = max_group_size / len(dataset_group)
            
            # adjust each group to ensure they have the same data size
            group_cumulative_sizes = [0] + dataset_group.cumulative_sizes
            expand_indices_sizes = (np.array(group_cumulative_sizes) * cp_times).astype(np.int)
            expand_indices_sizes[-1] = max_group_size
            # datasets in the same group have to expand its sample list
            expand_indices_sizes = expand_indices_sizes[1:] - expand_indices_sizes[:-1]
            
            for di, dataset_i in enumerate(dataset_group.datasets):
                # datasets residing in each group may have similar features
                # samples indices list
                dataset_i_ori_sample_list = self.dataset_merge.datasets[gi].datasets[di].sample_list
                if self.phase == 'train':
                    #sample_list_i = random.sample(dataset_i_ori_sample_list, len(dataset_i_ori_sample_list))
                    sample_list_i = dataset_i_ori_sample_list
                else:
                    # no shuffle in val or test
                    sample_list_i = dataset_i_ori_sample_list
                #samples_group.append(sample_list_i)
                          
                # expand the sample list for each dataset
                expand_size_i = expand_indices_sizes[di]
                indices_expand_list = copy.deepcopy(sample_list_i)
                
                for i in range(int(cp_times)-1):
                    #indices_expand_list += random.sample(sample_list_i, len(dataset_i))
                    indices_expand_list += sample_list_i
                random.seed(self.random_seed_cp)
                indices_expand_list += random.sample(sample_list_i, len(dataset_i))[:expand_size_i % len(dataset_i)]
                # adjust indices value
                indices_expand_list = np.array(indices_expand_list) + dataset_cumulative_sizes[gi] + group_cumulative_sizes[di]
                indices_list = np.array(sample_list_i) + dataset_cumulative_sizes[gi] + group_cumulative_sizes[di]

                # the expanded sample list for dataset_i
                indices_expand_group.append(indices_expand_list)
                # the original sample list for the dataset_i
                indices_group.append(indices_list)
                
                if main_process():
                    self.logger.info(f'"{dataset_i.data_name}", {self.phase} set in group {gi}: ' + 
                                     f'expand size {len(sample_list_i)} --->>>---, {expand_size_i}')

            concat_group = np.concatenate(indices_expand_group)
            # shuffle the grouped datasets samples, e.g. each group data is [a1, a2, a3, b1, b2, b3, b4, c1, c2], the shuffled one, maybe, is [a3, b1, b2, b3, b4, c1,...]
            np.random.seed(self.random_seed)
            if self.phase == 'train':
                np.random.shuffle(concat_group)
            self.indices_expand_mat.append(concat_group)
            self.indices_mat.append(np.concatenate(indices_group))
        
        # create sample list
        if "train" in self.phase:
            # data groups are cross sorted, i.e. [A, B, C, A, B, C....]
            self.sample_indices_array = np.array(self.indices_expand_mat).transpose(1, 0).reshape(-1)
            self.total_indices_size = max_group_size * len(self.dataset_merge.datasets)
        else:
            self.sample_indices_array = np.concatenate(self.indices_mat[:])
            self.total_indices_size = self.sample_indices_array.size
        
        self.total_sample_size = len(self.dataset_merge)
        self.each_gpu_size = int(np.ceil(self.total_indices_size * 1.0 / self.world_size)) # ignore some residual samples
        self.total_dist_size = self.each_gpu_size * self.world_size
        # add extra samples to make it evenly divisible
        diff_size = int(self.total_dist_size - self.total_indices_size)  # int(self.total_dist_size - self.total_sample_size)
        if diff_size > 0:
            self.sample_indices_array = np.append(self.sample_indices_array, self.sample_indices_array[:diff_size])
        #if main_process():
        self.logger.info(f'Expanded data size in merged dataset: {self.total_sample_size}, adjusted data size for distributed running: {self.total_dist_size}')
        self.random_seed += 413
        self.random_seed_cp += 377


def build_data_array(cfg, phase):
    """
    Construct data repo with cfg. In cfg, there is a data name array, which encloses the name of each data. 
    Each data name links to a data config file. With this config file, dataset can be constructed.
    e.g. [['A', 'B', 'C'], ['E', 'F'], 'M']. Each letter indicates a dataset. 
    """
  
    datasets_array = []
    data_array_names_for_log = []
    
    dataname_array = cfg.data_array
    for group_i in dataname_array:
        dataset_group_i = []
        data_group_i_names_for_log = []
        if not isinstance(group_i, list):
            group_i = [group_i, ]
        for data_i in group_i:
            if not isinstance(data_i, dict):
                raise TypeError(f'data name must be a dict, but got {type(data_i)}')
            # each data only can employ a single dataset config
            assert len(data_i.values()) == 1
            if list(data_i.values())[0] not in cfg:
                raise RuntimeError(f'cannot find the data config for {data_i}')
            
            # dataset configure for data i
            #data_i_cfg = cfg[data_i]
            args = copy.deepcopy(cfg) #data_i_cfg.copy()
            data_i_cfg_name = list(data_i.values())[0]
            data_i_db_info_name = list(data_i.keys())[0]
            data_i_db_info = cfg.db_info[data_i_db_info_name]

            # Online evaluation using only metric datasets
            # if phase == 'val' and 'exclude' in cfg.evaluation \
            #     and data_i_db_info_name in cfg.evaluation.exclude:
            #     continue

            # dataset lib name
            obj_name = cfg[data_i_cfg_name]['lib']
            obj_path = os.path.dirname(__file__).split(os.getcwd() + '/')[-1].replace('/', '.') + '.' + obj_name 
            obj_cls = get_func(obj_path)
            if obj_cls is None:
                raise KeyError(f'{obj_name} is not in .data')
                             
            dataset_i = obj_cls(
                args[data_i_cfg_name], 
                phase, 
                db_info=data_i_db_info, 
                **cfg.data_basic)
            # if 'Taskonomy' not in data_i:
            #     print('>>>>>>>>>>ditributed_sampler LN189', dataset_i.data_name, dataset_i.annotations['files'][0]['rgb'].split('/')[-1],
            #       dataset_i.annotations['files'][1000]['rgb'].split('/')[-1], dataset_i.annotations['files'][3000]['rgb'].split('/')[-1])
            # else:
            #     print('>>>>>>>>>>ditributed_sampler LN189', dataset_i.data_name, dataset_i.annotations['files'][0]['meta_data'].split('/')[-1],
            #       dataset_i.annotations['files'][1000]['meta_data'].split('/')[-1], dataset_i.annotations['files'][3000]['meta_data'].split('/')[-1])
            dataset_group_i.append(dataset_i)
            # get data name for log
            data_group_i_names_for_log.append(data_i_db_info_name)

        datasets_array.append(dataset_group_i)
        data_array_names_for_log.append(data_group_i_names_for_log)

    if main_process():
        logger = logging.getLogger()
        logger.info(f'{phase}: data array ({data_array_names_for_log}) has been constructed.')
    return datasets_array
            
def concatenate_datasets(datasets_array):
    """
    Merge grouped datasets to a single one.
    args:
        @ dataset_list: the list of constructed dataset. 
    """
    #max_size = 0
    dataset_merge = []                         
    for group in datasets_array:
        group_dataset = ConcatDataset(group)
        group_size = len(group_dataset)
        #max_size = max_size if group_size < max_size else group_size
        dataset_merge.append(group_dataset)
    return ConcatDataset(dataset_merge)


def log_canonical_transfer_info(cfg):
    logger = logging.getLogger()
    data = []
    canonical_focal_length = cfg.data_basic.canonical_space.focal_length
    canonical_size = cfg.data_basic.canonical_space.img_size
    for group_i in cfg.data_array:
        if not isinstance(group_i, list):
            group_i = [group_i, ]
        for data_i in group_i:
            if not isinstance(data_i, dict):
                raise TypeError(f'data name must be a dict, but got {type(data_i)}')
            assert len(data_i.values()) == 1
            if list(data_i.values())[0] not in cfg:
                raise RuntimeError(f'cannot find the data config for {data_i.values()}')
            if list(data_i.values())[0] not in data:
                data.append(list(data_i.values())[0])

    logger.info('>>>>>>>>>>>>>>Some data transfer details during augmentation.>>>>>>>>>>>>>>')
    for data_i in data:
        data_i_cfg = cfg[data_i]
        if type(data_i_cfg.original_focal_length) != tuple:
            ori_focal = (data_i_cfg.original_focal_length, )
        else:
            ori_focal = data_i_cfg.original_focal_length
        
        log_str = '%s transfer details: \n' % data_i
        for ori_f  in ori_focal:
            # to canonical space
            scalor = canonical_focal_length / ori_f
            img_size = (data_i_cfg.original_size[0]*scalor,  data_i_cfg.original_size[1]*scalor)
            log_str += 'To canonical space: focal length, %f -> %f; size, %s -> %s\n' %(ori_f, canonical_focal_length, data_i_cfg.original_size, img_size)
            
            # random resize in augmentaiton
            resize_range = data_i_cfg.data.train.pipeline[1].ratio_range
            resize_low = (img_size[0]*resize_range[0],  img_size[1]*resize_range[0])
            resize_up = (img_size[0]*resize_range[1],  img_size[1]*resize_range[1])
            log_str += 'Random resize bound: %s ~ %s; \n' %(resize_low, resize_up)
        
        logger.info(log_str)