import os
import os.path as osp
import time
import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(CODE_SPACE)
#os.chdir(CODE_SPACE)
import argparse
import mmcv
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from datetime import timedelta
import random
import numpy as np

from mono.datasets.distributed_sampler import log_canonical_transfer_info
from mono.utils.comm import init_env
from mono.utils.logger import setup_logger
from mono.utils.db import load_data_info, reset_ckpt_path
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.datasets.distributed_sampler import build_dataset_n_sampler_with_cfg
from mono.utils.running import load_ckpt
from mono.utils.do_test import do_test_with_dataloader, do_test_check_data

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--show-dir', help='the dir to save logs and visualization results')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument('--node_rank', type=int, default=0)
    parser.add_argument('--nnodes', 
                        type=int, 
                        default=1, 
                        help='number of nodes')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')   
    parser.add_argument(
        '--launcher', choices=['None', 'pytorch', 'slurm'], default='slurm',
        help='job launcher')
    args = parser.parse_args()
    return args

        
def main(args):
    os.chdir(CODE_SPACE)
    cfg = Config.fromfile(args.config)
    cfg.dist_params.nnodes = args.nnodes
    cfg.dist_params.node_rank = args.node_rank

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    #if cfg.get('cudnn_benchmark', False) and args.launcher != 'ror':
    #    torch.backends.cudnn.benchmark = True

    # show_dir is determined in this priority: CLI > segment in file > filename
    if args.show_dir is not None:
        # update configs according to CLI args if args.show_dir is not None
        cfg.show_dir = args.show_dir
    elif cfg.get('show_dir', None) is None:
        # use config filename + timestamp as default show_dir if cfg.show_dir is None
        cfg.show_dir = osp.join('./show_dirs',
                                osp.splitext(osp.basename(args.config))[0], 
                                args.timestamp)

    # ckpt path
    if args.load_from is None:
        raise RuntimeError('Please set model path!')
    cfg.load_from = args.load_from
    
    # create show dir
    os.makedirs(osp.abspath(cfg.show_dir), exist_ok=True)
    
    # init the logger before other steps
    cfg.log_file = osp.join(cfg.show_dir, f'{args.timestamp}.log')
    logger = setup_logger(cfg.log_file)

    # log some basic info
    logger.info(f'Config:\n{cfg.pretty_text}')

    # load db_info for data
    # load data info
    data_info = {}
    load_data_info('data_server_info', data_info=data_info)
    cfg.db_info = data_info
    # update check point info
    reset_ckpt_path(cfg.model, data_info)

    # log data transfer to canonical space info
    # log_canonical_transfer_info(cfg)
    
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.distributed = False
    else:
        cfg.distributed = True
        init_env(args.launcher, cfg)
    logger.info(f'Distributed training: {cfg.distributed}')

    # dump config
    cfg.dump(osp.join(cfg.show_dir, osp.basename(args.config)))
    
    if not cfg.distributed:
        main_worker(0, cfg, args.launcher)
    else:
        mp.spawn(main_worker, nprocs=cfg.dist_params.num_gpus_per_node, args=(cfg, args.launcher))

def main_worker(local_rank: int, cfg: dict, launcher: str):
    if cfg.distributed:
        cfg.dist_params.global_rank = cfg.dist_params.node_rank * cfg.dist_params.num_gpus_per_node + local_rank
        cfg.dist_params.local_rank = local_rank
        
        torch.cuda.set_device(local_rank)
        default_timeout = timedelta(minutes=30)
        dist.init_process_group(backend=cfg.dist_params.backend,
                            init_method=cfg.dist_params.dist_url,
                            world_size=cfg.dist_params.world_size,
                            rank=cfg.dist_params.global_rank,
                            timeout=default_timeout,)

    logger = setup_logger(cfg.log_file)
    # build model
    model = get_configured_monodepth_model(cfg,
                                           None,
                                           )
    
    # build datasets
    test_dataset, test_sampler = build_dataset_n_sampler_with_cfg(cfg, 'test')
    # build data loaders
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   num_workers=1,
                                                   sampler=test_sampler,
                                                   drop_last=False)

   
    # config distributed training
    if cfg.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), 
                                                          device_ids=[local_rank], 
                                                          output_device=local_rank, 
                                                          find_unused_parameters=True)
    else:
        model = torch.nn.DataParallel(model.cuda())
    
    # load ckpt
    #model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
    model.eval()
    do_test_with_dataloader(model, cfg, test_dataloader, logger=logger, is_distributed=cfg.distributed)
    # do_test_check_data(model, cfg, test_dataloader, logger=logger, is_distributed=cfg.distributed, local_rank=local_rank)


if __name__=='__main__':
    # load args
    args = parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.timestamp = timestamp
    main(args)
