from .losses import *
from mono.utils.comm import get_func
import os

def build_from_cfg(cfg, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise RuntimeError('should contain the loss name')
    args = cfg.copy()
    
    obj_name = args.pop('type')
    obj_path = os.path.dirname(__file__).split(os.getcwd() + '/')[-1].replace('/', '.') + '.losses.' + obj_name 
    
    obj_cls = get_func(obj_path)(**args)
    
    if obj_cls is None:
        raise KeyError(f'cannot find {obj_name}.')
    return obj_cls


        
        
def build_criterions(cfg):
    if 'losses' not in cfg:
        raise RuntimeError('Losses have not been configured.')
    cfg_data_basic = cfg.data_basic

    criterions = dict()
    losses = cfg.losses
    if not isinstance(losses, dict):
        raise RuntimeError(f'Cannot initial losses with the type {type(losses)}')
    for key, loss_list in losses.items():
        criterions[key] = []
        for loss_cfg_i in loss_list:
            # update the canonical_space configs to the current loss cfg
            loss_cfg_i.update(cfg_data_basic)
            if 'out_channel' in loss_cfg_i:
                loss_cfg_i.update(out_channel=cfg.out_channel)  # classification loss need to update the channels
            obj_cls = build_from_cfg(loss_cfg_i)
            criterions[key].append(obj_cls)
    return criterions
            

            
        
            
            
            
            
            
            
        
    
  
