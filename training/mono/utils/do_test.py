import torch
import logging
import os
from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, visual_train_data, create_html, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from mono.utils.logger import setup_logger
from mono.utils.comm import main_process
#from scipy.optimize import minimize
#from torchmin import minimize
import torch.optim as optim
from torch.autograd import Variable


def to_cuda(data: dict):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda(non_blocking=True)
        if isinstance(v, list) and len(v)>=1 and isinstance(v[0], torch.Tensor):
            for i, l_i in enumerate(v):
                data[k][i] = l_i.cuda(non_blocking=True)
    return data

def align_scale(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
    else:
        scale = 1
    pred_scale = pred * scale
    return pred_scale, scale

def align_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    if torch.sum(mask) > 10:
        shift = torch.median(target[mask]) - (torch.median(pred[mask]) + 1e-8)
    else:
        shift = 0
    pred_shift = pred + shift
    return pred_shift, shift

def align_scale_shift(pred: torch.tensor, target: torch.tensor):
    mask = target > 0
    target_mask = target[mask].cpu().numpy()
    pred_mask = pred[mask].cpu().numpy()
    if torch.sum(mask) > 10:
        scale, shift = np.polyfit(pred_mask, target_mask, deg=1)
        if scale < 0:
            scale = torch.median(target[mask]) / (torch.median(pred[mask]) + 1e-8)
            shift = 0
    else:
        scale = 1
        shift = 0
    pred = pred * scale + shift
    return pred, scale

def get_prediction(
    model: torch.nn.Module,
    input: torch.tensor, 
    cam_model: torch.tensor,
    pad_info: torch.tensor,
    scale_info: torch.tensor,
    gt_depth: torch.tensor,
    normalize_scale: float,
    intrinsic = None,
    clip_range = None,
    flip_aug = False):    
    #clip_range = [0, 10],
    #flip_aug = True):

    data = dict(
        input=input,
        #ref_input=ref_input,
        cam_model=cam_model
    )
    #output = model.module.inference(data)
    output = model.module.inference(data)
    pred_depth, confidence = output['prediction'], output['confidence']
    pred_depth = torch.abs(pred_depth)
    pred_depth = pred_depth.squeeze()

    if flip_aug == True:
        output_flip = model.module.inference(dict(
        input=torch.flip(input, [3]),
        #ref_input=ref_input,
        cam_model=cam_model
    ))

        if clip_range != None:
            output['prediction'] = torch.clamp(output['prediction'], clip_range[0], clip_range[1])
            output_flip['prediction'] = torch.clamp(output_flip['prediction'], clip_range[0] / normalize_scale * scale_info , clip_range[1] / normalize_scale * scale_info)

        output['prediction'] = 0.5 * (output['prediction'] + torch.flip(output_flip['prediction'], [3]))
        output['confidence'] = 0.5 * (output['confidence'] + torch.flip(output_flip['confidence'], [3]))

    output['pad'] = torch.Tensor(pad_info).cuda().unsqueeze(0).int()
    output['mask'] = torch.ones_like(pred_depth).bool().unsqueeze(0).unsqueeze(1)
    output['scale_info'] = scale_info 
    if intrinsic is not None:
        output['intrinsic'] = intrinsic

    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0]-pad_info[1], pad_info[2]: pred_depth.shape[1]-pad_info[3]]
    pred_depth = torch.nn.functional.interpolate(pred_depth[None, None, :, :], gt_depth.shape, mode='bilinear').squeeze() # to orginal size
    pred_depth = pred_depth * normalize_scale / scale_info 

    if clip_range != None:
        pred_depth = torch.clamp(pred_depth, clip_range[0], clip_range[1])

    pred_depth_scale, scale = align_scale(pred_depth, gt_depth) #align_scale_shift(pred_depth, gt_depth) 

    if clip_range != None:
        pred_depth_scale = torch.clamp(pred_depth_scale, clip_range[0], clip_range[1])

    return pred_depth, pred_depth_scale, scale, output


# def depth_normal_consistency_optimization(output_dict, consistency_fn): 
#     s = torch.zeros_like(output_dict['scale_info'])
#     def closure(x):
#         output_dict['scale'] = torch.exp(x) * output_dict['scale_info']  
#         error = consistency_fn(**output_dict)
#         return error + x * x

#     result = minimize(closure, s, method='newton-exact', disp=1, options={'max_iter':10, 'lr':0.1})
#     return float(torch.exp(-result.x))


def do_test_with_dataloader(
    model: torch.nn.Module, 
    cfg: dict, 
    dataloader: torch.utils.data,
    logger: logging.RootLogger,
    is_distributed: bool = True,
    local_rank: int = 0):
    
    show_dir = cfg.show_dir
    save_interval = 100
    save_html_path = show_dir + '/index.html'
    save_imgs_dir = show_dir + '/vis'
    os.makedirs(save_imgs_dir, exist_ok=True)
    save_raw_dir = show_dir + '/raw'
    os.makedirs(save_raw_dir, exist_ok=True)

    normalize_scale = cfg.data_basic.depth_range[1]

    dam = MetricAverageMeter(cfg.test_metrics)
    dam_scale = MetricAverageMeter(cfg.test_metrics)

    try:
        depth_range = cfg.data_basic.clip_depth_range if cfg.clip_depth else None
    except:
        depth_range = None

    for i, data in enumerate(tqdm(dataloader)):

        # logger.info(f'{local_rank}: {i}/{len(dataloader)}')
        data = to_cuda(data)
        gt_depth = data['target'].squeeze()
        mask = gt_depth > 1e-6
        pad_info = data['pad']
        pred_depth, pred_depth_scale, scale, output = get_prediction(
            model,
            data['input'],
            data['cam_model'],
            pad_info,
            data['scale'],
            gt_depth,
            normalize_scale,
            data['intrinsic'],
        )
        
        logger.info(f'{data["filename"]}: {scale}')

        # optimization
        #if "normal_out_list" in output.keys():
            #scale_opt = depth_normal_consistency_optimization(output, consistency_loss)
            #print('scale', scale_opt, float(scale)) 
        scale_opt = 1.0

        # update depth metrics
        dam_scale.update_metrics_gpu(pred_depth_scale, gt_depth, mask, is_distributed)
        dam.update_metrics_gpu(pred_depth, gt_depth, mask, is_distributed)

        # save evaluation results
        if i % save_interval == 0:
            # save 
            rgb = data['input'][:, :, pad_info[0]: data['input'].shape[2]-pad_info[1], pad_info[2]: data['input'].shape[3]-pad_info[3]]
            rgb = torch.nn.functional.interpolate(rgb, gt_depth.shape, mode='bilinear').squeeze()
            max_scale = save_val_imgs(i,
                          pred_depth, 
                          gt_depth, 
                          rgb, 
                          data['filename'][0], 
                          save_imgs_dir,
                          )
            logger.info(f'{data["filename"]}, {"max_scale"}: {max_scale}')

            # # save original depth/rgb
            # save_raw_imgs(
            #     pred_depth.cpu().squeeze().numpy(),
            #     data['raw_rgb'].cpu().squeeze().numpy(), 
            #     data['filename'][0], 
            #     save_raw_dir,
            # )

        
        # surface normal metrics
        if "normal_out_list" in output.keys():
            normal_out_list = output['normal_out_list']
            gt_normal = data['normal']

            pred_normal = normal_out_list[-1][:, :3, :, :] # (B, 3, H, W)
            H, W = pred_normal.shape[2:]
            pred_normal = pred_normal[:, :, pad_info[0]:H-pad_info[1], pad_info[2]:W-pad_info[3]]
            pred_normal = torch.nn.functional.interpolate(pred_normal, size=gt_normal.shape[2:], mode='bilinear', align_corners=True)

            gt_normal_mask = ~torch.all(gt_normal == 0, dim=1, keepdim=True)
            dam.update_normal_metrics_gpu(pred_normal, gt_normal, gt_normal_mask, cfg.distributed)# save valiad normal

            if i % save_interval == 0:
                save_normal_val_imgs(iter, 
                                    pred_normal, 
                                    gt_normal, 
                                    rgb, # data['input'], 
                                    'normal_' + data['filename'][0], 
                                    save_imgs_dir,
                                    )

    # get validation error
    if main_process():
        eval_error = dam.get_metrics()
        print('>>>>>W/o scale: ', eval_error)
        eval_error_scale = dam_scale.get_metrics()
        print('>>>>>W scale: ', eval_error_scale)
        # disp_eval_error = dam_disp.get_metrics()
        # print('>>>>>Disp to depth: ', disp_eval_error)
        # for i, dam in enumerate(dams):
        #     print(f'>>>>>W/o scale gru{i}: ', dam.get_metrics())

        logger.info(eval_error)
        logger.info(eval_error_scale)
        # logger.info(disp_eval_error)
        # [logger.info(dam.get_metrics()) for dam in dams]
