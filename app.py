
import torch
import torch.nn.functional as F
import logging
import os
import os.path as osp

import sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
from mono.utils.logger import setup_logger
import glob
from mono.utils.comm import init_env
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.running import load_ckpt
from mono.utils.do_test import transform_test_data_scalecano, get_prediction
from mono.utils.custom_data import load_from_annos, load_data

from mono.utils.avg_meter import MetricAverageMeter
from mono.utils.visualization import save_val_imgs, create_html, save_raw_imgs, save_normal_val_imgs
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mono.utils.unproj_pcd import reconstruct_pcd, save_point_cloud
from mono.utils.transform import gray_to_colormap
from mono.utils.visualization import vis_surface_normal
import gradio as gr

torch.hub.download_url_to_file('https://images.unsplash.com/photo-1437622368342-7a3d73a34c8f', 'turtle.jpg')
torch.hub.download_url_to_file('https://images.unsplash.com/photo-1519066629447-267fffa62d4b', 'lions.jpg')

cfg_large = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.large.py')
model_large = get_configured_monodepth_model(cfg_large, )
model_large, _,  _, _ = load_ckpt('./weight/metric_depth_vit_large_800k.pth', model_large, strict_match=False)
model_large.eval()

cfg_small = Config.fromfile('./mono/configs/HourglassDecoder/vit.raft5.small.py')
model_small = get_configured_monodepth_model(cfg_small, )
model_small, _,  _, _ = load_ckpt('./weight/metric_depth_vit_small_800k.pth', model_small, strict_match=False)
model_small.eval()

device = "cuda"
model_large.to(device)
model_small.to(device)

def depth_normal(img, model_selection="vit-small"):
    if model_selection == "vit-small":
        model = model_small
        cfg = cfg_small
    elif model_selection == "vit-large":
        model = model_large
        cfg = cfg_large

    else:
        raise NotImplementedError
    
    cv_image = np.array(img) 
    img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    intrinsic = [1000.0, 1000.0, img.shape[1]/2, img.shape[0]/2]
    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(img, intrinsic, cfg.data_basic)

    with torch.no_grad():
        pred_depth, pred_depth_scale, scale, output = get_prediction(
                    model = model,
                    input = rgb_input,
                    cam_model = cam_models_stacks,
                    pad_info = pad,
                    scale_info = label_scale_factor,
                    gt_depth = None,
                    normalize_scale = cfg.data_basic.depth_range[1],
                    ori_shape=[img.shape[0], img.shape[1]],
                )

        pred_normal = output['normal_out_list'][0][:, :3, :, :] 
        H, W = pred_normal.shape[2:]
        pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

    pred_depth = pred_depth.squeeze().cpu().numpy()
    pred_depth[pred_depth<0] = 0
    pred_color = gray_to_colormap(pred_depth)

    pred_normal = pred_normal.squeeze()
    if pred_normal.size(0) == 3:
        pred_normal = pred_normal.permute(1,2,0)
    pred_color_normal = vis_surface_normal(pred_normal)
    
    ##formatted = (output * 255 / np.max(output)).astype('uint8')
    img = Image.fromarray(pred_color)
    img_normal = Image.fromarray(pred_color_normal)
    return img, img_normal
        
#inputs =  gr.inputs.Image(type='pil', label="Original Image")
#depth = gr.outputs.Image(type="pil",label="Output Depth")
#normal = gr.outputs.Image(type="pil",label="Output Normal")

title = "Metric3D"
description = "Gradio demo for Metric3D (v2, more diverse models) running on CPU which takes in a single image for computing metric depth and surface normal. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/pdf/2307.10984.pdf'>Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image</a> | <a href='https://github.com/YvanYin/Metric3D'>Github Repo</a></p>"

examples = [
    ["turtle.jpg"],
    ["lions.jpg"]
]

gr.Interface(
    depth_normal, 
    inputs=[gr.Image(type='pil', label="Original Image"), gr.Dropdown(["vit-small", "vit-large"], label="Model", info="Will support more models later!")], 
    outputs=[gr.Image(type="pil",label="Output Depth"), gr.Image(type="pil",label="Output Normal")], 
    title=title, description=description, article=article, examples=examples, analytics_enabled=False).launch()