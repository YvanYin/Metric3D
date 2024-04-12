import matplotlib.pyplot as plt
import os, cv2
import numpy as np
from mono.utils.transform import gray_to_colormap
import shutil
import glob
from mono.utils.running import main_process
import torch
from html4vision import Col, imagetable

def save_raw_imgs( 
    pred: torch.tensor,  
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str,
    scale: float=1000.0, 
    target: torch.tensor=None,
    ):
    """
    Save raw GT, predictions, RGB in the same file.
    """
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_gt.png'), (pred*scale).astype(np.uint16))
    if target is not None:
        cv2.imwrite(os.path.join(save_dir, filename[:-4]+'_gt.png'), (target*scale).astype(np.uint16))
    
def save_normal_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    #targ: torch.tensor, 
    #rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None, 
    mask=None,
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    mean = np.array([123.675, 116.28, 103.53])[np.newaxis, np.newaxis, :]
    std= np.array([58.395, 57.12, 57.375])[np.newaxis, np.newaxis, :]
    pred = pred.squeeze()
    
    # if pred.size(0) == 3:
    #     pred = pred.permute(1,2,0)
    # pred_color = vis_surface_normal(pred, mask)

    # #save one image only
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'.jpg'), pred_color)

    targ = targ.squeeze()
    rgb = rgb.squeeze()

    if pred.size(0) == 3:
        pred = pred.permute(1,2,0)
    if targ.size(0) == 3:
        targ = targ.permute(1,2,0)
    if rgb.size(0) == 3:
        rgb = rgb.permute(1,2,0)

    pred_color = vis_surface_normal(pred, mask)
    targ_color = vis_surface_normal(targ, mask)
    rgb_color = ((rgb.cpu().numpy() * std) + mean).astype(np.uint8)

    try:
        cat_img = np.concatenate([rgb_color, pred_color, targ_color], axis=0)
    except:
        pred_color = cv2.resize(pred_color, (rgb.shape[1], rgb.shape[0]))
        targ_color = cv2.resize(targ_color, (rgb.shape[1], rgb.shape[0]))
        cat_img = np.concatenate([rgb_color, pred_color, targ_color], axis=0)

    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)
    # cv2.imwrite(os.path.join(save_dir, filename[:-4]+'.jpg'), pred_color)
    # save to tensorboard
    if tb_logger is not None:
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)

    


def save_val_imgs(
    iter: int, 
    pred: torch.tensor, 
    target: torch.tensor, 
    rgb: torch.tensor, 
    filename: str, 
    save_dir: str, 
    tb_logger=None
    ):
    """
    Save GT, predictions, RGB in the same file.
    """
    rgb, pred_scale, target_scale, pred_color, target_color, max_scale = get_data_for_log(pred, target, rgb)
    rgb = rgb.transpose((1, 2, 0))
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_rgb.jpg'), rgb)
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_pred.png'), pred_scale, cmap='rainbow')
    # plt.imsave(os.path.join(save_dir, filename[:-4]+'_gt.png'), target_scale, cmap='rainbow')
    cat_img = np.concatenate([rgb, pred_color, target_color], axis=0)
    plt.imsave(os.path.join(save_dir, filename[:-4]+'_merge.jpg'), cat_img)

    # save to tensorboard
    if tb_logger is not None:
        # tb_logger.add_image(f'{filename[:-4]}_rgb.jpg', rgb, iter)
        # tb_logger.add_image(f'{filename[:-4]}_pred.jpg', gray_to_colormap(pred_scale).transpose((2, 0, 1)), iter)
        # tb_logger.add_image(f'{filename[:-4]}_gt.jpg', gray_to_colormap(target_scale).transpose((2, 0, 1)), iter)
        tb_logger.add_image(f'{filename[:-4]}_merge.jpg', cat_img.transpose((2, 0, 1)), iter)
    return max_scale

def get_data_for_log(pred: torch.tensor, target: torch.tensor, rgb: torch.tensor):
    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]

    pred = pred.squeeze().cpu().numpy()
    target = target.squeeze().cpu().numpy()
    rgb = rgb.squeeze().cpu().numpy()

    pred[pred<0] = 0
    target[target<0] = 0
    #max_scale = max(pred.max(), target.max())
    max_scale = min(2.0 * target.max(), pred.max())
    pred[pred > max_scale] = max_scale

    pred_scale = (pred/max_scale * 10000).astype(np.uint16)
    target_scale = (target/max_scale * 10000).astype(np.uint16)
    pred_color = gray_to_colormap(pred, max_value=max_scale)
    target_color = gray_to_colormap(target, max_value=max_scale)
    
    dilate = True
    if dilate == True:
        k=np.ones((3,3),np.uint8)
        target_color=cv2.dilate(target_color,k,iterations=1)

    pred_color = cv2.resize(pred_color, (rgb.shape[2], rgb.shape[1]))
    target_color = cv2.resize(target_color, (rgb.shape[2], rgb.shape[1]))

    rgb = ((rgb * std) + mean).astype(np.uint8)
    return rgb, pred_scale, target_scale, pred_color, target_color, max_scale


def create_html(name2path, save_path='index.html', size=(256, 384)):
    # table description
    cols = []
    for k, v in name2path.items():
        col_i =  Col('img', k, v) # specify image content for column
        cols.append(col_i)
    # html table generation
    imagetable(cols, out_file=save_path, imsize=size)


def visual_train_data(gt_depth, rgb, filename, wkdir, replace=False, pred=None):
    gt_depth = gt_depth.cpu().squeeze().numpy()
    rgb = rgb.cpu().squeeze().numpy()

    mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
    std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]
    mask = gt_depth > 0
    
    rgb = ((rgb * std) + mean).astype(np.uint8).transpose((1, 2, 0))
    gt_vis = gray_to_colormap(gt_depth)
    if replace:
        rgb[mask] = gt_vis[mask]

    if pred is not None:
        pred_depth = pred.detach().cpu().squeeze().numpy()
        pred_vis = gray_to_colormap(pred_depth)

    merge = np.concatenate([rgb, gt_vis, pred_vis], axis=0)
    
    save_path = os.path.join(wkdir, 'test_train', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, merge)


def create_dir_for_validate_meta(work_dir, iter_id):
    curr_folders = glob.glob(work_dir + '/online_val/*0')
    curr_folders = [i for i in curr_folders if os.path.isdir(i)]
    if len(curr_folders) > 8:
        curr_folders.sort()
        del_foler = curr_folders.pop(0)
        print(del_foler)
        if main_process():
            # only rank==0 do it
            if os.path.exists(del_foler):   
                shutil.rmtree(del_foler)
            if os.path.exists(del_foler + '.html'):
                os.remove(del_foler + '.html')
        
    save_val_meta_data_dir = os.path.join(work_dir, 'online_val', '%08d'%iter_id)
    os.makedirs(save_val_meta_data_dir, exist_ok=True)
    return save_val_meta_data_dir


def vis_surface_normal(normal: torch.tensor, mask: torch.tensor=None) -> np.array:
    """
    Visualize surface normal. Transfer surface normal value from [-1, 1] to [0, 255]
    Aargs:
        normal (torch.tensor, [h, w, 3]): surface normal
        mask (torch.tensor, [h, w]): valid masks
    """
    normal = normal.cpu().numpy().squeeze()
    n_img_L2 = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    n_img_norm = normal / (n_img_L2 + 1e-8)
    normal_vis = n_img_norm * 127
    normal_vis += 128
    normal_vis = normal_vis.astype(np.uint8)
    if mask is not None:
        mask = mask.cpu().numpy().squeeze()
        normal_vis[~mask] = 0
    return normal_vis