import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkyRegularizationLoss(nn.Module):
    """
    Enforce losses on pixels without any gts.
    """
    def __init__(self, loss_weight=0.1, data_type=['sfm', 'stereo', 'lidar', 'denselidar', 'denselidar_nometric', 'denselidar_syn'], sky_id=142, sample_ratio=0.4, regress_value=1.8, normal_regress=None, normal_weight=1.0, **kwargs):
        super(SkyRegularizationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.sky_id = sky_id
        self.sample_ratio = sample_ratio
        self.eps = 1e-6
        self.regress_value = regress_value
        self.normal_regress = normal_regress
        self.normal_weight = normal_weight
    
    def loss1(self, pred_sky):
        loss = 1/ torch.exp((torch.sum(pred_sky) / (pred_sky.numel() + self.eps)))
        return loss

    def loss2(self, pred_sky):
        loss = torch.sum(torch.abs(pred_sky - self.regress_value)) / (pred_sky.numel() + self.eps)
        return loss

    def loss_norm(self, pred_norm, sky_mask):
        sky_norm = torch.FloatTensor(self.normal_regress).cuda()
        sky_norm = sky_norm.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        dot = torch.cosine_similarity(pred_norm[:, :3, :, :].clone(), sky_norm, dim=1)

        sky_mask_float = sky_mask.float().squeeze()
        valid_mask = sky_mask_float \
                        * (dot.detach() < 0.999).float() \
                        * (dot.detach() > -0.999).float() 

        al = (1 - dot) * valid_mask
        loss = torch.sum(al) / (torch.sum(sky_mask_float) + self.eps)
        return loss

    def forward(self, prediction, target, prediction_normal=None, mask=None, sem_mask=None,  **kwargs):
        sky_mask = sem_mask == self.sky_id
        pred_sky = prediction[sky_mask]
        pred_sky_numel = pred_sky.numel()

        if pred_sky.numel() > 50:
            samples = np.random.choice(pred_sky_numel, int(pred_sky_numel*self.sample_ratio), replace=False)
        
        if pred_sky.numel() > 0:
            #loss = - torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + 1e-8)
            loss = self.loss2(pred_sky)

            if (prediction_normal != None) and (self.normal_regress != None):
                loss_normal = self.loss_norm(prediction_normal, sky_mask)
                loss = loss + loss_normal * self.normal_weight

        else:
            loss = torch.sum(prediction) * 0
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = torch.sum(prediction) * 0
            print(f'SkyRegularization NAN error, {loss}')    
        #    raise RuntimeError(f'Sky Loss error, {loss}')    
        
        return loss * self.loss_weight

if __name__ == '__main__':
    import cv2
    sky = SkyRegularizationLoss()
    pred_depth = np.random.random([2, 1, 480, 640])
    gt_depth = np.zeros_like(pred_depth) #np.random.random([2, 1, 480, 640])
    intrinsic = [[[100, 0, 200], [0, 100, 200], [0, 0, 1]], [[100, 0, 200], [0, 100, 200], [0, 0, 1]],]
    gt_depth = torch.tensor(np.array(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.array(pred_depth, np.float32)).cuda()
    intrinsic = torch.tensor(np.array(intrinsic, np.float32)).cuda()
    mask = gt_depth > 0
    loss1 = sky(pred_depth, gt_depth, mask, mask, intrinsic)
    print(loss1)