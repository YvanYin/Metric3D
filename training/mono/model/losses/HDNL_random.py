import torch
import torch.nn as nn
import numpy as np

class HDNRandomLoss(nn.Module):
    """
    Hieratical depth normalization loss. Replace the original hieratical depth ranges with randomly sampled ranges.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, random_num=32, data_type=['sfm', 'stereo', 'lidar', 'denselidar', 'denselidar_nometric', 'denselidar_syn'], norm_dataset=['Taskonomy', 'Matterport3D', 'Replica', 'Hypersim'], disable_dataset=['MapillaryPSD'], **kwargs):
        super(HDNRandomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.random_num = random_num
        self.eps = 1e-6
        self.data_type = data_type
        self.disable_dataset = disable_dataset
    
    def get_random_masks_for_batch(self, depth_gt: torch.Tensor, mask_valid: torch.Tensor)-> torch.Tensor:
        valid_values = depth_gt[mask_valid]
        max_d = valid_values.max().item() if valid_values.numel() > 0 else 0.0 
        min_d = valid_values.min().item() if valid_values.numel() > 0 else 0.0

        sample_min_d = np.random.uniform(0, 0.75, self.random_num) * (max_d - min_d) + min_d
        sample_max_d = np.random.uniform(sample_min_d + 0.1, 1-self.eps, self.random_num) * (max_d - min_d) + min_d

        mask_new = [(depth_gt >= sample_min_d[i]) & (depth_gt < sample_max_d[i] + 1e-30) & mask_valid for i in range(self.random_num)]
        mask_new = torch.stack(mask_new, dim=0).cuda() #[N, 1, H, W]
        return mask_new

    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + self.eps
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        return  loss_sum

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        
        loss = 0.0
        valid_pix = 0.0
        
        device = target.device
        
        batches_dataset = kwargs['dataset']
        self.batch_valid = torch.tensor([1 if batch_dataset not in self.disable_dataset else 0 \
            for batch_dataset in batches_dataset], device=device)[:,None,None,None]
        
        batch_limit = 4
        loops = int(np.ceil(self.random_num / batch_limit))
        for i in range(B):                
            mask_i = mask[i, ...] #[1, H, W]

            if self.batch_valid[i, ...] < 0.5:
                loss += 0 * torch.sum(prediction[i, ...])
                valid_pix += 0 * torch.sum(mask_i)
                continue

            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            mask_random_drange = self.get_random_masks_for_batch(target[i, ...], mask_i) # [N, 1, H, W]
            for j in range(loops):
                mask_random_loopi = mask_random_drange[j*batch_limit:(j+1)*batch_limit, ...]
                loss += self.ssi_mae(
                    prediction=pred_i[:mask_random_loopi.shape[0], ...], 
                    target=target_i[:mask_random_loopi.shape[0], ...], 
                    mask_valid=mask_random_loopi)
                valid_pix += torch.sum(mask_random_loopi)

        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'HDNL NAN error, {loss}, valid pix: {valid_pix}')
        return loss * self.loss_weight
    
if __name__ == '__main__':
    ssil = HDNRandomLoss()
    pred = torch.rand((2, 1, 256, 256)).cuda()
    gt =  - torch.rand((2, 1, 256, 256)).cuda()#torch.zeros_like(pred).cuda() #
    gt[:, :, 100:256, 0:100] = -1
    mask = gt > 0
    out = ssil(pred, gt, mask)
    print(out)
