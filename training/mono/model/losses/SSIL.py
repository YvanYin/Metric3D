import torch
import torch.nn as nn

class SSILoss(nn.Module):
    """
    Scale shift invariant MAE loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(SSILoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def ssi_mae(self, target, prediction, mask):
        valid_pixes = torch.sum(mask) + self.eps

        gt_median = torch.median(target) if target.numel() else 0
        gt_s = torch.abs(target - gt_median).sum() / valid_pixes
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = torch.median(prediction) if prediction.numel() else 0
        pred_s = torch.abs(prediction - pred_median).sum() / valid_pixes
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)
        
        ssi_mae_sum = torch.sum(torch.abs(gt_trans - pred_trans))
        return ssi_mae_sum, valid_pixes

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = prediction.shape
        loss = 0
        valid_pix = 0
        for i in range(B):
            mask_i = mask[i, ...]
            gt_depth_i = target[i, ...][mask_i]
            pred_depth_i = prediction[i, ...][mask_i]
            ssi_sum, valid_pix_i = self.ssi_mae(pred_depth_i, gt_depth_i, mask_i) 
            loss += ssi_sum
            valid_pix += valid_pix_i
        loss /= (valid_pix + self.eps)
        return loss * self.loss_weight
    
if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    ssil = SSILoss()
    pred = torch.rand((2, 1, 256, 256)).cuda()
    gt = torch.rand((2, 1, 256, 256)).cuda()#torch.zeros_like(pred).cuda() #
    gt[:, :, 100:256, 0:100] = -1
    mask = gt > 0
    out = ssil(pred, gt, mask)
    print(out)
