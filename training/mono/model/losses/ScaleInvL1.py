import torch
import torch.nn as nn

class ScaleInvL1Loss(nn.Module):
    """
    Compute scale-invariant L1 loss.
    """
    def __init__(self, loss_weight=1, data_type=['sfm', 'denselidar_nometric', 'denselidar_syn'], **kwargs):
        super(ScaleInvL1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):    
        B, _, _, _ = target.shape
        target_nan = target.clone()
        target_nan[~mask] = torch.nan
        median_target = torch.nanmedian(target_nan.view(B, -1), dim=1)[0]
        prediction_nan = prediction.clone().detach()
        prediction_nan[~mask] = torch.nan
        median_prediction = torch.nanmedian(prediction_nan.view(B, -1), dim=1)[0]
        scale = median_target / median_prediction
        scale[torch.isnan(scale)] = 0
        pred_scale = prediction * scale[:, None, None, None]
        
        target_valid = target * mask
        pred_valid = pred_scale * mask
        diff = torch.abs(pred_valid - target_valid)
        # disp_diff = diff / (target_valid + self.eps)
        loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'Scale-invariant L1 NAN error, {loss}')
            #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight
