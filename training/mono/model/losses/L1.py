import torch
import torch.nn as nn

class L1Loss(nn.Module):
    """
    Compute L1 loss.
    """
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'stereo', 'denselidar_syn'], **kwargs):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):
        diff = torch.abs(prediction - target)* mask
        loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'L1 NAN error, {loss}')
            #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight

class L1DispLoss(nn.Module):
    """
    Compute L1 disparity loss of disparity.
    """
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'stereo', 'denselidar_syn'], **kwargs):
        super(L1DispLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction_disp, inv_depth, mask=None, **kwargs):
        # gt_disp_mask = ~torch.all(inv_depth == 0, dim=1, keepdim=True)
        # if mask is None:
        #     mask = gt_disp_mask
        diff = torch.abs(prediction_disp - inv_depth)* mask
        loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction_disp)
            #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight
    
class L1InverseLoss(nn.Module):
    """
    Compute L1 disparity loss of disparity.
    """
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'stereo'], **kwargs):
        super(L1InverseLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, inv_depth, mask=None, **kwargs):
        mask = torch.logical_and(mask, inv_depth>0)
        inv_pred = 1.0 / prediction * 10.0
        inv_pred[~mask] = -1
        diff = torch.abs(inv_pred - inv_depth)* mask
        loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(inv_pred)
            #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight