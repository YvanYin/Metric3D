import torch
import torch.nn as nn

class ConfidenceLoss(nn.Module):
    """
    confidence loss.
    """
    def __init__(self, loss_weight=1, data_type=['stereo', 'lidar', 'denselidar'], **kwargs):
        super(ConfidenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, confidence, mask=None, **kwargs):
        conf_mask = torch.abs(target - prediction) < target
        conf_mask = conf_mask & mask        
        gt_confidence = (1 - torch.abs((prediction - target) / target)) * conf_mask
        loss = torch.sum(torch.abs(confidence - gt_confidence) * conf_mask) / (torch.sum(conf_mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(confidence) 
            print(f'ConfidenceLoss NAN error, {loss}')
        return loss * self.loss_weight