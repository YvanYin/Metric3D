import torch
import torch.nn as nn

class RegularizationLoss(nn.Module):
    """
    Enforce losses on pixels without any gts.
    """
    def __init__(self, loss_weight=0.1, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(RegularizationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):
        pred_wo_gt = prediction[~mask]
        #loss = - torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + 1e-8)
        loss = 1/ (torch.sum(pred_wo_gt) / (pred_wo_gt.numel() + self.eps))
        return loss * self.loss_weight