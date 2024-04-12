import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceGuideLoss(nn.Module):
    """
    confidence guide depth loss.
    """
    def __init__(self, loss_weight=1, data_type=['stereo', 'lidar', 'denselidar'], loss_gamma=0.9, conf_loss=True, **kwargs):
        super(ConfidenceGuideLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.conf_loss = conf_loss

    def forward(self, samples_pred_list, target, coord_list, mask=None, **kwargs):
        loss = 0.0
        n_predictions = len(samples_pred_list)
        for i, (pred, coord) in enumerate(zip(samples_pred_list, coord_list)):
            # coord: B, 1, N, 2
            # pred: B, 2, N
            gt_depth_ = F.grid_sample(target, coord, mode='nearest', align_corners=True) # (B, 1, 1, N)
            gt_depth_mask_ = F.grid_sample(mask.float(), coord, mode='nearest', align_corners=True) # (B, 1, 1, N)
            gt_depth_ = gt_depth_[:, :, 0, :]
            gt_depth_mask_ = gt_depth_mask_[:, :, 0, :] > 0.5

            pred_depth, pred_conf = pred[:, :1, :], pred[:, 1:, :]

            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            # depth L1 loss
            diff = torch.abs(pred_depth - gt_depth_) * gt_depth_mask_
            curr_loss = torch.sum(diff) / (torch.sum(gt_depth_mask_) + self.eps)
            if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
                curr_loss = 0 * torch.sum(pred_depth)
                print(f'GRUSequenceLoss-depth NAN error, {loss}')

            # confidence L1 loss
            conf_loss = 0.0
            if self.conf_loss:
                conf_mask = torch.abs(gt_depth_ - pred_depth) < gt_depth_
                conf_mask = conf_mask & gt_depth_mask_
                gt_confidence = (1 - torch.abs((pred_depth - gt_depth_) / gt_depth_)) * conf_mask
                conf_loss = torch.sum(torch.abs(pred_conf - gt_confidence) * conf_mask) / (torch.sum(conf_mask) + self.eps)
                if torch.isnan(conf_loss).item() | torch.isinf(conf_loss).item():
                    conf_loss = 0 * torch.sum(pred_conf)
                    print(f'GRUSequenceLoss-confidence NAN error, {conf_loss}')

            loss += (conf_loss + curr_loss) * i_weight

        return loss * self.loss_weight