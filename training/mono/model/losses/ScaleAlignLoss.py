import torch
import torch.nn as nn

class ScaleAlignLoss(nn.Module):
    """
    Loss function defined over sequence of depth predictions
    """
    def __init__(self, data_type=['lidar', 'denselidar', 'stereo', 'denselidar_syn'], loss_weight=1.0, disable_dataset=['MapillaryPSD'], **kwargs):
        super(ScaleAlignLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.disable_dataset = disable_dataset

    def forward(self, prediction, target, mask, scale, **kwargs):
        device = target.device

        B, C, H, W = prediction.shape


        # median_pred, _ = torch.median(prediction.view(B, C*H*W), 1)
        # median_pred = median_pred.detach()

        # scale_factor = torch.zeros_like(scale).squeeze(3).squeeze(2).squeeze(1)
        # for i in range(B):
        #     mask_i = mask[i, ...]
        #     if torch.sum(mask_i) > 10:
        #         scale_factor[i] = torch.median(target[i, ...][mask_i]) / (torch.median(prediction[i, ...][mask_i]) + 1e-8)
        #     else:
        #         scale_factor[i] = 0
        
        # target_scale = (median_pred * scale_factor)

        # batches_dataset = kwargs['dataset']
        # self.batch_valid = torch.tensor([1 if batch_dataset not in self.disable_dataset else 0 \
        #     for batch_dataset in batches_dataset], device=device)

        # batch_valid = self.batch_valid * (scale_factor > 1e-8)

        # scale_diff = torch.abs(scale.squeeze(3).squeeze(2).squeeze(1) - scale_factor * median_pred)

        batches_dataset = kwargs['dataset']
        self.batch_valid = torch.tensor([1 if batch_dataset not in self.disable_dataset else 0 \
            for batch_dataset in batches_dataset], device=device)

        scale_tgt = torch.zeros_like(scale).squeeze(3).squeeze(2).squeeze(1)
        for i in range(B):
            mask_i = mask[i, ...]
            if torch.sum(mask_i) > 10:
                scale_tgt[i] = torch.median(target[i, ...][mask_i])
            else:
                scale_tgt[i] = 0
        
        batch_valid = self.batch_valid * (scale_tgt > 1e-8)
        scale_diff = torch.abs(scale.squeeze(3).squeeze(2).squeeze(1) - scale_tgt)
        loss = torch.sum(scale_diff * batch_valid) / (torch.sum(batch_valid) + 1e-8)

        return loss * self.loss_weight