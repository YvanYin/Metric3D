import torch
import torch.nn as nn

class SoftWeight(nn.Module):
    """
    Transfer n-channel discrete depth bins to a depth map.
    Args:
        @depth_bin: n-channel output of the network, [b, c, h, w]
    Return: 1-channel depth, [b, 1, h, w]
    """
    def __init__(self, depth_bins_border):
        super(SoftWeight, self).__init__()
        self.register_buffer("depth_bins_border", torch.tensor(depth_bins_border), persistent=False)

    def forward(self, pred_logit):
        if type(pred_logit).__module__ != torch.__name__:
            pred_logit = torch.tensor(pred_logit, dtype=torch.float32, device="cuda")
        pred_score = nn.functional.softmax(pred_logit, dim=1)
        pred_score_ch = pred_score.permute(0, 2, 3, 1) #[b, h, w, c]
        pred_score_weight = pred_score_ch * self.depth_bins_border
        depth_log = torch.sum(pred_score_weight, dim=3, dtype=torch.float32, keepdim=True)
        depth = 10 ** depth_log
        depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]
        confidence, _ = torch.max(pred_logit, dim=1, keepdim=True)
        return depth, confidence

def soft_weight(pred_logit, depth_bins_border):
    """
    Transfer n-channel discrete depth bins to depth map.
    Args:
        @depth_bin: n-channel output of the network, [b, c, h, w]
    Return: 1-channel depth, [b, 1, h, w]
    """
    if type(pred_logit).__module__ != torch.__name__:
        pred_logit = torch.tensor(pred_logit, dtype=torch.float32, device="cuda")
    if type(depth_bins_border).__module__ != torch.__name__:
        depth_bins_border = torch.tensor(depth_bins_border, dtype=torch.float32, device="cuda")

    pred_score = nn.functional.softmax(pred_logit, dim=1)
    depth_bins_ch = pred_score.permute(0, 2, 3, 1) #[b, h, w, c]    depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
    depth = 10 ** depth
    depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]

    confidence, _ = torch.max(pred_logit, dim=1, keepdim=True)
    return depth, confidence



if __name__ == '__main__':
    import numpy as np
    depth_max = 100
    depth_min = 0.5

    depth_bin_interval = (np.log10(depth_max) - np.log10(depth_min)) / 200
    depth_bins_border = [np.log10(depth_min) + depth_bin_interval * (i + 0.5)
                     for i in range(200)]
    
    sw = SoftWeight(depth_bins_border)