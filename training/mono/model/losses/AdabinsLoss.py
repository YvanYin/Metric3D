import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
#from pytorch3d.loss import chamfer_distance

class AdabinsLoss(nn.Module):
    """
    Losses employed in Adabins.
    """
    def __init__(self, depth_normalize, variance_focus=0.85, loss_weight=1, out_channel=100, data_type=['stereo', 'lidar'],  w_ce=False, w_chamber=False, **kwargs):
        super(AdabinsLoss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
        self.data_type = data_type
        #self.bins_num = out_channel
        #self.cel = nn.CrossEntropyLoss(ignore_index=self.bins_num + 1)
        self.depth_min = depth_normalize[0]
        self.depth_max = depth_normalize[1]
        self.w_ce = w_ce
        self.eps = 1e-6
    
    def silog_loss(self, prediction, target, mask):
        d = torch.log(prediction[mask]) - torch.log(target[mask])
        d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
        d_mean = torch.sum(d) / (d.numel() + self.eps)
        loss = torch.sqrt(d_square_mean - self.variance_focus * (d_mean ** 2))
        return loss
    
    def chamfer_distance_loss(self, bins, target_depth_maps, mask):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        #mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_depth_maps, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points], dtype=torch.long, device="cuda")
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss
    
    # def depth_to_bins(self, depth, mask, depth_edges, size_limite=(512, 960)):
    #     """
    #     Discretize depth into depth bins. Predefined bins edges are provided.
    #     Mark invalid padding area as bins_num + 1
    #     Args:
    #         @depth: 1-channel depth, [B, 1, h, w]
    #     return: depth bins [B, C, h, w]
    #     """ 
    #     def _depth_to_bins_block_(depth, mask, depth_edges):
    #         bins_id = torch.sum(depth_edges[:, None, None, None, :] < torch.abs(depth)[:, :, :, :, None], dim=-1)
    #         bins_id = bins_id - 1
    #         invalid_mask = ~mask
    #         mask_lower = (depth <= self.depth_min) 
    #         mask_higher = (depth >= self.depth_max)
            
    #         bins_id[mask_lower] = 0
    #         bins_id[mask_higher] = self.bins_num - 1
    #         bins_id[bins_id == self.bins_num] = self.bins_num - 1

    #         bins_id[invalid_mask] = self.bins_num + 1
    #         return bins_id
    #     # _, _, H, W = depth.shape
    #     # bins = mask.clone().long()
    #     # h_blocks = np.ceil(H / size_limite[0]).astype(np.int)
    #     # w_blocks = np.ceil(W/ size_limite[1]).astype(np.int)
    #     # for i in range(h_blocks):
    #     #     for j in range(w_blocks):
    #     #         h_start = i*size_limite[0]
    #     #         h_end_proposal = (i + 1) * size_limite[0]
    #     #         h_end = h_end_proposal if h_end_proposal < H else H
    #     #         w_start = j*size_limite[1]
    #     #         w_end_proposal = (j + 1) * size_limite[1]
    #     #         w_end = w_end_proposal if w_end_proposal < W else W
    #     #         bins_ij = _depth_to_bins_block_(
    #     #             depth[:, :, h_start:h_end, w_start:w_end], 
    #     #             mask[:, :, h_start:h_end, w_start:w_end],
    #     #             depth_edges
    #     #             )
    #     #         bins[:, :, h_start:h_end, w_start:w_end] = bins_ij        
    #     bins = _depth_to_bins_block_(depth, mask, depth_edges)
    #     return bins
    
    # def ce_loss(self, pred_logit, target, mask, bins_edges):
    #     target_depth_bins = self.depth_to_bins(target, mask, bins_edges)
    #     loss = self.cel(pred_logit, target_depth_bins.squeeze().long())
    #     return loss


    def forward(self, prediction, target, bins_edges, mask=None, **kwargs):
        silog_loss = self.silog_loss(prediction=prediction, target=target, mask=mask)
        #cf_loss = self.chamfer_distance_loss(bins=bins_edges, target_depth_maps=target, mask=mask)
        loss = silog_loss * 10 #+ 0.1 * cf_loss
        # if self.w_ce:
        #     loss = loss + self.ce_loss(kwargs['pred_logit'], target, mask, bins_edges)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'Adabins loss error, {loss}')
        return loss * self.loss_weight