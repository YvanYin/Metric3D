import torch
import torch.nn as nn
import numpy as np


class WCELoss(nn.Module):
    """
    Weighted Cross-entropy Loss Function.
    """
    def __init__(self, depth_normalize, out_channel=200, loss_weight=1.0, data_type=['stereo', 'lidar'], **kwargs):
        super(WCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.depth_min = depth_normalize[0]
        self.depth_max = depth_normalize[1]
        self.bins_num = out_channel
        self.depth_min_log = torch.log10(torch.tensor(self.depth_min))
        
        self.alpha = 2 #0.2
        self.config_bins()
        self.noise_sample_ratio = 0.9 #kwargs['noise_sample_ratio'] if 'noise_sample_ratio' in kwargs else 1.0
        self.data_type = data_type
        self.eps = 1e-6
    
    def config_bins(self):
        # Modify some configs
        self.depth_bins_interval = (torch.log10(torch.tensor(self.depth_max)) - 
                                   self.depth_min_log) / self.bins_num
        bins_edges_in_log = self.depth_min_log +  self.depth_bins_interval * torch.tensor(list(range(self.bins_num)) + [self.bins_num,])
        #bins_edges_in_log = torch.from_numpy(bins_edges_in_log)
        # The boundary of each bin
        # bins_edges_in_log = np.array([self.depth_min_log + self.depth_bins_interval * (i + 0.5)
        #                         for i in range(self.bins_num)])
        bins_weight = torch.tensor([[np.exp(-self.alpha * (i - j) ** 2) for i in range(self.bins_num )]
                            for j in np.arange(self.bins_num )]).cuda()
        self.register_buffer("bins_weight", bins_weight.float(), persistent=False)
        self.register_buffer("bins_edges_in_log", bins_edges_in_log.float(), persistent=False)

    def depth_to_bins_in_log(self, depth, mask):
        """
        Discretize depth into depth bins. Predefined bins edges are in log space.
        Mark invalid padding area as bins_num + 1
        Args:
            @depth: 1-channel depth, [B, 1, h, w]
        return: depth bins [B, C, h, w]
        """
        invalid_mask = ~mask
        #depth[depth < self.depth_min] = self.depth_min
        #depth[depth > self.depth_max] = self.depth_max
        mask_lower = (depth <= self.depth_min) 
        mask_higher = (depth >= self.depth_max)
        depth_bins_log = ((torch.log10(torch.abs(depth)) - self.depth_min_log) / self.depth_bins_interval).to(torch.int)
        
        depth_bins_log[mask_lower] = 0
        depth_bins_log[mask_higher] = self.bins_num - 1
        depth_bins_log[depth_bins_log == self.bins_num] = self.bins_num - 1

        depth_bins_log[invalid_mask] = self.bins_num + 1
        return depth_bins_log
    
    def depth_to_bins(self, depth, mask, depth_edges, size_limite=(300, 300)):
        """
        Discretize depth into depth bins. Predefined bins edges are provided.
        Mark invalid padding area as bins_num + 1
        Args:
            @depth: 1-channel depth, [B, 1, h, w]
        return: depth bins [B, C, h, w]
        """ 
        def _depth_to_bins_block_(depth, mask, depth_edges):
            bins_id = torch.sum(depth_edges[:, None, None, None, :] < torch.abs(depth)[:, :, :, :, None], dim=-1)
            bins_id = bins_id - 1
            invalid_mask = ~mask
            mask_lower = (depth <= self.depth_min) 
            mask_higher = (depth >= self.depth_max)
            
            bins_id[mask_lower] = 0
            bins_id[mask_higher] = self.bins_num - 1
            bins_id[bins_id == self.bins_num] = self.bins_num - 1

            bins_id[invalid_mask] = self.bins_num + 1
            return bins_id
        _, _, H, W = depth.shape
        bins = mask.clone().long()
        h_blocks = np.ceil(H / size_limite[0]).astype(np.int)
        w_blocks = np.ceil(W/ size_limite[1]).astype(np.int)
        for i in range(h_blocks):
            for j in range(w_blocks):
                h_start = i*size_limite[0]
                h_end_proposal = (i + 1) * size_limite[0]
                h_end = h_end_proposal if h_end_proposal < H else H
                w_start = j*size_limite[1]
                w_end_proposal = (j + 1) * size_limite[1]
                w_end = w_end_proposal if w_end_proposal < W else W
                bins_ij = _depth_to_bins_block_(
                    depth[:, :, h_start:h_end, w_start:w_end], 
                    mask[:, :, h_start:h_end, w_start:w_end],
                    depth_edges
                    )
                bins[:, :, h_start:h_end, w_start:w_end] = bins_ij        
        return bins

    
    # def mask_maximum_loss(self, loss_pixels, mask):
    #     mask = mask.reshape(mask.size(0), -1)
    #     valid_pix_bt = torch.sum(mask, dim=1)
    #     mask_noise_num = (valid_pix_bt * self.noise_sample_ratio).int()
        
    #     loss_sample = []
    #     for i in range(loss_pixels.size(0)):
    #         sorted_losses, _ = torch.sort(loss_pixels[i, :][mask[i, ...]])
    #         loss_sample.append(torch.sum(sorted_losses[:mask_noise_num[i]]))
            
    #     return torch.tensor(loss_sample), mask_noise_num


    def forward(self, prediction, target, mask=None, pred_logit=None, **kwargs): #pred_logit, gt_bins, gt):
        B, _, H, W = target.shape
        if 'bins_edges' not in kwargs or kwargs['bins_edges'] is None:
            # predefined depth bins in log space
            gt_bins = self.depth_to_bins_in_log(target, mask) 
        else:
            bins_edges = kwargs['bins_edges']
            gt_bins = self.depth_to_bins(target, mask, bins_edges)

        classes_range = torch.arange(self.bins_num, device=gt_bins.device, dtype=gt_bins.dtype)
        log_pred = torch.nn.functional.log_softmax(pred_logit, 1)
        log_pred = log_pred.reshape(B, log_pred.size(1), -1).permute((0, 2, 1))
        gt_reshape = gt_bins.reshape((B, -1))[:, :, None]
        one_hot = (gt_reshape == classes_range).to(dtype=torch.float, device=pred_logit.device)
        weight = torch.matmul(one_hot, self.bins_weight)
        weight_log_pred = weight * log_pred
        loss_pixeles = - torch.sum(weight_log_pred, dim=2)

        valid_pixels = torch.sum(mask).to(dtype=torch.float, device=pred_logit.device)
        loss = torch.sum(loss_pixeles) / (valid_pixels + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'WCEL error, {loss}')
        return loss * self.loss_weight



if __name__ == '__main__':
    import cv2
    wcel = WCELoss((0.0004, 1))
    pred_depth = np.abs(np.random.random([2, 1, 480, 640]))
    pred_logit = np.random.random([2, 200, 480, 640])
    gt_depth = np.random.random([2, 1, 480, 640]) - 0.5 #np.zeros_like(pred_depth) #
    intrinsic = [[100, 100, 200, 200], [200, 200, 300, 300]]
    gt_depth = torch.tensor(np.array(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.array(pred_depth, np.float32)).cuda()
    intrinsic = torch.tensor(np.array(intrinsic, np.float32)).cuda()
    pred_logit = torch.tensor(np.array(pred_logit, np.float32)).cuda()


    mask = gt_depth > 0
    loss1 = wcel(gt_depth, gt_depth, mask, intrinsic=intrinsic, pred_logit=pred_logit)
    loss2 = wcel(gt_depth, gt_depth, mask, intrinsic=intrinsic, pred_logit=pred_logit)
    print(loss1, loss2)
