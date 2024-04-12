import torch
import torch.nn as nn

class HDNLoss(nn.Module):
    """
    Hieratical depth normalization loss.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1, grid=3, data_type=['sfm', 'stereo', 'lidar'], **kwargs):
        super(HDNLoss, self).__init__()
        self.loss_weight = loss_weight
        self.grid = grid
        self.data_type = data_type
    
    def get_hierachy_masks(self, grid, depth_gt, mask_valid):

        batch_map_grid = []
        for mask_index in range(depth_gt.shape[0]):
            depth_map = depth_gt[mask_index]
            valid_map = mask_valid[mask_index]

            # print (depth_map[valid_map].view(-1).shape)
            if depth_map[valid_map].numel() == 0:
                map_grid_list = [valid_map for _ in range(2 ** (grid) - 1)]
            else:
                valid_values = depth_map[valid_map]

                max_d = valid_values.max()
                min_d = valid_values.min()

                anchor_power = [(1 / 2) ** (i) for i in range(grid)]
                anchor_power.reverse()

                map_grid_list = []
                for anchor in anchor_power:
                    # range
                    for i in range(int(1 / anchor)):
                        mask_new = (depth_map >= min_d + (max_d - min_d) * i * anchor) & (
                                    depth_map < min_d + (max_d - min_d) * (i + 1) * anchor+1e-30)
                        # print (f'[{i*anchor},{(i+1)*anchor}]')
                        mask_new = mask_new & valid_map
                        map_grid_list.append(mask_new)
            map_grid_list = torch.stack(map_grid_list, dim=0)
            batch_map_grid.append(map_grid_list)
        batch_map_grid = torch.stack(batch_map_grid, dim=1)
        return batch_map_grid
    
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + 1e-8)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + 1e-8)

        loss = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid) / (torch.sum(mask_valid) + 1e-8)
        return pred_trans, gt_trans, loss

    def forward(self, prediction, target, mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        hierachy_masks = self.get_hierachy_masks(self.grid, target, mask)
        hierachy_masks_shape = hierachy_masks.reshape(-1, C, H, W)    
        prediction_hie = prediction.unsqueeze(0).repeat(hierachy_masks.shape[0], 1, 1, 1, 1).reshape(-1, C, H, W)     

        target_hie = target.unsqueeze(0).repeat(hierachy_masks.shape[0], 1, 1, 1, 1).reshape(-1, C, H, W)

        #_, _, loss = self.ssi_mae(prediction, target, mask)
        _, _, loss = self.ssi_mae(prediction_hie, target_hie, hierachy_masks_shape)
        return loss * self.loss_weight
    
if __name__ == '__main__':
    ssil = HDNLoss()
    pred = torch.rand((2, 1, 256, 256)).cuda()
    gt = torch.rand((2, 1, 256, 256)).cuda()#torch.zeros_like(pred).cuda() #
    gt[:, :, 100:256, 0:100] = -1
    mask = gt > 0
    out = ssil(pred, gt, mask)
    print(out)
