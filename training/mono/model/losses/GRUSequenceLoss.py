import torch
import torch.nn as nn

class GRUSequenceLoss(nn.Module):
    """
    Loss function defined over sequence of depth predictions
    """
    def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'stereo', 'denselidar_syn'], loss_gamma=0.9, silog=False, stereo_sup=0.001, stereo_dataset=['KITTI', 'NYU'], **kwargs):
        super(GRUSequenceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
        self.loss_gamma = loss_gamma
        self.silog = silog
        self.variance_focus = 0.5
        self.stereo_sup = stereo_sup
        self.stereo_dataset = stereo_dataset

        # assert stereo_mode in ['stereo', 'self_sup']
        # self.stereo_mode = stereo_mode
        # self.stereo_max = stereo_max

    def silog_loss(self, prediction, target, mask):
        mask = mask & (prediction > 0.01) & (target > 0.01)
        d = torch.log(prediction[mask]) - torch.log(target[mask])
        # d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
        # d_mean = torch.sum(d) / (d.numel() + self.eps)
        # loss = d_square_mean - self.variance_focus * (d_mean ** 2)
        loss = torch.sum(torch.abs(d)) / (d.numel() + self.eps)
        print("new log l1 loss")
        return loss 
    
    def conf_loss(self, confidence, prediction, target, mask):
        conf_mask = torch.abs(target - prediction) < target
        conf_mask = conf_mask & mask
        gt_confidence = (1 - torch.abs((prediction - target) / target)) * conf_mask
        loss = torch.sum(torch.abs(confidence - gt_confidence) * conf_mask) / (torch.sum(conf_mask) + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            print(f'GRUSequenceLoss-confidence NAN error, {loss}')
            loss = 0 * torch.sum(confidence)
        return loss

    def forward(self, predictions_list, target, stereo_depth, confidence_list=None, mask=None, **kwargs):
        device = target.device

        batches_dataset = kwargs['dataset']
        self.batch_with_stereo = torch.tensor([1 if batch_dataset in self.stereo_dataset else 0 \
                                              for batch_dataset in batches_dataset], device=device)[:,None,None,None]
        
        n_predictions = len(predictions_list)
        assert n_predictions >= 1
        loss = 0.0

        for i, prediction in enumerate(predictions_list):
            # if self.stereo_mode == 'self_sup' and self.stereo_sup > 1e-8:
            #     B, C, H, W = target.shape
            #     prediction_nan = prediction.clone().detach()
            #     target_nan = target.clone()
            #     prediction_nan[~mask] = float('nan')
            #     target_nan[~mask] = float('nan')
            #     gt_median = target_nan.reshape((B, C,-1)).nanmedian(2)[0][:, :, None, None]
                
            #     pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2)[0][:, :, None, None]
            #     scale = gt_median / (pred_median + 1e-8)

            #     stereo_depth = (0.0 * stereo_depth + scale * prediction * (prediction < (self.stereo_max - 1)) + \
            #         prediction * (prediction > (self.stereo_max - 1))).detach()
            
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            # depth L1 loss
            if self.silog and mask.sum() > 0:
                curr_loss = self.silog_loss(prediction, target, mask)
            else:
                diff = torch.abs(prediction - target) * mask
                #diff = diff + diff * diff * 1.0
                curr_loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
            if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
                print(f'GRUSequenceLoss-depth NAN error, {curr_loss}')
                curr_loss = 0 * torch.sum(prediction)

            # confidence L1 loss
            conf_loss = 0
            if confidence_list is not None:
                conf_loss = self.conf_loss(confidence_list[i], prediction, target, mask)

            # stereo depth loss
            mask_stereo = 1 + torch.nn.functional.max_pool2d(\
                - torch.nn.functional.max_pool2d(mask * 1.0, 3, stride=1, padding=1, dilation=1), 3, stride=1, padding=1, dilation=1)

            stereo_diff = torch.abs(prediction - stereo_depth) * mask_stereo
            #stereo_diff = stereo_diff + stereo_diff * stereo_diff * 1.0
            stereo_depth_loss = torch.sum(self.batch_with_stereo * stereo_diff * mask_stereo) / (torch.sum(mask_stereo) + self.eps)
            stereo_depth_loss = self.stereo_sup * stereo_depth_loss

            loss += (conf_loss + curr_loss + stereo_depth_loss) * i_weight
            #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight

# import torch
# import torch.nn as nn

# class GRUSequenceLoss(nn.Module):
#     """
#     Loss function defined over sequence of depth predictions
#     """
#     def __init__(self, loss_weight=1, data_type=['lidar', 'denselidar', 'stereo', 'denselidar_syn'], loss_gamma=0.9, silog=False, stereo_sup=0.001, stereo_dataset=['BigData'], **kwargs):
#         super(GRUSequenceLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.data_type = data_type
#         self.eps = 1e-6
#         self.loss_gamma = loss_gamma
#         self.silog = silog
#         self.variance_focus = 0.5
#         self.stereo_sup = stereo_sup
#         self.stereo_dataset = stereo_dataset

#     def silog_loss(self, prediction, target, mask):
#         mask = mask & (prediction > 0.01) & (target > 0.01)
#         d = torch.log(prediction[mask]) - torch.log(target[mask])
#         # d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
#         # d_mean = torch.sum(d) / (d.numel() + self.eps)
#         # loss = d_square_mean - self.variance_focus * (d_mean ** 2)
#         loss = torch.sum(torch.abs(d)) / (d.numel() + self.eps)
#         print("new log l1 loss")
#         return loss 
    
#     def conf_loss(self, confidence, prediction, target, mask):
#         conf_mask = torch.abs(target - prediction) < target
#         conf_mask = conf_mask & mask
#         gt_confidence = (1 - torch.abs((prediction - target) / target)) * conf_mask
#         loss = torch.sum(torch.abs(confidence - gt_confidence) * conf_mask) / (torch.sum(conf_mask) + self.eps)
#         if torch.isnan(loss).item() | torch.isinf(loss).item():
#             print(f'GRUSequenceLoss-confidence NAN error, {loss}')
#             loss = 0 * torch.sum(confidence)
#         return loss

#     def forward(self, predictions_list, target, stereo_depth, confidence_list=None, mask=None, **kwargs):
#         device = target.device

#         batches_dataset = kwargs['dataset']
#         self.batch_with_stereo = torch.tensor([1 if batch_dataset in self.stereo_dataset else 0 \
#                                               for batch_dataset in batches_dataset], device=device)[:,None,None,None]
        
#         n_predictions = len(predictions_list)
#         assert n_predictions >= 1
#         loss = 0.0

#         for i, prediction in enumerate(predictions_list):
#             # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
#             adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
#             i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

#             # depth L1 loss
#             if self.silog and mask.sum() > 0:
#                 curr_loss = self.silog_loss(prediction, target, mask)
#             else:
#                 diff = torch.abs(prediction - target) * mask
#                 curr_loss = torch.sum(diff) / (torch.sum(mask) + self.eps)
#             if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
#                 print(f'GRUSequenceLoss-depth NAN error, {curr_loss}')
#                 curr_loss = 0 * torch.sum(prediction)

#             # confidence L1 loss
#             conf_loss = 0
#             if confidence_list is not None:
#                 conf_loss = self.conf_loss(confidence_list[i], prediction, target, mask)

#             # stereo depth loss
#             mask_stereo = 1 + torch.nn.functional.max_pool2d(\
#                 - torch.nn.functional.max_pool2d(mask * 1.0, 5, stride=1, padding=2, dilation=1), 5, stride=1, padding=2, dilation=1)

#             stereo_diff = torch.abs(prediction - stereo_depth) * mask_stereo
#             stereo_depth_loss = torch.sum(self.batch_with_stereo * stereo_diff * mask_stereo) / (torch.sum(mask_stereo) + self.eps)
#             stereo_depth_loss = self.stereo_sup * stereo_depth_loss

#             loss += (conf_loss + curr_loss + stereo_depth_loss) * i_weight
#             #raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
#         return loss * self.loss_weight