import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .depth_to_normal import Depth2Normal

# compute loss
class NormalBranchLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_type=['sfm', 'stereo', 'denselidar', 'denselidar_syn'], d2n_dataset=['ScanNetAll'], loss_fn='UG_NLL_ours', **kwargs):
        """loss_fn can be one of following:
            - L1            - L1 loss (no uncertainty)
            - L2            - L2 loss (no uncertainty)
            - AL            - Angular loss (no uncertainty)
            - NLL_vMF       - NLL of vonMF distribution
            - NLL_ours      - NLL of Angular vonMF distribution
            - UG_NLL_vMF    - NLL of vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - UG_NLL_ours   - NLL of Angular vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - NLL_ours_GRU  - NLL of Angular vonMF distribution for GRU sequence
        """
        super(NormalBranchLoss, self).__init__()
        self.loss_type = loss_fn
        if self.loss_type in ['L1', 'L2', 'AL', 'NLL_vMF', 'NLL_ours']:
            # self.loss_fn = self.forward_R
            raise NotImplementedError
        elif self.loss_type in ['UG_NLL_vMF']:
            # self.loss_fn = self.forward_UG
            raise NotImplementedError
        elif self.loss_type in ['UG_NLL_ours']:
            self.loss_fn = self.forward_UG
        elif self.loss_type in ['NLL_ours_GRU', 'NLL_ours_GRU_auxi']:
            self.loss_type = 'NLL_ours'
            self.loss_fn = self.forward_GRU
            self.loss_gamma = 0.9
            try:
                self.loss_weight_auxi = kwargs['loss_weight_auxi']
            except:
                self.loss_weight_auxi = 0.0
        else:
            raise Exception('invalid loss type')
        
        self.loss_weight = loss_weight
        self.data_type = data_type
        
        #self.d2n_dataset = d2n_dataset
        #self.depth2normal = Depth2Normal()

        
    
    def forward(self, **kwargs):
        # device = kwargs['mask'].device
        # B, _, H, W = kwargs['mask'].shape
        # pad_mask = torch.zeros_like(kwargs['mask'], device=device)
        # for b in range(B):
        #     pad = kwargs['pad'][b].squeeze()
        #     pad_mask[b, :, pad[0]:H-pad[1], pad[2]:W-pad[3]] = True

        # loss  = self.loss_fn(pad_mask=pad_mask, **kwargs)
        loss  = self.loss_fn(**kwargs)

        return loss * self.loss_weight


    def forward_GRU(self, normal_out_list, normal, target, mask, intrinsic, pad_mask=None, auxi_normal=None, **kwargs):
        n_predictions = len(normal_out_list)
        assert n_predictions >= 1
        loss = 0.0

        # device = pad_mask.device
        # batches_dataset = kwargs['dataset']
        # self.batch_with_d2n = torch.tensor([0 if batch_dataset not in self.d2n_dataset else 1 \
        #                                       for batch_dataset in batches_dataset], device=device)[:,None,None,None]

        # scale = kwargs['scale'][:, None, None].float()
        # normal_d2n, new_mask_d2n = self.depth2normal(target, intrinsic, pad_mask, scale)

        gt_normal_mask = ~torch.all(normal == 0, dim=1, keepdim=True) & mask

        if auxi_normal != None:
            auxi_normal_mask = ~gt_normal_mask

        #normal = normal * (1 -  self.batch_with_d2n) + normal_d2n * self.batch_with_d2n
        # gt_normal_mask = gt_normal_mask * (1 -  self.batch_with_d2n) + mask * new_mask_d2n * self.batch_with_d2n

        if gt_normal_mask.sum() < 10:
            if auxi_normal == None:
                for norm_out in normal_out_list:
                    loss += norm_out.sum() * 0
                return loss

        for i, norm_out in enumerate(normal_out_list):
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            curr_loss = self.forward_R(norm_out.clone(), normal, gt_normal_mask)
            if auxi_normal != None:
                auxi_loss = self.forward_R(norm_out.clone(), auxi_normal[:, :3], auxi_normal_mask)
                curr_loss = curr_loss + self.loss_weight_auxi * auxi_loss

            if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
                curr_loss = 0 * torch.sum(norm_out)
                print(f'NormalBranchLoss forward_GRU NAN error, {curr_loss}')
            
            loss += curr_loss * i_weight

        return loss

    def forward_R(self, norm_out, gt_norm, gt_norm_mask):
        pred_norm, pred_kappa = norm_out[:, 0:3, :, :], norm_out[:, 3:, :, :]

        if self.loss_type == 'L1':
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l1[gt_norm_mask])

        elif self.loss_type == 'L2':
            l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l2[gt_norm_mask])

        elif self.loss_type == 'AL':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = torch.acos(dot[valid_mask])
            loss = torch.mean(al)

        elif self.loss_type == 'NLL_vMF':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss = torch.mean(loss_pixelwise)

        elif self.loss_type == 'NLL_ours':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.5

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                             + kappa * torch.acos(dot) \
                             + torch.log(1 + torch.exp(-kappa * np.pi))
            loss = torch.mean(loss_pixelwise)

        else:
            raise Exception('invalid loss type')

        return loss


    def forward_UG(self, normal_pred_list, normal_coord_list, normal, mask, **kwargs):
        gt_normal_mask = ~torch.all(normal == 0, dim=1, keepdim=True) & mask

        # gt_norm = norms[0]
        # gt_normal_mask = (gt_norm[:, 0:1, :, :] == 0) & (gt_norm[:, 1:2, :, :] == 0) & (gt_norm[:, 2:3, :, :] == 0)
        # gt_normal_mask = ~gt_normal_mask
        loss = 0.0

        if gt_normal_mask[gt_normal_mask].numel() < 10:
            for (pred, coord) in zip(normal_pred_list, normal_coord_list):
                if pred is not None:
                    loss += pred.sum() * 0.
                if coord is not None:
                    loss += coord.sum() * 0.
            return loss

        
        for (pred, coord) in zip(normal_pred_list, normal_coord_list):
            if coord is None:
                pred = F.interpolate(pred, size=[normal.size(2), normal.size(3)], mode='bilinear', align_corners=True)
                pred_norm, pred_kappa = pred[:, 0:3, :, :], pred[:, 3:, :, :]

                # if self.loss_type == 'UG_NLL_vMF':
                #     dot = torch.cosine_similarity(pred_norm, normal, dim=1)

                #     valid_mask = normal_mask[:, 0, :, :].float() \
                #                 * (dot.detach() < 0.999).float() \
                #                 * (dot.detach() > -0.999).float()
                #     valid_mask = valid_mask > 0.5

                #     # mask
                #     dot = dot[valid_mask]
                #     kappa = pred_kappa[:, 0, :, :][valid_mask]

                #     loss_pixelwise = - torch.log(kappa) \
                #                      - (kappa * (dot - 1)) \
                #                      + torch.log(1 - torch.exp(- 2 * kappa))
                #     loss = loss + torch.mean(loss_pixelwise)

                if self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, normal, dim=1)

                    valid_mask = gt_normal_mask[:, 0, :, :].float() \
                                * (dot.detach() < 0.999).float() \
                                * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                else:
                    raise Exception

            else:
                # coord: B, 1, N, 2
                # pred: B, 4, N
                gt_norm_ = F.grid_sample(normal, coord, mode='nearest', align_corners=True)  # (B, 3, 1, N)
                gt_norm_mask_ = F.grid_sample(gt_normal_mask.float(), coord, mode='nearest', align_corners=True)  # (B, 1, 1, N)
                gt_norm_ = gt_norm_[:, :, 0, :]  # (B, 3, N)
                gt_norm_mask_ = gt_norm_mask_[:, :, 0, :] > 0.5  # (B, 1, N)

                pred_norm, pred_kappa = pred[:, 0:3, :], pred[:, 3:, :]

                # if self.loss_type == 'UG_NLL_vMF':
                #     dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                #     valid_mask = gt_norm_mask_[:, 0, :].float() \
                #                  * (dot.detach() < 0.999).float() \
                #                  * (dot.detach() > -0.999).float()
                #     valid_mask = valid_mask > 0.5

                #     dot = dot[valid_mask]
                #     kappa = pred_kappa[:, 0, :][valid_mask]

                #     loss_pixelwise = - torch.log(kappa) \
                #                      - (kappa * (dot - 1)) \
                #                      + torch.log(1 - torch.exp(- 2 * kappa))
                #     loss = loss + torch.mean(loss_pixelwise)

                if self.loss_type == 'UG_NLL_ours':
                    dot = torch.cosine_similarity(pred_norm, gt_norm_, dim=1)  # (B, N)

                    valid_mask = gt_norm_mask_[:, 0, :].float() \
                                 * (dot.detach() < 0.999).float() \
                                 * (dot.detach() > -0.999).float()
                    valid_mask = valid_mask > 0.5

                    dot = dot[valid_mask]
                    kappa = pred_kappa[:, 0, :][valid_mask]

                    loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                                     + kappa * torch.acos(dot) \
                                     + torch.log(1 + torch.exp(-kappa * np.pi))
                    loss = loss + torch.mean(loss_pixelwise)

                else:
                    raise Exception
        return loss




# confidence-guided sampling
@torch.no_grad()
def sample_points(init_normal, confidence_map, gt_norm_mask, sampling_ratio, beta=1):
    device = init_normal.device
    B, _, H, W = init_normal.shape
    N = int(sampling_ratio * H * W)
    beta = beta

    # confidence map
    # confidence_map = init_normal[:, 3, :, :]  # B, H, W

    # gt_invalid_mask (B, H, W)
    if gt_norm_mask is not None:
        gt_invalid_mask = F.interpolate(gt_norm_mask.float(), size=[H, W], mode='nearest')
        gt_invalid_mask = gt_invalid_mask < 0.5
        confidence_map[gt_invalid_mask] = -1e4

    # (B, H*W)
    _, idx = confidence_map.view(B, -1).sort(1, descending=True)

    # confidence sampling
    if int(beta * N) > 0:
        importance = idx[:, :int(beta * N)]    # B, beta*N

        # remaining
        remaining = idx[:, int(beta * N):]     # B, H*W - beta*N

        # coverage
        num_coverage = N - int(beta * N)

        if num_coverage <= 0:
            samples = importance
        else:
            coverage_list = []
            for i in range(B):
                idx_c = torch.randperm(remaining.size()[1])    # shuffles "H*W - beta*N"
                coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))     # 1, N-beta*N
            coverage = torch.cat(coverage_list, dim=0)                                      # B, N-beta*N
            samples = torch.cat((importance, coverage), dim=1)                              # B, N

    else:
        # remaining
        remaining = idx[:, :]  # B, H*W

        # coverage
        num_coverage = N

        coverage_list = []
        for i in range(B):
            idx_c = torch.randperm(remaining.size()[1])  # shuffles "H*W - beta*N"
            coverage_list.append(remaining[i, :][idx_c[:num_coverage]].view(1, -1))  # 1, N-beta*N
        coverage = torch.cat(coverage_list, dim=0)  # B, N-beta*N
        samples = coverage

    # point coordinates
    rows_int = samples // W         # 0 for first row, H-1 for last row
    # rows_float = rows_int / float(H-1)         # 0 to 1.0
    # rows_float = (rows_float * 2.0) - 1.0       # -1.0 to 1.0

    cols_int = samples % W          # 0 for first column, W-1 for last column
    # cols_float = cols_int / float(W-1)         # 0 to 1.0
    # cols_float = (cols_float * 2.0) - 1.0       # -1.0 to 1.0

    # point_coords = torch.zeros(B, 1, N, 2)
    # point_coords[:, 0, :, 0] = cols_float             # x coord
    # point_coords[:, 0, :, 1] = rows_float             # y coord
    # point_coords = point_coords.to(device)
    # return point_coords, rows_int, cols_int

    sample_mask = torch.zeros((B,1,H,W), dtype=torch.bool, device=device)
    for i in range(B):
        sample_mask[i, :, rows_int[i,:], cols_int[i,:]] = True
    return sample_mask

# depth-normal consistency loss
class DeNoConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, data_type=['stereo', 'lidar', 'denselidar', 'denselidar_nometric', 'denselidar_syn'], loss_fn='NLL_ours', \
                 sky_id=142, scale=1, norm_dataset=['Taskonomy', 'Matterport3D', 'Replica', 'Hypersim', 'NYU'], no_sky_dataset=['BigData', 'DIODE', 'Completion', 'Matterport3D'], disable_dataset=[], depth_detach=False, **kwargs):
        """loss_fn can be one of following:
            - L1            - L1 loss (no uncertainty)
            - L2            - L2 loss (no uncertainty)
            - AL            - Angular loss (no uncertainty)
            - NLL_vMF       - NLL of vonMF distribution
            - NLL_ours      - NLL of Angular vonMF distribution
            - UG_NLL_vMF    - NLL of vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - UG_NLL_ours   - NLL of Angular vonMF distribution (+ pixel-wise MLP + uncertainty-guided sampling)
            - NLL_ours_GRU  - NLL of Angular vonMF distribution for GRU sequence
            - CEL           - cosine embedding loss
            - CEL_GRU
        """
        super(DeNoConsistencyLoss, self).__init__()
        self.loss_type = loss_fn
        if self.loss_type in ['L1', 'L2', 'NLL_vMF']:
            # self.loss_fn = self.forward_R
            raise NotImplementedError
        elif self.loss_type in ['UG_NLL_vMF']:
            # self.loss_fn = self.forward_UG
            raise NotImplementedError
        elif self.loss_type in ['UG_NLL_ours']:
            # self.loss_fn = self.forward_UG
            raise NotImplementedError
        elif self.loss_type in ['NLL_ours']:
            self.loss_fn = self.forward_J # confidence Joint optimization
            self.loss_gamma = 0.9
        elif self.loss_type in ['AL', 'CEL', 'CEL_L2']:
            self.loss_fn = self.forward_S # confidence Sample
        elif self.loss_type in ['CEL_GRU']:
            self.loss_fn = self.forward_S_GRU # gru
            self.loss_gamma = 0.9
        elif 'Search' in self.loss_type:
            self.loss_fn = self.forward_S_Search
        else:
            raise Exception('invalid loss type')
        
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.sky_id = sky_id

        # For datasets without surface normal gt, enhance its weight (decrease the weight of the dataset with gt).
        self.nonorm_data_scale = scale 
        self.norm_dataset = norm_dataset
        self.no_sky_dataset = no_sky_dataset
        self.disable_dataset = disable_dataset

        self.depth_detach = depth_detach
        self.depth2normal = Depth2Normal()
    
    def forward(self, **kwargs):
        device = kwargs['mask'].device

        batches_dataset = kwargs['dataset']
        self.batch_with_norm = torch.tensor([self.nonorm_data_scale if batch_dataset not in self.norm_dataset else 1 \
                                              for batch_dataset in batches_dataset], device=device)[:,None,None,None]

        self.batch_enabled= torch.tensor([1 if batch_dataset not in  self.disable_dataset  else 0 \
                                              for batch_dataset in batches_dataset], device=device, dtype=torch.bool)[:,None,None,None]
        self.batch_with_norm = self.batch_with_norm * self.batch_enabled


        self.batch_with_norm_sky = torch.tensor([1 if batch_dataset not in  self.no_sky_dataset  else 0 \
                                              for batch_dataset in batches_dataset], device=device, dtype=torch.bool)[:,None,None,None]

        B, _, H, W = kwargs['mask'].shape
        pad_mask = torch.zeros_like(kwargs['mask'], device=device)
        for b in range(B):
            pad = kwargs['pad'][b].squeeze()
            pad_mask[b, :, pad[0]:H-pad[1], pad[2]:W-pad[3]] = True

        loss  = self.loss_fn(pad_mask=pad_mask, **kwargs)
        return loss * self.loss_weight


    def forward_J(self, prediction, confidence, normal_out_list, intrinsic, pad_mask, sem_mask=None, **kwargs):
        prediction_normal = normal_out_list[-1].clone()

        # get normal from depth-prediction 
        normal, new_mask = self.depth2normal(prediction.detach() if self.depth_detach else prediction, intrinsic, pad_mask)
        # mask sky
        sky_mask = sem_mask != self.sky_id
        new_mask = new_mask & sky_mask
        # normal = normal * (~sky_mask)
        # normal[:,1:2,:,:][sky_mask] = 1
        # confidence sampling (sample good depth -> good normal -> to )
        sample_mask_d = sample_points(prediction, confidence, new_mask, sampling_ratio=0.7)

        # all mask
        normal_mask = ~torch.all(normal == 0, dim=1, keepdim=True) & new_mask & sample_mask_d
        if normal_mask.sum() < 10:
            return 0 * prediction_normal.sum()

        loss = self.forward_R(prediction_normal, normal, normal_mask)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction_normal)
            print(f'NormalBranchLoss forward_GRU NAN error, {loss}')

        return loss

    #def forward_S(self, prediction, confidence, normal_out_list, intrinsic, pad_mask, sem_mask=None, **kwargs):
    def forward_S(self, prediction, confidence, intrinsic, pad_mask, normal_pred=None, sem_mask=None, target=None, is_initial_pair=False, **kwargs):
        
        if normal_pred is None:
            prediction_normal = kwargs['normal_out_list'][-1]
        else:
            prediction_normal = normal_pred

        # get normal from depth-prediction 
        #try:
        scale = kwargs['scale'][:, None, None].float()
        #except:
            #scale = 1.0
        normal, new_mask = self.depth2normal(prediction.detach() if self.depth_detach else prediction, intrinsic, pad_mask, scale)

        sky_mask = sem_mask != self.sky_id
        if target != None:
            sampling_ratio = 0.7
            target_mask = (target > 0) 
            if is_initial_pair == False:
                pass
            # mask sky
            else:
                sky_mask = torch.nn.functional.interpolate(sky_mask.float(), scale_factor=0.25).bool()
                target_mask = torch.nn.functional.interpolate(target_mask.float(), scale_factor=0.25).bool()
                new_mask = new_mask & ((sky_mask & self.batch_with_norm_sky) | target_mask)
        else:
            new_mask =  torch.ones_like(prediction).bool()
            sampling_ratio = 0.5

        # normal = normal * (~sky_mask)
        # normal[:,1:2,:,:][sky_mask] = 1

        # dual sampling
        confidence_normal = prediction_normal[:, 3:, :, :]
        sample_mask_n = sample_points(prediction_normal, confidence_normal, new_mask, sampling_ratio=sampling_ratio)
        sample_mask_d = sample_points(prediction, confidence, new_mask, sampling_ratio=sampling_ratio)
        conf_mask = confidence > 0.5

        # all mask
        normal_mask = ~torch.all(normal == 0, dim=1, keepdim=True) & new_mask & sample_mask_n & sample_mask_d & conf_mask
        if normal_mask.sum() < 10:
            return 0 * prediction_normal.sum() 

        loss = self.forward_R(prediction_normal, normal, normal_mask)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction_normal)
            print(f'NormalBranchLoss forward_GRU NAN error, {loss}')

        return loss

    def forward_S_GRU(self, predictions_list, confidence_list, normal_out_list, intrinsic, pad_mask, sem_mask, target, low_resolution_init, **kwargs):
        n_predictions = len(normal_out_list)
        assert n_predictions >= 1
        loss = 0.0

        for i, (norm, conf, depth) in enumerate(zip(normal_out_list, confidence_list, predictions_list)):
            # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
            adjusted_loss_gamma = self.loss_gamma**(15/(n_predictions - 1))
            i_weight = adjusted_loss_gamma**(n_predictions - i - 1)

            if i == 0:
                is_initial_pair = True
                new_intrinsic =  torch.cat((intrinsic[:, :2, :]/4, intrinsic[:, 2:3, :]), dim=1)
                curr_loss = self.forward_S(low_resolution_init[0], low_resolution_init[1], new_intrinsic, torch.nn.functional.interpolate(pad_mask.float(), scale_factor=0.25).bool(), low_resolution_init[2], sem_mask, target, is_initial_pair, scale=kwargs['scale'])
            else:
                is_initial_pair = False
                curr_loss = self.forward_S(depth, conf, intrinsic, pad_mask, norm, sem_mask, target, is_initial_pair, scale=kwargs['scale'])
            
            if torch.isnan(curr_loss).item() | torch.isinf(curr_loss).item():
                curr_loss = 0 * torch.sum(norm)
                print(f'NormalBranchLoss forward_GRU NAN error, {curr_loss}')
            
            loss += curr_loss * i_weight

        return loss


    def forward_R(self, norm_out, gt_norm, gt_norm_mask, pred_kappa=None):
        pred_norm = norm_out[:, 0:3, :, :]
        if pred_kappa is None:
            pred_kappa = norm_out[:, 3:, :, :]

        if self.loss_type == 'L1':
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l1[gt_norm_mask])

        elif self.loss_type == 'L2' or self.loss_type == 'CEL_L2':
            l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l2[gt_norm_mask])

        elif self.loss_type == 'AL':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = torch.acos(dot * valid_mask)
            al = al * self.batch_with_norm[:, 0, :, :]
            loss = torch.mean(al)
        
        elif self.loss_type == 'CEL' or self.loss_type == 'CEL_GRU':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = 1 - dot * valid_mask
            al = al * self.batch_with_norm[:, 0, :, :]
            loss = torch.mean(al)

        elif self.loss_type == 'NLL_vMF':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss = torch.mean(loss_pixelwise)

        elif self.loss_type == 'NLL_ours':
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.5

            dot = dot * valid_mask
            kappa = pred_kappa[:, 0, :, :] * valid_mask

            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                             + kappa * torch.acos(dot) \
                             + torch.log(1 + torch.exp(-kappa * np.pi))
            loss_pixelwise = loss_pixelwise * self.batch_with_norm[:, 0, :, :]
            loss = torch.mean(loss_pixelwise)

        else:
            raise Exception('invalid loss type')

        return loss

    def forward_S_Search(self, prediction, confidence, intrinsic, pad_mask, normal_pred=None, sem_mask=None, target=None, is_initial_pair=False, **kwargs):
        
        if normal_pred is None:
            prediction_normal = kwargs['normal_out_list'][-1]
        else:
            prediction_normal = normal_pred

        # get normal from depth-prediction 
        scale = kwargs['scale'][:, None, None].float()
        candidate_scale = kwargs['candidate_scale'][:, None, None, None].float() 
        normal, new_mask = self.depth2normal(prediction.detach() if self.depth_detach else prediction, intrinsic, pad_mask, scale)

        sky_mask = sem_mask != self.sky_id
        if target != None:
            sampling_ratio = 0.7
            target_mask = (target > 0) 
            if is_initial_pair == False:
                pass
            # mask sky
            else:
                sky_mask = torch.nn.functional.interpolate(sky_mask.float(), scale_factor=0.25).bool()
                target_mask = torch.nn.functional.interpolate(target_mask.float(), scale_factor=0.25).bool()
                new_mask = new_mask & ((sky_mask & self.batch_with_norm_sky) | target_mask)
        else:
            new_mask =  torch.ones_like(prediction).bool()
            sampling_ratio = 0.5

        # normal = normal * (~sky_mask)
        # normal[:,1:2,:,:][sky_mask] = 1

        # dual sampling
        confidence_normal = prediction_normal[:, 3:, :, :]
        sample_mask_n = sample_points(prediction_normal, confidence_normal, new_mask, sampling_ratio=sampling_ratio)
        sample_mask_d = sample_points(prediction, confidence, new_mask, sampling_ratio=sampling_ratio)
        conf_mask = confidence > 0.5

        # all mask
        normal_mask = ~torch.all(normal == 0, dim=1, keepdim=True) & new_mask & sample_mask_n & sample_mask_d & conf_mask
        if normal_mask.sum() < 10:
            return 0 * prediction_normal.sum() 

        prediction_normal = torch.cat((prediction_normal[:,:2]*torch.ones_like(candidate_scale), prediction_normal[:,2:3]*candidate_scale, prediction_normal[:,3:4]*torch.ones_like(candidate_scale)), dim=1)
        
        norm_x = prediction_normal[:,0:1]
        norm_y = prediction_normal[:,1:2]
        norm_z = prediction_normal[:,2:3]
        
        prediction_normal[:,:3] = prediction_normal[:,:3] / (torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10)
        
        loss = self.forward_R_Search(prediction_normal, normal, normal_mask)
        #if torch.isnan(loss).item() | torch.isinf(loss).item():
            #loss = 0 * torch.sum(prediction_normal)
            #print(f'NormalBranchLoss forward_GRU NAN error, {loss}')

        return loss


    def forward_R_Search(self, norm_out, gt_norm, gt_norm_mask, pred_kappa=None):
        pred_norm = norm_out[:, 0:3, :, :]
        if pred_kappa is None:
            pred_kappa = norm_out[:, 3:, :, :]

        if 'L1' in self.loss_type:
            l1 = torch.sum(torch.abs(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l1*gt_norm_mask, dim=[1, 2, 3])

        elif 'L2' in self.loss_type:
            l2 = torch.sum(torch.square(gt_norm - pred_norm), dim=1, keepdim=True)
            loss = torch.mean(l2*gt_norm_mask, dim=[1, 2, 3])

        elif 'AL' in self.loss_type:
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = torch.acos(dot * valid_mask)
            loss = torch.mean(al, dim=[1, 2])

        elif 'CEL' in self.loss_type:
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            al = 1 - dot * valid_mask
            loss = torch.mean(al, dim=[1, 2])

        elif 'NLL_vMF' in self.loss_type:
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.0

            dot = dot[valid_mask]
            kappa = pred_kappa[:, 0, :, :][valid_mask]

            loss_pixelwise = - torch.log(kappa) \
                             - (kappa * (dot - 1)) \
                             + torch.log(1 - torch.exp(- 2 * kappa))
            loss = torch.mean(loss_pixelwise, dim=[1, 2])

        elif 'NLL_ours' in self.loss_type:
            dot = torch.cosine_similarity(pred_norm, gt_norm, dim=1)

            valid_mask = gt_norm_mask[:, 0, :, :].float() \
                         * (dot.detach() < 0.999).float() \
                         * (dot.detach() > -0.999).float()
            valid_mask = valid_mask > 0.5

            dot = dot * valid_mask
            kappa = pred_kappa[:, 0, :, :] * valid_mask

            loss_pixelwise = - torch.log(torch.square(kappa) + 1) \
                             + kappa * torch.acos(dot) \
                             + torch.log(1 + torch.exp(-kappa * np.pi))
            loss = torch.mean(loss_pixelwise, dim=[1, 2])

        else:
            raise Exception('invalid loss type')

        return loss