import torch
import torch.nn as nn
import numpy as np
#from numba import jit

class HDSNRandomLoss(nn.Module):
    """
    Hieratical depth spatial normalization loss.
    Replace the original grid masks with the random created masks.
    loss = MAE((d-median(d)/s - (d'-median(d'))/s'), s = mean(d- median(d))
    """
    def __init__(self, loss_weight=1.0, random_num=32, data_type=['sfm', 'stereo', 'lidar', 'denselidar', 'denselidar_nometric','denselidar_syn'], disable_dataset=['MapillaryPSD'], sky_id=142, batch_limit=8, **kwargs):
        super(HDSNRandomLoss, self).__init__()
        self.loss_weight = loss_weight
        self.random_num = random_num
        self.data_type = data_type
        self.sky_id = sky_id
        self.batch_limit = batch_limit
        self.eps = 1e-6
        self.disable_dataset = disable_dataset

    def get_random_masks_for_batch(self, image_size: list)-> torch.Tensor:
        height, width = image_size
        crop_h_min = int(0.125 * height)
        crop_h_max = int(0.5 * height)
        crop_w_min = int(0.125 * width)
        crop_w_max = int(0.5 * width)
        h_max = height - crop_h_min
        w_max = width - crop_w_min
        crop_height = np.random.choice(np.arange(crop_h_min, crop_h_max), self.random_num, replace=False)
        crop_width = np.random.choice(np.arange(crop_w_min, crop_w_max), self.random_num, replace=False)
        crop_y = np.random.choice(h_max, self.random_num, replace=False)
        crop_x = np.random.choice(w_max, self.random_num, replace=False)
        crop_y_end = crop_height + crop_y
        crop_y_end[crop_y_end>=height] = height
        crop_x_end = crop_width + crop_x
        crop_x_end[crop_x_end>=width] = width

        mask_new = torch.zeros((self.random_num,  height, width), dtype=torch.bool, device="cuda") #.cuda() #[N, H, W]
        for i in range(self.random_num):
           mask_new[i, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]] = True

        return mask_new
        #return crop_y, crop_y_end, crop_x, crop_x_end
    
    def reorder_sem_masks(self, sem_label):
        # reorder the semantic mask of a batch
        assert sem_label.ndim == 3
        semantic_ids = torch.unique(sem_label[(sem_label>0) & (sem_label != self.sky_id)])
        sem_masks = [sem_label == id for id in semantic_ids]
        if len(sem_masks) == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out

        sem_masks = torch.cat(sem_masks, dim=0)
        mask_batch = torch.sum(sem_masks.reshape(sem_masks.shape[0], -1), dim=1) > 500
        sem_masks = sem_masks[mask_batch]
        if sem_masks.shape[0] > self.random_num:
            balance_samples = np.random.choice(sem_masks.shape[0], self.random_num, replace=False)
            sem_masks = sem_masks[balance_samples, ...]
        
        if sem_masks.shape[0] == 0:
            # no valid semantic labels
            out = sem_label > 0
            return out

        if sem_masks.ndim == 2:
            sem_masks = sem_masks[None, :, :]
        return sem_masks
  
    def ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        prediction_nan = prediction.clone().detach()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + 1e-10
        valid_pixs = valid_pixs[:, :, :, None]

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) ).reshape((B, C, -1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        gt_trans = (target - gt_median) / (gt_s + self.eps)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median)).reshape((B, C, -1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None] / valid_pixs
        pred_trans = (prediction - pred_median) / (pred_s + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        return loss_sum
    
    def conditional_ssi_mae(self, prediction, target, mask_valid):
        B, C, H, W = target.shape
        conditional_rank_ids = np.random.choice(B, B, replace=False)

        prediction_nan = prediction.clone()
        target_nan = target.clone()
        prediction_nan[~mask_valid] = float('nan')
        target_nan[~mask_valid] = float('nan')

        valid_pixs = mask_valid.reshape((B, C,-1)).sum(dim=2, keepdims=True) + self.eps
        valid_pixs = valid_pixs[:, :, :, None].contiguous()

        gt_median = target_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        gt_median[torch.isnan(gt_median)] = 0
        gt_diff = (torch.abs(target - gt_median) * mask_valid).reshape((B, C,-1))
        gt_s = gt_diff.sum(dim=2)[:, :, None, None].contiguous() / valid_pixs

        # in case some batches have no valid pixels
        gt_s_small_mask = gt_s < (torch.mean(gt_s)*0.1)
        gt_s[gt_s_small_mask] = torch.mean(gt_s)
        gt_trans = (target - gt_median[conditional_rank_ids]) / (gt_s[conditional_rank_ids] + self.eps)

        pred_median = prediction_nan.reshape((B, C,-1)).nanmedian(2, keepdims=True)[0].unsqueeze(-1) # [b,c,h,w]
        pred_median[torch.isnan(pred_median)] = 0
        pred_diff = (torch.abs(prediction - pred_median) * mask_valid).reshape((B, C,-1))
        pred_s = pred_diff.sum(dim=2)[:, :, None, None].contiguous() / valid_pixs
        pred_s[gt_s_small_mask] = torch.mean(pred_s)
        pred_trans = (prediction - pred_median[conditional_rank_ids]) / (pred_s[conditional_rank_ids] + self.eps)

        loss_sum = torch.sum(torch.abs(gt_trans - pred_trans)*mask_valid)
        # print(torch.abs(gt_trans - pred_trans)[mask_valid])
        return loss_sum


    def forward(self, prediction, target, mask=None, sem_mask=None, **kwargs):
        """
        Calculate loss.
        """
        B, C, H, W = target.shape
        
        loss = 0.0
        valid_pix = 0.0

        device = target.device
        
        batches_dataset = kwargs['dataset']
        self.batch_valid = torch.tensor([1 if batch_dataset not in self.disable_dataset else 0 \
            for batch_dataset in batches_dataset], device=device)[:,None,None,None]

        batch_limit = self.batch_limit
        
        random_sample_masks = self.get_random_masks_for_batch((H, W)) # [N, H, W]
        for i in range(B):
            # each batch
            mask_i = mask[i, ...] #[1, H, W]
            if self.batch_valid[i, ...] < 0.5:
                loss += 0 * torch.sum(prediction[i, ...])
                valid_pix += 0 * torch.sum(mask_i)
                continue

            pred_i = prediction[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)
            target_i = target[i, ...].unsqueeze(0).repeat(batch_limit, 1, 1, 1)

            # get semantic masks
            sem_label_i = sem_mask[i, ...] if sem_mask is not None else None
            if sem_label_i is not None:
                sem_masks = self.reorder_sem_masks(sem_label_i) # [N, H, W]
                random_sem_masks = torch.cat([random_sample_masks, sem_masks], dim=0)
            else:
                random_sem_masks = random_sample_masks
            #random_sem_masks = random_sample_masks


            sampled_masks_num = random_sem_masks.shape[0]
            loops = int(np.ceil(sampled_masks_num / batch_limit))
            conditional_rank_ids = np.random.choice(sampled_masks_num, sampled_masks_num, replace=False)

            for j in range(loops):
                mask_random_sem_loopi = random_sem_masks[j*batch_limit:(j+1)*batch_limit, ...]
                mask_sample = (mask_i & mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                loss += self.ssi_mae(
                    prediction=pred_i[:mask_sample.shape[0], ...], 
                    target=target_i[:mask_sample.shape[0], ...], 
                    mask_valid=mask_sample)
                valid_pix += torch.sum(mask_sample)

                # conditional ssi loss
                # rerank_mask_random_sem_loopi = random_sem_masks[conditional_rank_ids, ...][j*batch_limit:(j+1)*batch_limit, ...]
                # rerank_mask_sample = (mask_i & rerank_mask_random_sem_loopi).unsqueeze(1) # [N, 1, H, W]
                # loss_cond = self.conditional_ssi_mae(
                #     prediction=pred_i[:rerank_mask_sample.shape[0], ...], 
                #     target=target_i[:rerank_mask_sample.shape[0], ...], 
                #     mask_valid=rerank_mask_sample)
                # print(loss_cond / (torch.sum(rerank_mask_sample) + 1e-10), loss_cond, torch.sum(rerank_mask_sample))
                # loss += loss_cond
                # valid_pix += torch.sum(rerank_mask_sample)

        # crop_y, crop_y_end, crop_x, crop_x_end = self.get_random_masks_for_batch((H, W)) # [N,]
        # for j in range(B):
        #     for i in range(self.random_num):
        #         mask_crop = mask[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...] #[1, 1, crop_h, crop_w]
        #         target_crop = target[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...]
        #         pred_crop = prediction[j, :, crop_y[i]:crop_y_end[i], crop_x[i]:crop_x_end[i]][None, ...]
        #         loss += self.ssi_mae(prediction=pred_crop, target=target_crop, mask_valid=mask_crop)
        #         valid_pix += torch.sum(mask_crop)
        
        # the whole image
        mask = mask * self.batch_valid.bool()
        loss += self.ssi_mae(
                    prediction=prediction, 
                    target=target, 
                    mask_valid=mask)
        valid_pix += torch.sum(mask)
        loss = loss / (valid_pix + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'HDSNL NAN error, {loss}, valid pix: {valid_pix}')
        return loss * self.loss_weight
    
if __name__ == '__main__':
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    ssil = HDSNRandomLoss()
    pred = torch.rand((8, 1, 256, 512)).cuda()
    gt = torch.rand((8, 1, 256, 512)).cuda()#torch.zeros_like(pred).cuda() #
    gt[1:, :, 100:256, 100:350] = -1
    gt[:2, ...] = -1
    mask = gt > 0
    sem_mask = np.random.randint(-1, 200, (8, 1, 256, 512))
    sem_mask[sem_mask>0] = -1
    sem_mask_torch = torch.from_numpy(sem_mask).cuda()

    out = ssil(pred, gt, mask, sem_mask_torch)
    print(out)
