import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling), and IGS (Instance-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# predictions[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSampling(predictions, targets, masks, threshold, sample_num):

    # find A-B point pairs from predictions
    inputs_index = torch.masked_select(predictions, targets.gt(threshold))
    num_effect_pixels = len(inputs_index)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
    inputs_A = inputs_index[shuffle_effect_pixels[0:sample_num*2:2]]
    inputs_B = inputs_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # find corresponding pairs from GT
    target_index = torch.masked_select(targets, targets.gt(threshold))
    targets_A = target_index[shuffle_effect_pixels[0:sample_num*2:2]]
    targets_B = target_index[shuffle_effect_pixels[1:sample_num*2:2]]
    # only compute the losses of point pairs with valid GT
    consistent_masks_index = torch.masked_select(masks, targets.gt(threshold))
    consistent_masks_A = consistent_masks_index[shuffle_effect_pixels[0:sample_num*2:2]]
    consistent_masks_B = consistent_masks_index[shuffle_effect_pixels[1:sample_num*2:2]]

    # The amount of A and B should be the same!!
    if len(targets_A) > len(targets_B):
        targets_A = targets_A[:-1]
        inputs_A = inputs_A[:-1]
        consistent_masks_A = consistent_masks_A[:-1]

    return inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B

###########
# EDGE-GUIDED SAMPLING
# input:
# predictions[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = torch.div(idx, cols, rounding_mode='floor') #idx // cols
    c = idx % cols
    return r, c

def sub2ind(r, c, cols):
    idx = (r * cols + c).int()
    return idx

def edgeGuidedSampling(predictions, targets, edges_img, thetas_img, masks, h, w):

    # find edges
    edges_max = edges_img.max()
    edges_mask = edges_img.ge(edges_max*0.1)
    edges_loc = edges_mask.nonzero()

    inputs_edge = torch.masked_select(predictions, edges_mask)
    targets_edge = torch.masked_select(targets, edges_mask)
    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = inputs_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long, device="cuda")
    anchors = torch.gather(inputs_edge, 0, index_anchors)
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(2, 40, (4,sample_num), device="cuda")
    pos_or_neg = torch.ones(4, sample_num, device="cuda")
    pos_or_neg[:2,:] = -pos_or_neg[:2,:]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = col_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.float() * torch.abs(torch.cos(theta_anchors)).unsqueeze(0)).long()
    row = row_anchors.unsqueeze(0).expand(4, sample_num).long() + torch.round(distance_matrix.float() * torch.abs(torch.sin(theta_anchors)).unsqueeze(0)).long()

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col<0] = 0
    col[col>w-1] = w-1
    row[row<0] = 0
    row[row>h-1] = h-1

    # a-b, b-c, c-d
    a = sub2ind(row[0,:], col[0,:], w)
    b = sub2ind(row[1,:], col[1,:], w)
    c = sub2ind(row[2,:], col[2,:], w)
    d = sub2ind(row[3,:], col[3,:], w)
    A = torch.cat((a,b,c), 0)
    B = torch.cat((b,c,d), 0)

    inputs_A = torch.gather(predictions, 0, A.long())
    inputs_B = torch.gather(predictions, 0, B.long())
    targets_A = torch.gather(targets, 0, A.long())
    targets_B = torch.gather(targets, 0, B.long())
    masks_A = torch.gather(masks, 0, A.long())
    masks_B = torch.gather(masks, 0, B.long())

    # create A, B, C, D mask for visualization
    # vis_mask = masks.reshape(h, w).cpu().numpy()
    # vis_row = row.cpu()
    # vis_col = col.cpu()
    # visual_A = np.zeros((h, w)).astype(np.bool)
    # visual_B = np.zeros_like(visual_A)
    # visual_C = np.zeros_like(visual_A)
    # visual_D = np.zeros_like(visual_A)
    # visual_A[vis_row[0, :], vis_col[0, :]] = True
    # visual_B[vis_row[1, :], vis_col[1, :]] = True
    # visual_C[vis_row[2, :], vis_col[2, :]] = True
    # visual_D[vis_row[3, :], vis_col[3, :]] = True
    # visual_ABCD = [visual_A & vis_mask, visual_B & vis_mask, 
    # visual_C& vis_mask, visual_D& vis_mask]
    return inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num


######################################################
# Ranking loss (Random sampling)
#####################################################
class RankingLoss(nn.Module):
    def __init__(self, point_pairs=5000, sigma=0.03, alpha=1.0, mask_value=-1e-8, loss_weight=1, **kwargs):
        super(RankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha  # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        self.loss_weight = loss_weight
        self.eps = 1e-6

    def forward(self, prediction, target, mask=None, **kwargs):
        n,c,h,w = target.size()
        if mask == None:
            mask = target > self.mask_value
        if n != 1:
            prediction = prediction.view(n, -1)#.double()
            target = target.view(n, -1)#.double()
            mask = mask.view(n, -1)#.double()
        else:
            prediction = prediction.contiguous().view(1, -1)#.double()
            target = target.contiguous().view(1, -1)#.double()
            mask = mask.contiguous().view(1, -1)#.double()

        loss = 0.0 #torch.tensor([0.0]).cuda()
        valid_samples = 0
        for i in range(n):
            # find A-B point pairs
            inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B = randomSampling(prediction[i,:], target[i, :], mask[i, :], self.mask_value, self.point_pairs)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A, targets_B+self.eps)
            mask_eq = target_ratio.lt(1.0 + self.sigma) * target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, only compute the losses of point pairs with valid GT
            consistency_mask = consistent_masks_A & consistent_masks_B

            # compute loss
            equal_loss = (inputs_A - inputs_B).pow(2)[mask_eq & consistency_mask]
            unequal_loss = torch.log(1 + torch.exp((-inputs_A + inputs_B) * labels))[(~mask_eq) & consistency_mask]

            loss = loss + self.alpha * equal_loss.sum() + unequal_loss.sum()
            valid_samples = valid_samples + unequal_loss.numel() + equal_loss.numel()
        loss = loss / (valid_samples + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'VNL error, {loss}')
        return loss * self.loss_weight





######################################################
# EdgeguidedRankingLoss (with regularization term)
# Please comment regularization_loss if you don't want to use multi-scale gradient matching term
#####################################################
class EdgeguidedRankingLoss(nn.Module):
    def __init__(self, point_pairs=5000, sigma=0.03, alpha=1.0, mask_value=1e-6, loss_weight=1.0, data_type=['rel', 'sfm', 'stereo', 'lidar'], **kwargs):
        super(EdgeguidedRankingLoss, self).__init__()
        self.point_pairs = point_pairs # number of point pairs
        self.sigma = sigma # used for determining the ordinal relationship between a selected pair
        self.alpha = alpha # used for balancing the effect of = and (<,>)
        self.mask_value = mask_value
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def getEdge(self, images):
        n,c,h,w = images.size()
        a = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda").view((1,1,3,3)).repeat(1, 1, 1, 1)
        b = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device="cuda").view((1,1,3,3)).repeat(1, 1, 1, 1)
        if c == 3:
            gradient_x = F.conv2d(images[:,0,:,:].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:,0,:,:].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x,2)+ torch.pow(gradient_y,2))
        edges = F.pad(edges, (1,1,1,1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1,1,1,1), "constant", 0)

        return edges, thetas
    
    def visual_check(self, rgb, samples):
        rgb = rgb.cpu().squeeze().numpy()

        mean = np.array([123.675, 116.28, 103.53])[:, np.newaxis, np.newaxis]
        std= np.array([58.395, 57.12, 57.375])[:, np.newaxis, np.newaxis]
        
        rgb = ((rgb * std) + mean).astype(np.uint8).transpose((1, 2, 0))
        mask_A, mask_B, mask_C, mask_D = samples
        rgb[mask_A.astype(np.bool)] = [255, 0, 0]
        rgb[mask_B.astype(np.bool)] = [0, 255, 0]
        rgb[mask_C.astype(np.bool)] = [0, 0, 255]
        rgb[mask_D.astype(np.bool)] = [255, 255, 0]
        
        filename = str(np.random.randint(10000))
        save_path = os.path.join('test_ranking', filename + '.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.imsave(save_path, rgb)

    def forward(self, prediction, target, mask=None, input=None, **kwargs):
        loss  = self.get_loss(prediction, target, mask, input, **kwargs)
        return loss
    
    def get_loss(self, prediction, target, mask=None, input=None, **kwargs):
        if mask == None:
            mask = target > self.mask_value
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(input)
        # find edges from target depths
        edges_depth, thetas_depth = self.getEdge(target)

        #=============================
        n,c,h,w = target.size()
        if n != 1:
            prediction = prediction.view(n, -1)#.double()
            target = target.view(n, -1)#.double()
            mask = mask.view(n, -1)#.double()
            edges_img = edges_img.view(n, -1)#.double()
            thetas_img = thetas_img.view(n, -1)#.double()
            edges_depth = edges_depth.view(n, -1)#.double()
            thetas_depth = thetas_depth.view(n, -1)#.double()
        else:
            prediction = prediction.contiguous().view(1, -1)#.double()
            target = target.contiguous().view(1, -1)#.double()
            mask = mask.contiguous().view(1, -1)#.double()
            edges_img = edges_img.contiguous().view(1, -1)#.double()
            thetas_img = thetas_img.contiguous().view(1, -1)#.double()
            edges_depth = edges_depth.view(1, -1)#.double()
            thetas_depth = thetas_depth.view(1, -1)#.double()

        # initialization
        loss = 0.0 #torch.tensor([0.0]).cuda()
        valid_samples = 0

        for i in range(n):
            # Edge-Guided sampling from RGB predictions, targets, edges_img, thetas_img, masks, h, w
            inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B, sample_num = edgeGuidedSampling(
                prediction[i,:], 
                target[i, :], 
                edges_img[i], 
                thetas_img[i], 
                mask[i, :], 
                h, 
                w
                )
            # # Edge-Guided sampling from depth
            # inputs_A_depth, inputs_B_depth, targets_A_depth, targets_B_depth, masks_A_depth, masks_B_depth, sample_num_depth = edgeGuidedSampling(
            #     prediction[i,:], 
            #     target[i, :], 
            #     edges_depth[i], 
            #     thetas_depth[i], 
            #     mask[i, :], 
            #     h, 
            #     w
            #     )

            # Random Sampling predictions, targets, masks, threshold, sample_num
            random_sample_num = sample_num
            random_inputs_A, random_inputs_B, random_targets_A, random_targets_B, random_masks_A, random_masks_B = randomSampling(
                prediction[i,:], 
                target[i, :], 
                mask[i, :], 
                self.mask_value, 
                random_sample_num
                )

            # Combine EGS + RS + EGS_depth
            inputs_A_merge = torch.cat((inputs_A, random_inputs_A,), 0)
            inputs_B_merge = torch.cat((inputs_B, random_inputs_B,), 0)
            targets_A_merge = torch.cat((targets_A, random_targets_A,), 0)
            targets_B_merge = torch.cat((targets_B, random_targets_B,), 0)
            masks_A_merge = torch.cat((masks_A, random_masks_A,), 0)
            masks_B_merge = torch.cat((masks_B, random_masks_B,), 0)

            #GT ordinal relationship
            target_ratio = torch.div(targets_A_merge + 1e-6, targets_B_merge + 1e-6)
            mask_eq = target_ratio.lt(1.0 + self.sigma) & target_ratio.gt(1.0/(1.0+self.sigma))
            labels = torch.zeros_like(target_ratio)
            labels[target_ratio.ge(1.0 + self.sigma)] = 1
            labels[target_ratio.le(1.0/(1.0+self.sigma))] = -1

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A_merge & masks_B_merge

            equal_loss = (inputs_A_merge - inputs_B_merge).pow(2)[mask_eq & consistency_mask]
            unequal_loss = torch.log(1 + torch.exp((-inputs_A_merge + inputs_B_merge) * labels))[(~mask_eq) & consistency_mask]

            loss = loss + self.alpha * torch.sum(equal_loss) + torch.sum(unequal_loss)
            valid_samples = valid_samples + equal_loss.numel()
            valid_samples = valid_samples + unequal_loss.numel()
        loss = loss / (valid_samples + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'VNL error, {loss}')
        return loss * self.loss_weight


if __name__ == '__main__':
    import cv2

    rank_loss = EdgeguidedRankingLoss()
    pred_depth = np.random.randn(2, 1, 480, 640)
    gt_depth = np.ones((2, 1, 480, 640)) #np.random.randn(2, 1, 480, 640)
    # gt_depth = cv2.imread('/hardware/yifanliu/SUNRGBD/sunrgbd-meta-data/sunrgbd_test_depth/2.png', -1)
    # gt_depth = gt_depth[None, :, :, None]
    # pred_depth = gt_depth[:, :, ::-1, :]
    gt_depth = torch.tensor(np.asarray(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.asarray(pred_depth, np.float32)).cuda()
    input = np.random.randn(2, 3, 480, 640)
    input_torch = torch.tensor(np.asarray(input, np.float32)).cuda()
    loss = rank_loss(gt_depth, gt_depth, gt_depth>0, input=input_torch)
    print(loss)
