import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .depth_to_normal import Depth2Normal
"""
Sampling strategies: RS (Random Sampling), EGS (Edge-Guided Sampling), and IGS (Instance-Guided Sampling)
"""
###########
# RANDOM SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], self.mask_value, self.point_pairs
# return:
# inputs_A, inputs_B, targets_A, targets_B, consistent_masks_A, consistent_masks_B
###########
def randomSamplingNormal(inputs, targets, masks, sample_num):

    # find A-B point pairs from prediction
    num_effect_pixels = torch.sum(masks)
    shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
    valid_inputs = inputs[:, masks]
    valid_targes = targets[:, masks]
    inputs_A = valid_inputs[:, shuffle_effect_pixels[0 : sample_num * 2 : 2]]
    inputs_B = valid_inputs[:, shuffle_effect_pixels[1 : sample_num * 2 : 2]]
    # find corresponding pairs from GT
    targets_A = valid_targes[:, shuffle_effect_pixels[0 : sample_num * 2 : 2]]
    targets_B = valid_targes[:, shuffle_effect_pixels[1 : sample_num * 2 : 2]]
    if inputs_A.shape[1] != inputs_B.shape[1]:
        num_min = min(targets_A.shape[1], targets_B.shape[1])
        inputs_A = inputs_A[:, :num_min]
        inputs_B = inputs_B[:, :num_min]
        targets_A = targets_A[:, :num_min]
        targets_B = targets_B[:, :num_min]
    return inputs_A, inputs_B, targets_A, targets_B


###########
# EDGE-GUIDED SAMPLING
# input:
# inputs[i,:], targets[i, :], masks[i, :], edges_img[i], thetas_img[i], masks[i, :], h, w
# return:
# inputs_A, inputs_B, targets_A, targets_B, masks_A, masks_B
###########
def ind2sub(idx, cols):
    r = torch.div(idx, cols, rounding_mode='floor')
    c = idx - r * cols
    return r, c


def sub2ind(r, c, cols):
    idx = r * cols + c
    return idx


def edgeGuidedSampling(inputs, targets, edges_img, thetas_img, masks, h, w):
    # find edges
    edges_max = edges_img.max()
    edges_min = edges_img.min()
    edges_mask = edges_img.ge(edges_max * 0.1)
    edges_loc = edges_mask.nonzero(as_tuple=False)

    thetas_edge = torch.masked_select(thetas_img, edges_mask)
    minlen = thetas_edge.size()[0]

    # find anchor points (i.e, edge points)
    sample_num = minlen
    index_anchors = torch.randint(0, minlen, (sample_num,), dtype=torch.long, device="cuda")
    theta_anchors = torch.gather(thetas_edge, 0, index_anchors)
    row_anchors, col_anchors = ind2sub(edges_loc[index_anchors].squeeze(1), w)
    ## compute the coordinates of 4-points,  distances are from [2, 30]
    distance_matrix = torch.randint(3, 20, (4, sample_num), device="cuda")
    pos_or_neg = torch.ones(4, sample_num, device="cuda")
    pos_or_neg[:2, :] = -pos_or_neg[:2, :]
    distance_matrix = distance_matrix.float() * pos_or_neg
    col = (
        col_anchors.unsqueeze(0).expand(4, sample_num).long()
        + torch.round(
            distance_matrix.float() * torch.abs(torch.cos(theta_anchors)).unsqueeze(0)
        ).long()
    )
    row = (
        row_anchors.unsqueeze(0).expand(4, sample_num).long()
        + torch.round(
            distance_matrix.float() * torch.abs(torch.sin(theta_anchors)).unsqueeze(0)
        ).long()
    )

    # constrain 0=<c<=w, 0<=r<=h
    # Note: index should minus 1
    col[col < 0] = 0
    col[col > w - 1] = w - 1
    row[row < 0] = 0
    row[row > h - 1] = h - 1

    # a-b, b-c, c-d
    a = sub2ind(row[0, :], col[0, :], w)
    b = sub2ind(row[1, :], col[1, :], w)
    c = sub2ind(row[2, :], col[2, :], w)
    d = sub2ind(row[3, :], col[3, :], w)
    A = torch.cat((a, b, c), 0)
    B = torch.cat((b, c, d), 0)

    

    inputs_A = inputs[:, A]
    inputs_B = inputs[:, B]
    targets_A = targets[:, A]
    targets_B = targets[:, B]
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
    return (
        inputs_A,
        inputs_B,
        targets_A,
        targets_B,
        masks_A,
        masks_B,
        sample_num,
        row,
        col,
    )


######################################################
# EdgeguidedNormalRankingLoss
#####################################################
class EdgeguidedNormalLoss(nn.Module):
    def __init__(
        self,
        point_pairs=10000,
        cos_theta1=0.25,
        cos_theta2=0.98,
        cos_theta3=0.5,
        cos_theta4=0.86,
        mask_value=1e-8,
        loss_weight=1.0, 
        data_type=['stereo', 'denselidar', 'denselidar_nometric','denselidar_syn'],
        **kwargs
    ):
        super(EdgeguidedNormalLoss, self).__init__()
        self.point_pairs = point_pairs  # number of point pairs
        self.mask_value = mask_value
        self.cos_theta1 = cos_theta1  # 75 degree
        self.cos_theta2 = cos_theta2  # 10 degree
        self.cos_theta3 = cos_theta3  # 60 degree
        self.cos_theta4 = cos_theta4  # 30 degree
        # self.kernel = torch.tensor(
        #     np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32),
        #     requires_grad=False,
        # )[None, None, :, :].cuda()
        self.depth2normal = Depth2Normal()
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6


    def getEdge(self, images):
        n, c, h, w = images.size()
        a = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda")
            .contiguous()
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        b = (
            torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device="cuda")
            .contiguous()
            .view((1, 1, 3, 3))
            .repeat(1, 1, 1, 1)
        )
        if c == 3:
            gradient_x = F.conv2d(images[:, 0, :, :].unsqueeze(1), a)
            gradient_y = F.conv2d(images[:, 0, :, :].unsqueeze(1), b)
        else:
            gradient_x = F.conv2d(images, a)
            gradient_y = F.conv2d(images, b)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def getNormalEdge(self, normals):
        n, c, h, w = normals.size()
        a = (
            torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device="cuda")
            .contiguous()
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        b = (
            torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device="cuda")
            .contiguous()
            .view((1, 1, 3, 3))
            .repeat(3, 1, 1, 1)
        )
        gradient_x = torch.abs(F.conv2d(normals, a, groups=c))
        gradient_y = torch.abs(F.conv2d(normals, b, groups=c))
        gradient_x = gradient_x.mean(dim=1, keepdim=True)
        gradient_y = gradient_y.mean(dim=1, keepdim=True)
        edges = torch.sqrt(torch.pow(gradient_x, 2) + torch.pow(gradient_y, 2))
        edges = F.pad(edges, (1, 1, 1, 1), "constant", 0)
        thetas = torch.atan2(gradient_y, gradient_x)
        thetas = F.pad(thetas, (1, 1, 1, 1), "constant", 0)
        return edges, thetas

    def visual_check(self, rgb, samples):
        import os
        import matplotlib.pyplot as plt
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

    def forward(self, prediction, target, mask, input, intrinsic, **kwargs):
        loss  = self.get_loss(prediction, target, mask, input, intrinsic, **kwargs)
        return loss

    def get_loss(self, prediction, target, mask, input, intrinsic, **kwargs):
        """
        input and target: surface normal input
        input: rgb images
        """
        gt_depths = target

        if 'predictions_normals' not in kwargs:
            predictions_normals, _ = self.depth2normal(prediction, intrinsic, mask)
            targets_normals, targets_normals_masks = self.depth2normal(target, intrinsic, mask)
        else:
            predictions_normals = kwargs['predictions_normals']
            targets_normals = kwargs['targets_normals']
            targets_normals_masks = kwargs['targets_normals_masks']
        masks_normals = mask & targets_normals_masks
        
        # find edges from RGB
        edges_img, thetas_img = self.getEdge(input)

        # find edges from normals
        # edges_normal, thetas_normal = self.getNormalEdge(targets_normals)
        #mask_img_border = torch.ones_like(edges_normal)  # normals on the borders
        #mask_img_border[:, :, 5:-5, 5:-5] = 0
        # edges_normal[~targets_normals_masks] = 0

        # find edges from depth
        edges_depth, thetas_depth = self.getEdge(gt_depths)
        # edges_depth_mask = edges_depth.ge(edges_depth.max() * 0.1)
        # edges_mask_dilate = torch.clamp(
        #     torch.nn.functional.conv2d(
        #         edges_depth_mask.float(), self.kernel, padding=(1, 1)
        #     ),
        #     0,
        #     1,
        # ).bool()
        # edges_normal[edges_mask_dilate] = 0
        # edges_img[edges_mask_dilate] = 0

        # =============================
        n, c, h, w = targets_normals.size()

        predictions_normals = predictions_normals.contiguous().view(n, c, -1)
        targets_normals = targets_normals.contiguous().view(n, c, -1)
        masks_normals = masks_normals.contiguous().view(n, -1)
        edges_img = edges_img.contiguous().view(n, -1)
        thetas_img = thetas_img.contiguous().view(n, -1)
        # edges_normal = edges_normal.view(n, -1)
        # thetas_normal = thetas_normal.view(n, -1)
        edges_depth = edges_depth.contiguous().view(n, -1)
        thetas_depth = thetas_depth.contiguous().view(n, -1)

        # # initialization
        losses = 0.0
        valid_samples = 0.0
        for i in range(n):
            # Edge-Guided sampling
            (
                inputs_A,
                inputs_B,
                targets_A,
                targets_B,
                masks_A,
                masks_B,
                sample_num,
                row_img,
                col_img,
            ) = edgeGuidedSampling(
                predictions_normals[i, :],
                targets_normals[i, :],
                edges_img[i],
                thetas_img[i],
                masks_normals[i, :],
                h,
                w,
            )
            # Depth-Guided sampling
            # (
            #     depth_inputs_A,
            #     depth_inputs_B,
            #     depth_targets_A,
            #     depth_targets_B,
            #     depth_masks_A,
            #     depth_masks_B,
            #     depth_sample_num,
            #     row_img,
            #     col_img,
            # ) = edgeGuidedSampling(
            #     predictions_normals[i, :],
            #     targets_normals[i, :],
            #     edges_depth[i],
            #     thetas_depth[i],
            #     masks_normals[i, :],
            #     h,
            #     w,
            # )
            # Normal-Guided sampling
            # (
            #     normal_inputs_A,
            #     normal_inputs_B,
            #     normal_targets_A,
            #     normal_targets_B,
            #     normal_masks_A,
            #     normal_masks_B,
            #     normal_sample_num,
            #     row_normal,
            #     col_normal,
            # ) = edgeGuidedSampling(
            #     predictions_normals[i, :],
            #     targets_normals[i, :],
            #     edges_normal[i],
            #     thetas_normal[i],
            #     masks_normals[i, :],
            #     h,
            #     w,
            # )

            # Combine EGS + DEGS
            # inputs_A = torch.cat((inputs_A, depth_inputs_A), 1) #normal_inputs_A
            # inputs_B = torch.cat((inputs_B, depth_inputs_B), 1) # normal_inputs_B
            # targets_A = torch.cat((targets_A, depth_targets_A), 1) #normal_targets_A
            # targets_B = torch.cat((targets_B, depth_targets_B), 1) #normal_targets_B
            # masks_A = torch.cat((masks_A, depth_masks_A), 0) #normal_masks_A
            # masks_B = torch.cat((masks_B, depth_masks_B), 0) #normal_masks_B

            # consider forward-backward consistency checking, i.e, only compute losses of point pairs with valid GT
            consistency_mask = masks_A & masks_B

            # GT ordinal relationship
            target_cos = torch.sum(targets_A * targets_B, dim=0)
            input_cos = torch.sum(inputs_A * inputs_B, dim=0)

            losses += torch.sum(torch.abs(torch.ones_like(target_cos)-input_cos) * consistency_mask.float())
            valid_samples += torch.sum(consistency_mask.float())

        loss = (losses / (valid_samples + self.eps)) * self.loss_weight
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'Pair-wise Normal Regression Loss NAN error, {loss}, valid pix: {valid_samples}')
        return loss

def tmp_check_normal(normals, masks, depth):
    import matplotlib.pyplot as plt
    import os
    import cv2
    from mono.utils.visualization import vis_surface_normal
    vis_normal1 = vis_surface_normal(normals[0, ...].permute(1, 2, 0).detach(), masks[0,...].detach().squeeze())
    vis_normal2 = vis_surface_normal(normals[1, ...].permute(1, 2, 0).detach(), masks[1,...].detach().squeeze())
    vis_depth1 = depth[0, ...].detach().cpu().squeeze().numpy()
    vis_depth2 = depth[1, ...].detach().cpu().squeeze().numpy()

    name = np.random.randint(100000)
    os.makedirs('test_normal', exist_ok=True)
    cv2.imwrite(f'test_normal/{name}.png', vis_normal1)
    cv2.imwrite(f'test_normal/{name + 1}.png', vis_normal2)
    plt.imsave(f'test_normal/{name}_d.png', vis_depth1)
    plt.imsave(f'test_normal/{name + 1}_d.png', vis_depth2)

if __name__ == '__main__':
    ENL = EdgeguidedNormalLoss()
    depth = np.random.randn(2, 1, 20, 22)
    intrin = np.array([[300, 0, 10], [0, 300, 10], [0,0,1]])
    prediction = np.random.randn(2, 1, 20, 22)
    imgs = np.random.randn(2, 3, 20, 22)
    intrinsics = np.stack([intrin, intrin], axis=0)

    depth_t = torch.from_numpy(depth).cuda().float()
    prediction = torch.from_numpy(prediction).cuda().float()
    intrinsics = torch.from_numpy(intrinsics).cuda().float()
    imgs = torch.from_numpy(imgs).cuda().float()
    depth_t = -1 * torch.abs(depth_t)

    loss = ENL(prediction, depth_t, masks=depth_t>0, images=imgs, intrinsic=intrinsics)
    print(loss)