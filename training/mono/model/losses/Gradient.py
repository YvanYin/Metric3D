import torch
import torch.nn as nn

EPSILON = 1e-6
"""
  # @Zhengqi Li version.  
  def GradientLoss(self, log_prediction_d, mask, log_gt):
        log_d_diff = log_prediction_d - log_gt

        v_gradient = torch.abs(log_d_diff[:, :-2, :] - log_d_diff[:, 2:, :])
        v_mask = torch.mul(mask[:, :-2, :], mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, :, :-2] - log_d_diff[:, :, 2:])
        h_mask = torch.mul(mask[:, :, :-2], mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / N

        return gradient_loss
"""
def gradient_log_loss(log_prediction_d, log_gt, mask):
    log_d_diff = log_prediction_d - log_gt

    v_gradient = torch.abs(log_d_diff[:, :, :-2, :] - log_d_diff[:, :, 2:, :])
    v_mask = torch.mul(mask[:, :, :-2, :], mask[:, :, 2:, :])
    v_gradient = torch.mul(v_gradient, v_mask)

    h_gradient = torch.abs(log_d_diff[:, :, :, :-2] - log_d_diff[:, :, :, 2:])
    h_mask = torch.mul(mask[:, :, :, :-2], mask[:, :, :, 2:])
    h_gradient = torch.mul(h_gradient, h_mask)

    N = torch.sum(h_mask) + torch.sum(v_mask) + EPSILON

    gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
    gradient_loss = gradient_loss / N

    return gradient_loss

class GradientLoss_Li(nn.Module):
    def __init__(self, scale_num=1, loss_weight=1, data_type = ['lidar', 'stereo'], **kwargs):
        super(GradientLoss_Li, self).__init__()
        self.__scales = scale_num
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6

    def forward(self, prediction, target, mask, **kwargs):
        total = 0
        target_trans = target + (~mask) * 100
        pred_log = torch.log(prediction)
        gt_log = torch.log(target_trans)
        for scale in range(self.__scales):
            step = pow(2, scale)
            
            total += gradient_log_loss(pred_log[:, ::step, ::step], gt_log[:, ::step, ::step], mask[:, ::step, ::step])
        loss = total / self.__scales
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'VNL error, {loss}')
        return loss * self.loss_weight
  
######################################################
# Multi-scale gradient matching loss, @Ke Xian implementation.
#####################################################
def gradient_loss(prediction, target, mask):
    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))
    valid = M.nonzero()
    if image_loss[valid].numel() > 0:
        image_loss[valid] = image_loss[valid] / M[valid]
        loss = torch.mean(image_loss)
    else:
        loss = 0 * torch.sum(prediction)

    return loss


class GradientLoss(nn.Module):
    def __init__(self, scale_num=4, loss_weight=1, **kwargs):
        super(GradientLoss, self).__init__()
        self.__scales = scale_num
        self.loss_weight = loss_weight
    def forward(self, prediction, target, mask, **kwargs):
        total = 0
        for scale in range(self.__scales):
            step = pow(2, scale)
            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step], mask[:, ::step, ::step])
         
        return total * self.loss_weight


if __name__ == '__main__':
    import numpy as np
    gradient = GradientLoss_Li(4)

    pred_depth = np.random.random([2, 1, 480, 640])
    gt_depth = np.ones_like(pred_depth) * (-1) #np.random.random([2, 1, 480, 640]) - 0.5 #
    #gt_depth = np.abs(gt_depth)
    intrinsic = [[100, 100, 200, 200], [200, 200, 300, 300]]

    pred = torch.from_numpy(pred_depth).cuda()
    gt = torch.from_numpy(gt_depth).cuda()
    mask = gt > 0

    loss = gradient(gt, gt, mask)
    print(loss)