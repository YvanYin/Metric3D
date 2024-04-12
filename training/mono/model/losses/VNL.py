import torch
import torch.nn as nn
import numpy as np


class VNLoss(nn.Module):
    """
    Virtual Normal Loss.
    """
    def __init__(self,
                 delta_cos=0.867, delta_diff_x=0.01,
                 delta_diff_y=0.01, delta_diff_z=0.01,
                 delta_z=1e-5, sample_ratio=0.15,
                 loss_weight=1.0, data_type=['sfm', 'stereo', 'lidar', 'denselidar', 'denselidar_nometric', 'denselidar_syn'], **kwargs):
        super(VNLoss, self).__init__()
        self.delta_cos = delta_cos
        self.delta_diff_x = delta_diff_x
        self.delta_diff_y = delta_diff_y
        self.delta_diff_z = delta_diff_z
        self.delta_z = delta_z
        self.sample_ratio = sample_ratio
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6


    def init_image_coor(self, intrinsic, height, width):
        # x_row = torch.arange(0, W, device="cuda")
        # x = torch.tile(x_row, (H, 1))
        # x = x.to(torch.float32)
        # u_m_u0 = x[None, None, :, :] - u0
        # self.register_buffer('u_m_u0', u_m_u0, persistent=False)

        # y_col = torch.arange(0, H, device="cuda")  # y_col = np.arange(0, height)
        # y = torch.transpose(torch.tile(y_col, (W, 1)), 1, 0)
        # y = y.to(torch.float32)
        # v_m_v0 = y[None, None, :, :] - v0
        # self.register_buffer('v_m_v0', v_m_v0, persistent=False)

        # pix_idx_mat = torch.arange(H*W, device="cuda").reshape((H, W))
        # self.register_buffer('pix_idx_mat', pix_idx_mat, persistent=False)
        #self.pix_idx_mat = torch.arange(height*width, device="cuda").reshape((height, width))
        
        u0 = intrinsic[:, 0, 2][:, None, None, None]
        v0 = intrinsic[:, 1, 2][:, None, None, None]
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device="cuda"),
                               torch.arange(0, width, dtype=torch.float32, device="cuda")], indexing='ij')
        u_m_u0 = x[None, None, :, :] - u0
        v_m_v0 = y[None, None, :, :] - v0
        # return u_m_u0, v_m_v0
        self.register_buffer('v_m_v0', v_m_v0, persistent=False)
        self.register_buffer('u_m_u0', u_m_u0, persistent=False)

    def transfer_xyz(self, depth, focal_length, u_m_u0, v_m_v0):
        x = u_m_u0 * depth / focal_length
        y = v_m_v0 * depth / focal_length
        z = depth
        pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1).contiguous() # [b, h, w, c]
        return pw

    def select_index(self, B, H, W, mask):
        """
        
        """
        p1 = []
        p2 = []
        p3 = []
        pix_idx_mat = torch.arange(H*W, device="cuda").reshape((H, W))
        for i in range(B):
            inputs_index = torch.masked_select(pix_idx_mat, mask[i, ...].gt(self.eps))
            num_effect_pixels = len(inputs_index)

            intend_sample_num = int(H * W * self.sample_ratio)
            sample_num = intend_sample_num if num_effect_pixels >= intend_sample_num else num_effect_pixels

            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p1i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p2i = inputs_index[shuffle_effect_pixels[:sample_num]]
            shuffle_effect_pixels = torch.randperm(num_effect_pixels, device="cuda")
            p3i = inputs_index[shuffle_effect_pixels[:sample_num]]

            cat_null = torch.tensor(([0,] * (intend_sample_num - sample_num)), dtype=torch.long, device="cuda")
            p1i = torch.cat([p1i, cat_null])
            p2i = torch.cat([p2i, cat_null])
            p3i = torch.cat([p3i, cat_null])

            p1.append(p1i)
            p2.append(p2i)
            p3.append(p3i)
        
        p1 = torch.stack(p1, dim=0)
        p2 = torch.stack(p2, dim=0)
        p3 = torch.stack(p3, dim=0)

        p1_x = p1 % W
        p1_y = torch.div(p1, W, rounding_mode='trunc').long() # p1 // W

        p2_x = p2 % W
        p2_y = torch.div(p2, W, rounding_mode='trunc').long() # p2 // W

        p3_x = p3 % W
        p3_y = torch.div(p3, W, rounding_mode='trunc').long() # p3 // W
        p123 = {'p1_x': p1_x, 'p1_y': p1_y, 'p2_x': p2_x, 'p2_y': p2_y, 'p3_x': p3_x, 'p3_y': p3_y}
        return p123

    def form_pw_groups(self, p123, pw):
        """
        Form 3D points groups, with 3 points in each grouup.
        :param p123: points index
        :param pw: 3D points
        :return:
        """
        B, _, _, _ = pw.shape
        p1_x = p123['p1_x']
        p1_y = p123['p1_y']
        p2_x = p123['p2_x']
        p2_y = p123['p2_y']
        p3_x = p123['p3_x']
        p3_y = p123['p3_y']
        
        pw_groups = []
        for i in range(B):
            pw1 = pw[i, p1_y[i], p1_x[i], :]
            pw2 = pw[i, p2_y[i], p2_x[i], :]
            pw3 = pw[i, p3_y[i], p3_x[i], :]
            pw_bi = torch.stack([pw1, pw2, pw3], dim=2)
            pw_groups.append(pw_bi)
        # [B, N, 3(x,y,z), 3(p1,p2,p3)]
        pw_groups = torch.stack(pw_groups, dim=0)
        return pw_groups

    def filter_mask(self, p123, gt_xyz, delta_cos=0.867,
                    delta_diff_x=0.005,
                    delta_diff_y=0.005,
                    delta_diff_z=0.005):
        pw = self.form_pw_groups(p123, gt_xyz)
        pw12 = pw[:, :, :, 1] - pw[:, :, :, 0]
        pw13 = pw[:, :, :, 2] - pw[:, :, :, 0]
        pw23 = pw[:, :, :, 2] - pw[:, :, :, 1]
        ###ignore linear
        pw_diff = torch.cat([pw12[:, :, :, np.newaxis], pw13[:, :, :, np.newaxis], pw23[:, :, :, np.newaxis]],
                            3)  # [b, n, 3, 3]
        m_batchsize, groups, coords, index = pw_diff.shape
        proj_query = pw_diff.view(m_batchsize * groups, -1, index).permute(0, 2, 1).contiguous()  # (B* X CX(3)) [bn, 3(p123), 3(xyz)]
        proj_key = pw_diff.contiguous().view(m_batchsize * groups, -1, index)  # B X  (3)*C [bn, 3(xyz), 3(p123)]
        q_norm = proj_query.norm(2, dim=2)
        nm = torch.bmm(q_norm.contiguous().view(m_batchsize * groups, index, 1), q_norm.view(m_batchsize * groups, 1, index)) #[]
        energy = torch.bmm(proj_query, proj_key)  # transpose check [bn, 3(p123), 3(p123)]
        norm_energy = energy / (nm + self.eps)
        norm_energy = norm_energy.contiguous().view(m_batchsize * groups, -1)
        mask_cos = torch.sum((norm_energy > delta_cos) + (norm_energy < -delta_cos), 1) > 3  # igonre
        mask_cos = mask_cos.contiguous().view(m_batchsize, groups)
        ##ignore padding and invilid depth
        mask_pad = torch.sum(pw[:, :, 2, :] > self.delta_z, 2) == 3

        ###ignore near
        mask_x = torch.sum(torch.abs(pw_diff[:, :, 0, :]) < delta_diff_x, 2) > 0
        mask_y = torch.sum(torch.abs(pw_diff[:, :, 1, :]) < delta_diff_y, 2) > 0
        mask_z = torch.sum(torch.abs(pw_diff[:, :, 2, :]) < delta_diff_z, 2) > 0

        mask_ignore = (mask_x & mask_y & mask_z) | mask_cos
        mask_near = ~mask_ignore
        mask = mask_pad & mask_near

        return mask, pw

    def select_points_groups(self, gt_depth, pred_depth, intrinsic, mask):
        B, C, H, W = gt_depth.shape
        focal_length = intrinsic[:, 0, 0][:, None, None, None]
        u_m_u0, v_m_v0 = self.u_m_u0, self.v_m_v0 # self.init_image_coor(intrinsic, height=H, width=W)
        
        pw_gt = self.transfer_xyz(gt_depth, focal_length, u_m_u0, v_m_v0)
        pw_pred = self.transfer_xyz(pred_depth, focal_length, u_m_u0, v_m_v0)
        
        p123 = self.select_index(B, H, W, mask)
        # mask:[b, n], pw_groups_gt: [b, n, 3(x,y,z), 3(p1,p2,p3)]
        mask, pw_groups_gt = self.filter_mask(p123, pw_gt,
                                              delta_cos=0.867,
                                              delta_diff_x=0.005,
                                              delta_diff_y=0.005,
                                              delta_diff_z=0.005)

        # [b, n, 3, 3]
        pw_groups_pred = self.form_pw_groups(p123, pw_pred)
        pw_groups_pred[pw_groups_pred[:, :, 2, :] == 0] = 0.0001
        mask_broadcast = mask.repeat(1, 9).reshape(B, 3, 3, -1).permute(0, 3, 1, 2).contiguous()
        pw_groups_pred_not_ignore = pw_groups_pred[mask_broadcast].reshape(1, -1, 3, 3)
        pw_groups_gt_not_ignore = pw_groups_gt[mask_broadcast].reshape(1, -1, 3, 3)

        return pw_groups_gt_not_ignore, pw_groups_pred_not_ignore

    def forward(self, prediction, target, mask, intrinsic, select=True, **kwargs): #gt_depth, pred_depth, select=True):
        """
        Virtual normal loss.
        :param prediction: predicted depth map, [B,W,H,C]
        :param data: target label, ground truth depth, [B, W, H, C], padding region [padding_up, padding_down]
        :return:
        """
        loss  = self.get_loss(prediction, target, mask, intrinsic, select, **kwargs)
        return loss
 
    
    def get_loss(self, prediction, target, mask, intrinsic, select=True, **kwargs):
        # configs for the cameras
        # focal_length = intrinsic[:, 0, 0][:, None, None, None]
        # u0 = intrinsic[:, 0, 2][:, None, None, None]
        # v0 = intrinsic[:, 1, 2][:, None, None, None]
        B, _, H, W = target.shape
        if 'u_m_u0' not in self._buffers or 'v_m_v0' not in self._buffers \
            or self.u_m_u0.shape != torch.Size([B,1,H,W]) or self.v_m_v0.shape != torch.Size([B,1,H,W]):
            self.init_image_coor(intrinsic, H, W)


        gt_points, pred_points = self.select_points_groups(target, prediction, intrinsic, mask)

        gt_p12 = gt_points[:, :, :, 1] - gt_points[:, :, :, 0]
        gt_p13 = gt_points[:, :, :, 2] - gt_points[:, :, :, 0]
        pred_p12 = pred_points[:, :, :, 1] - pred_points[:, :, :, 0]
        pred_p13 = pred_points[:, :, :, 2] - pred_points[:, :, :, 0]

        gt_normal = torch.cross(gt_p12, gt_p13, dim=2)
        pred_normal = torch.cross(pred_p12, pred_p13, dim=2)
        pred_norm = torch.norm(pred_normal, 2, dim=2, keepdim=True)
        gt_norm = torch.norm(gt_normal, 2, dim=2, keepdim=True)
        pred_mask = pred_norm == 0.0
        gt_mask = gt_norm == 0.0
        pred_mask = pred_mask.to(torch.float32)
        gt_mask = gt_mask.to(torch.float32)
        pred_mask *= self.eps
        gt_mask *= self.eps
        gt_norm = gt_norm + gt_mask
        pred_norm = pred_norm + pred_mask
        gt_normal = gt_normal / gt_norm
        pred_normal = pred_normal / pred_norm
        loss = torch.abs(gt_normal - pred_normal)
        loss = torch.sum(torch.sum(loss, dim=2), dim=0)
        if select:
            loss, indices = torch.sort(loss, dim=0, descending=False)
            loss = loss[int(loss.size(0) * 0.25):]
        loss = torch.sum(loss) / (loss.numel() + self.eps)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            loss = 0 * torch.sum(prediction)
            print(f'VNL NAN error, {loss}')        
        return loss * self.loss_weight


if __name__ == '__main__':
    import cv2
    vnl_loss = VNLoss()
    pred_depth = np.random.random([2, 1, 480, 640])
    gt_depth = np.zeros_like(pred_depth) #np.random.random([2, 1, 480, 640])
    intrinsic = [[[100, 0, 200], [0, 100, 200], [0, 0, 1]], [[100, 0, 200], [0, 100, 200], [0, 0, 1]],]
    gt_depth = torch.tensor(np.array(gt_depth, np.float32)).cuda()
    pred_depth = torch.tensor(np.array(pred_depth, np.float32)).cuda()
    intrinsic = torch.tensor(np.array(intrinsic, np.float32)).cuda()
    mask = gt_depth > 0
    loss1 = vnl_loss(pred_depth, gt_depth, mask, intrinsic)
    loss2 = vnl_loss(pred_depth, gt_depth, mask, intrinsic)
    print(loss1, loss2)
