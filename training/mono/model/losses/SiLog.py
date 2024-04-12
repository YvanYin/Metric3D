import torch
import torch.nn as nn

class SilogLoss(nn.Module):
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.
    """
    def __init__(self, variance_focus=0.5, loss_weight=1, data_type=['stereo', 'lidar'], **kwargs):
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus
        self.loss_weight = loss_weight
        self.data_type = data_type
        self.eps = 1e-6
    
    def silog_loss(self, prediction, target, mask):
        d = torch.log(prediction[mask]) - torch.log(target[mask])
        d_square_mean = torch.sum(d ** 2) / (d.numel() + self.eps)
        d_mean = torch.sum(d) / (d.numel() + self.eps)
        loss = d_square_mean - self.variance_focus * (d_mean ** 2)
        return loss

    def forward(self, prediction, target, mask=None, **kwargs):
        if target[mask].numel() > 0:
            loss = self.silog_loss(prediction, target, mask)
        else:
            loss = 0 * torch.sum(prediction)
        if torch.isnan(loss).item() | torch.isinf(loss).item():
            raise RuntimeError(f'Silog error, {loss}, d_square_mean: {d_square_mean}, d_mean: {d_mean}')
        return loss * self.loss_weight
      
if __name__ == '__main__':
    silog = SilogLoss()
    pred = torch.rand((2, 3, 256, 256)).cuda()
    gt =  torch.zeros_like(pred) #torch.rand((2, 3, 256, 256)).cuda()
    mask = gt > 0
    out = silog(pred, gt, mask)
    print(out)
