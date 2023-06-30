import kornia.losses
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from loss.ms_ssim import MSSSIM

class FusionLoss(nn.Module):
    def __init__(self, alpha=1, beta=20, theta=5):
        super(FusionLoss, self).__init__()

        self.ms_ssim = MSSSIM()
        self.l1_loss = nn.L1Loss()
        self.grad_loss = JointGrad()

        self.alpha = alpha
        self.beta = beta
        self.theta = theta

    def forward(self, im_fus, im_ir, im_vi, map_ir, map_vi):
        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, im_ir)) + (1 - self.ms_ssim(im_fus, im_vi))

        ms_ssim_loss = (1 - self.ms_ssim(im_fus, (0.9 * map_ir * im_ir + 0.6 * map_vi * im_vi)))
        l1_loss = self.l1_loss(im_fus, (0.9 * map_ir * im_ir + 0.6 * map_vi * im_vi))
        grad_loss = self.grad_loss(im_fus, im_ir, im_vi)
        fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss

        # ms_ssim_loss = (1 - self.ms_ssim(im_fus, (map_ir * im_ir + map_vi * im_vi)))
        # l1_loss   = self.l1_loss(im_fus, (map_ir * im_ir + map_vi * im_vi))
        # grad_loss = self.grad_loss(im_fus, im_ir, im_vi)
        # fuse_loss = self.alpha * ms_ssim_loss + self.beta * l1_loss + self.theta * grad_loss

        return fuse_loss, l1_loss, grad_loss


class JointGrad(nn.Module):
    def __init__(self):
        super(JointGrad, self).__init__()

        self.laplacian = kornia.filters.laplacian
        self.l1_loss = nn.L1Loss()

    def forward(self, im_fus, im_ir, im_vi):

        ir_grad = torch.abs(self.laplacian(im_ir, 3))
        vi_grad = torch.abs(self.laplacian(im_vi, 3))
        fus_grad = torch.abs(self.laplacian(im_fus, 3))

        loss_JGrad = self.l1_loss(torch.max(ir_grad, vi_grad), fus_grad)

        # loss_JGrad = 0.8 * self.l1_loss(ir_grad, fus_grad) + 0.2 * self.l1_loss(vi_grad, fus_grad)

        return loss_JGrad


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss