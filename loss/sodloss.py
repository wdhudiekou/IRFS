import torch
import torch.nn as nn
import torch.nn.functional as F


class SODLoss(nn.Module):
    def __init__(self, lambda1=1, lambda2=1):
        super(SODLoss, self).__init__()

        self.lambda_1 = lambda1
        self.lambda_2 = lambda2

    def forward(self, pred, mask):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * mask) * weit).sum(dim=(2, 3))
        union = ((pred + mask) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        sod_loss = (self.lambda_1 * wbce + self.lambda_2 * wiou).mean()
        return sod_loss


def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter+1)/(union-inter+1)
    iou = iou.mean()
    return iou + 0.6 * bce


def bce_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)
    return bce

"""
boundary_loss:
    边界损失，用于训练对显著性目标边界的预测.
"""


def boundary_loss(pred_list, gt, ksize):
    N, _, H, W = gt.shape
    HW = H * W

    # 制作边缘 GT
    padding = int((ksize - 1) / 2)
    gt = torch.abs(gt - F.avg_pool2d(gt, ksize, 1, padding))
    gt = torch.where(gt > torch.zeros_like(gt), torch.ones_like(gt), torch.zeros_like(gt))  # [N, 1, H, W]

    loss_acc = 0
    for pred in pred_list:
        resized_pred = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
        loss = (resized_pred - gt) ** 2

        # pos_rate 和 neg_rate 用于平衡边缘/非边缘像素之间的损失
        pos_rate = torch.mean(gt.view(N, HW), dim=1).view(N, 1, 1, 1)  # [N, 1, 1, 1]
        neg_rate = 1.0 - pos_rate

        weight = gt * neg_rate + (1.0 - gt) * pos_rate
        loss = loss * weight
        loss_acc += loss.mean()
    return loss_acc
