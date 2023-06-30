import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:,:1,:,:]
        # x_in_max = torch.max(image_y,image_ir)
        # loss_in = F.l1_loss(x_in_max,generate_img)

        loss_in = F.l1_loss(image_y, generate_img) + F.l1_loss(image_ir, generate_img)

        y_grad  = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)

        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad    = F.l1_loss(x_grad_joint, generate_img_grad)

        loss_total = loss_in + 10 * loss_grad

        return loss_total, loss_in, loss_grad

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self,x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

        self.loss = nn.L1Loss()

    def forward(self, mask, image_vis, image_ir, generate_img):

        image_y = image_vis[:, :1, :, :]
        loss_ir = self.loss(image_ir, (mask * generate_img + (1 - mask) * image_ir))
        loss_vi = self.loss(image_y, (1 - mask) * generate_img + mask * image_y)

        loss_mask = loss_ir + loss_vi

        return loss_mask
