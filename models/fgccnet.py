import torch
from torch import nn
import torch.nn.functional as F
import time
import numpy as np

from models.resnet34 import ResNet


# =====================================BasicModule============================================ #

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation_=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        # add
        padding = 1
        if dilation_ == 2:
           padding = 2
        elif dilation_ == 4:
           padding = 4
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation_)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class _ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class _DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


# ========= Saliency-enhanced Module ========= #
class SEM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SEM, self).__init__()

        act_fn = nn.ReLU(inplace=True)

        self.layer_rgb = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), act_fn)
        self.layer_t   = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), act_fn)

        self.gamma_rf = nn.Parameter(torch.zeros(1))
        self.gamma_tf = nn.Parameter(torch.zeros(1))

        # self.gamma_rf = 1.0
        # self.gamma_tf = 1.0

        self.layer_fus = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        rgb, t, fus = x[0], x[1], x[2]

        x_rgb = self.layer_rgb(rgb)
        x_t   = self.layer_t(t)
        x_fus = self.layer_fus(fus)

        att_fus = nn.Sigmoid()(x_fus)

        x_rgb_fus = x_rgb.mul(att_fus)
        x_t_fus   = x_t.mul(att_fus)

        x_rgb_en = self.gamma_rf * x_rgb_fus + rgb
        x_t_en   = self.gamma_tf * x_t_fus   + t

        out_rgbt = torch.cat([x_rgb_en, x_t_en], dim=0)

        return out_rgbt, fus

# =========  SiameseEncoder  ========= #
class SiameseEncoder(nn.Module):
    def __init__(self):
        super(SiameseEncoder, self).__init__()

        self.backbone = ResNet(BasicBlock, [3, 4, 6, 3])
        self.load_pretrained_model('../pretrained/resnet34-333f7ec4.pth')
        self.relu = nn.ReLU(inplace=True)

        cp = []
        cp.append(nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1),   self.relu, GCM(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),  self.relu, GCM(64, 64)))
        cp.append(nn.Sequential(nn.Conv2d(256, 96, 3, 1, 1),  self.relu, GCM(96, 64)))
        cp.append(nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), self.relu, GCM(128, 64)))
        self.CP = nn.ModuleList(cp)

        # cross collaboration encoding
        self.att_block2 = FGSE(in_dim=64, sr_ratio=4)
        self.att_block3 = FGSE(in_dim=128, sr_ratio=4)
        self.att_block4 = FGSE(in_dim=256, sr_ratio=2)
        self.att_block5 = FGSE(in_dim=512, sr_ratio=1)

        # self.att_block2 = CCE_LFS(in_dim=64, sr_ratio=4)
        # self.att_block3 = CCE_LFS(in_dim=128, sr_ratio=4)
        # self.att_block4 = CCE_LFS(in_dim=256, sr_ratio=2)
        # self.att_block5 = CCE_LFS(in_dim=512, sr_ratio=1)

        # self.att_block2 = FGSE_C2FT(in_dim=64, sr_ratio=4)
        # self.att_block3 = FGSE_C2FT(in_dim=128, sr_ratio=4)
        # self.att_block4 = FGSE_C2FT(in_dim=256, sr_ratio=2)
        # self.att_block5 = FGSE_C2FT(in_dim=512, sr_ratio=1)

        self.att_block2_1 = SEM(in_dim=64, out_dim=64)
        self.att_block3_1 = SEM(in_dim=128, out_dim=128)
        self.att_block4_1 = SEM(in_dim=256, out_dim=256)
        self.att_block5_1 = SEM(in_dim=512, out_dim=512)

    def load_pretrained_model(self, model_path):
        # resnet pretrained parameter
        pretrained_dict_res = torch.load(model_path)
        res_model_dict = self.backbone.state_dict()
        pretrained_dict_res = {k: v for k, v in pretrained_dict_res.items() if k in res_model_dict}
        res_model_dict.update(pretrained_dict_res)
        self.backbone.load_state_dict(res_model_dict)
        print('=====>Load ResNet-34 Model parameters Sucessfully~')


    def forward(self, x):
        # x: torch.Size([6, 3, 288, 288])
        B = x.shape[0]
        feature_extract = []
        tmp_x = []

        ############################ stage 0 ###########################
        res1 = self.backbone.conv1(x)
        res1 = self.backbone.bn1(res1)
        res1 = self.backbone.relu(res1)
        # tmp_x.append(res1)
        res1 = self.backbone.maxpool(res1)  # torch.Size([6, 64, 72, 72])

        ############################ stage 1 ###########################
        x1 = self.backbone.layer1(res1)  # torch.Size([6, 64, 72, 72])
        x1 = x1.reshape([3, -1] + list(x1.shape[-3:])) # torch.Size([3, 2, 64, 72, 72])
        x1_rgbt, x1_fus = self.att_block2_1(x1) # torch.Size([4, 64, 72, 72]) torch.Size([2, 64, 72, 72])
        x1_rgbt = x1_rgbt.reshape([2, -1] + list(x1_rgbt.shape[-3:])) # torch.Size([2, 2, 64, 72, 72])
        x2 = self.att_block2(x1_rgbt) # torch.Size([4, 64, 72, 72])
        res2 = torch.cat([x2, x1_fus], dim=0) # torch.Size([6, 64, 72, 72])
        tmp_x.append(res2)

        ########################### stage 2 ###########################
        x2 = self.backbone.layer2(res2)  #   torch.Size([6, 128, 36, 36])
        x2 = x2.reshape([3, -1] + list(x2.shape[-3:]))
        x2_rgbt, x2_fus = self.att_block3_1(x2)
        x2_rgbt = x2_rgbt.reshape([2, -1] + list(x2_rgbt.shape[-3:]))
        x3 = self.att_block3(x2_rgbt)
        res3 = torch.cat([x3, x2_fus], dim=0)
        tmp_x.append(res3)

        ############################ stage 3 ###########################
        x3 = self.backbone.layer3(res3)  # torch.Size([6, 256, 18, 18])
        x3 = x3.reshape([3, -1] + list(x3.shape[-3:])) # torch.Size([3, 2, 256, 18, 18])
        x3_rgbt, x3_fus = self.att_block4_1(x3) # torch.Size([4, 256, 18, 18]) torch.Size([2, 256, 18, 18])
        x3_rgbt = x3_rgbt.reshape([2, -1] + list(x3_rgbt.shape[-3:])) # torch.Size([2, 2, 256, 18, 18])
        x4 = self.att_block4(x3_rgbt) # torch.Size([4, 256, 18, 18])
        res4 = torch.cat([x4, x3_fus], dim=0) # torch.Size([6, 256, 18, 18])
        tmp_x.append(res4)

        ############################ stage 4 ###########################
        x4 = self.backbone.layer4(res4)  # torch.Size([6, 512, 9, 9])
        x4 = x4.reshape([3, -1] + list(x4.shape[-3:]))
        x4_rgbt, x4_fus = self.att_block5_1(x4)
        x4_rgbt = x4_rgbt.reshape([2, -1] + list(x4_rgbt.shape[-3:]))
        x5 = self.att_block5(x4_rgbt)
        res5 = torch.cat([x5, x4_fus], dim=0) # torch.Size([6, 512, 9, 9])
        tmp_x.append(res5)

        for i in range(4):
            feature_extract.append(self.CP[i](tmp_x[i]))

        return feature_extract

# =========  Fusion-Guided Saliency-Enhanced module (FGSE)  ========= #
class FGSE(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(FGSE, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim

        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.sr_ratio = sr_ratio
        dim = in_dim

        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim * 2, dim // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1), self.relu)

    def forward(self, x):

        # xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        xr, xd = x[0], x[1]  # torch.Size([2, 64, 72, 72]) torch.Size([2, 64, 72, 72])
        m_batchsize, C, width, height = xr.size()

        query_r = self.query_convrd(xr).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_d = self.norm_k(self.sr_k(xd))
            value_d = self.norm_v(self.sr_v(xd))
            key_d = self.key_convrd(key_d).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
            value_d = self.value_convrd(value_d).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
        else:
            key_d = self.key_convrd(xd).view(m_batchsize, -1, width * height)
            value_d = self.value_convrd(xd).view(m_batchsize, -1, width * height)
        attention_rd = self.softmax(torch.bmm(query_r, key_d))
        out_rd = torch.bmm(value_d, attention_rd.permute(0, 2, 1))
        out_rd = out_rd.view(m_batchsize, C, width, height)

        query_d = self.query_convdr(xd).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_r = self.norm_kk(self.sr_kk(xr))
            value_r = self.norm_vv(self.sr_vv(xr))
            key_r = self.key_convdr(key_r).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
            value_r = self.value_convdr(value_r).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
        else:
            key_r = self.key_convdr(xr).view(m_batchsize, -1, width * height)
            value_r = self.value_convdr(xr).view(m_batchsize, -1, width * height)
        attention_dr = self.softmax(torch.bmm(query_d, key_r))
        out_dr = torch.bmm(value_r, attention_dr.permute(0, 2, 1))
        out_dr = out_dr.view(m_batchsize, C, width, height)

        out_rd = self.gamma_rd * out_rd + xr
        out_dr = self.gamma_dr * out_dr + xd
        out_rd = self.relu(out_rd)
        out_dr = self.relu(out_dr)

        rgb_gap = nn.AvgPool2d(out_rd.shape[2:])(out_rd).view(len(out_rd), C, 1, 1)
        hha_gap = nn.AvgPool2d(out_dr.shape[2:])(out_dr).view(len(out_dr), C, 1, 1)
        stack_gap = torch.cat([rgb_gap, hha_gap], dim=1)
        stack_gap = self.fc1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.fc2(stack_gap)
        rgb_ = stack_gap[:, 0:C, :, :] * out_rd
        hha_ = stack_gap[:, C:2 * C, :, :] * out_dr
        merge_feature = torch.cat([rgb_, hha_], dim=1)
        merge_feature = self.merge_conv1x1(merge_feature)

        rgb_out = (xr + merge_feature) / 2
        hha_out = (xd + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        out_x = torch.cat([rgb_out, hha_out], dim=0)

        return out_x

# =========  FGSE (w/o C2FT)  ========= #
class FGSE_LFS(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(FGSE_LFS, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim

        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.sr_ratio = sr_ratio
        dim = in_dim

        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim * 2, dim // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1), self.relu)

    def forward(self, x):
        # xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        xr, xd = x[0], x[1]  # torch.Size([2, 64, 72, 72]) torch.Size([2, 64, 72, 72])
        m_batchsize, C, width, height = xr.size()

        out_rd  = xr
        out_dr  = xd
        rgb_gap = nn.AvgPool2d(out_rd.shape[2:])(out_rd).view(len(out_rd), C, 1, 1)
        hha_gap = nn.AvgPool2d(out_dr.shape[2:])(out_dr).view(len(out_dr), C, 1, 1)
        stack_gap = torch.cat([rgb_gap, hha_gap], dim=1)
        stack_gap = self.fc1(stack_gap)
        stack_gap = self.relu(stack_gap)
        stack_gap = self.fc2(stack_gap)
        rgb_ = stack_gap[:, 0:C, :, :] * out_rd
        hha_ = stack_gap[:, C:2 * C, :, :] * out_dr
        merge_feature = torch.cat([rgb_, hha_], dim=1)
        merge_feature = self.merge_conv1x1(merge_feature)

        rgb_out = (xr + merge_feature) / 2
        hha_out = (xd + merge_feature) / 2
        rgb_out = self.relu1(rgb_out)
        hha_out = self.relu2(hha_out)

        out_x = torch.cat([rgb_out, hha_out], dim=0)

        return out_x

# =========  FGSE (w/o LFS)  ========= #
class FGSE_C2FT(nn.Module):
    def __init__(self, in_dim=2048, sr_ratio=1):
        super(FGSE_C2FT, self).__init__()
        input_dim = in_dim
        self.chanel_in = input_dim

        self.query_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convrd   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convrd = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.query_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.key_convdr   = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)
        self.value_convdr = nn.Conv2d(in_channels=input_dim, out_channels=input_dim, kernel_size=1)

        self.sr_ratio = sr_ratio
        dim = in_dim

        if sr_ratio > 1:
            self.sr_k = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_v = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_k = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_v = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

            self.sr_kk = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_vv = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_kk = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)
            self.norm_vv = nn.BatchNorm2d(dim, eps=1e-05, momentum=0.1, affine=True)

        self.gamma_rd = nn.Parameter(torch.zeros(1))
        self.gamma_dr = nn.Parameter(torch.zeros(1))
        self.gamma_x = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Conv2d(dim * 2, dim // 2, kernel_size=1)
        self.fc2 = nn.Conv2d(dim // 2, dim * 2, kernel_size=1)
        self.merge_conv1x1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1), self.relu)

    def forward(self, x):

        # xr, xd = x[0].unsqueeze(dim=0), x[1].unsqueeze(dim=0)
        xr, xd = x[0], x[1]  # torch.Size([2, 64, 72, 72]) torch.Size([2, 64, 72, 72])
        m_batchsize, C, width, height = xr.size()

        query_r = self.query_convrd(xr).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_d = self.norm_k(self.sr_k(xd))
            value_d = self.norm_v(self.sr_v(xd))
            key_d = self.key_convrd(key_d).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
            value_d = self.value_convrd(value_d).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
        else:
            key_d = self.key_convrd(xd).view(m_batchsize, -1, width * height)
            value_d = self.value_convrd(xd).view(m_batchsize, -1, width * height)
        attention_rd = self.softmax(torch.bmm(query_r, key_d))
        out_rd = torch.bmm(value_d, attention_rd.permute(0, 2, 1))
        out_rd = out_rd.view(m_batchsize, C, width, height)

        query_d = self.query_convdr(xd).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        if self.sr_ratio > 1:
            key_r = self.norm_kk(self.sr_kk(xr))
            value_r = self.norm_vv(self.sr_vv(xr))
            key_r = self.key_convdr(key_r).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
            value_r = self.value_convdr(value_r).view(m_batchsize, -1, width // self.sr_ratio * height // self.sr_ratio)
        else:
            key_r = self.key_convdr(xr).view(m_batchsize, -1, width * height)
            value_r = self.value_convdr(xr).view(m_batchsize, -1, width * height)
        attention_dr = self.softmax(torch.bmm(query_d, key_r))
        out_dr = torch.bmm(value_r, attention_dr.permute(0, 2, 1))
        out_dr = out_dr.view(m_batchsize, C, width, height)

        out_rd = self.gamma_rd * out_rd + xr
        out_dr = self.gamma_dr * out_dr + xd
        out_rd = self.relu(out_rd)
        out_dr = self.relu(out_dr)

        rgb_out = out_rd
        hha_out = out_dr
        out_x = torch.cat([rgb_out, hha_out], dim=0)

        return out_x

# ========= Global Contextual module (GCM) ========= #
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation_scale(nn.Module):
    def __init__(self, in_dim, out_dim, dilation=[1, 2, 3], residual=False):
        super(aggregation_scale, self).__init__()

        if in_dim == out_dim:
            residual = True
        self.use_res_connect = residual
        mid_dim = out_dim * 2

        self.conv1 = BasicConv2d(in_dim, mid_dim, kernel_size=1)
        self.hidden_conv1 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=1, groups=mid_dim, dilation=1)
        self.hidden_conv2 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=2, groups=mid_dim, dilation=2)
        self.hidden_conv3 = nn.Conv2d(mid_dim, mid_dim, kernel_size=3, padding=3, groups=mid_dim, dilation=3)

        self.hidden_bnact = nn.Sequential(nn.BatchNorm2d(mid_dim), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(mid_dim, out_dim, 1, 1, 0, bias=False))

    def forward(self, input_):
        x = self.conv1(input_)
        x1 = self.hidden_conv1(x)
        x2 = self.hidden_conv2(x)
        x3 = self.hidden_conv3(x)
        intra = self.hidden_bnact(x1 + x2 + x3)
        output = self.out_conv(intra)

        if self.use_res_connect:
            output = input_ + output

        return output

# ========== SPLayer =========== #
class SPLayer(nn.Module):
    def __init__(self):
        super(SPLayer, self).__init__()

    def forward(self, list_x):

        layer0 = list_x[0].reshape([3,-1] + list(list_x[0].shape[-3:]))
        rgb0 = layer0[0]
        tma0 = layer0[1]
        fus0 = layer0[2]

        layer1 = list_x[1].reshape([3, -1] + list(list_x[1].shape[-3:]))
        rgb1 = layer1[0]
        tma1 = layer1[1]
        fus1 = layer1[2]

        layer2 = list_x[2].reshape([3, -1] + list(list_x[2].shape[-3:]))
        rgb2 = layer2[0]
        tma2 = layer2[1]
        fus2 = layer2[2]

        layer3 = list_x[3].reshape([3, -1] + list(list_x[3].shape[-3:]))
        rgb3 = layer3[0]
        tma3 = layer3[1]
        fus3 = layer3[2]

        return rgb0, rgb1, rgb2, rgb3, tma0, tma1, tma2, tma3, fus0, fus1, fus2, fus3

# ========== FGCCNet =========== #
class FGCCNet(nn.Module):
    def __init__(self):
        super(FGCCNet, self).__init__()
        self.JLModule = SiameseEncoder()

        self.cm = SPLayer()
        self.score_JL = nn.Conv2d(64, 1, 1, 1)

        channel = 32
        self.rfb2_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.rfb3_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.rfb4_1 = nn.Conv2d(64, channel, 1, padding=0)
        # self.agg1 = aggregation_cross_v1(channel)
        self.agg1 = aggregation_cross(channel)

        self.thm2_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.thm3_1 = nn.Conv2d(64, channel, 1, padding=0)
        self.thm4_1 = nn.Conv2d(64, channel, 1, padding=0)
        # self.thm_agg1 = aggregation_cross_v1(channel)
        self.thm_agg1 = aggregation_cross(channel)

        self.conv_s_f = BasicConv(2 * channel, channel, kernel_size=3, padding=1)

        self.inplanes = 32
        self.agant1 = self._make_agant_layer(32, 32)
        self.agant2 = self._make_agant_layer(32, 32)
        self.deconv1 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.deconv2 = self._make_transpose(TransBasicBlock, 32, 3, stride=2)
        self.out2_conv = nn.Conv2d(32 * 1, 1, kernel_size=1, stride=1, bias=True)

        self.upsample  = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.JLModule(x)

        rgb2, rgb3, rgb4, rgb5, tma2, tma3, tma4, tma5, fus2, fus3, fus4, fus5 = self.cm(x)
        s_coarse = self.score_JL(x[3])  # torch.Size([6, 1, 9, 9])

        rgb3_1 = self.rfb2_1(rgb3)
        rgb4_1 = self.rfb3_1(rgb4)
        rgb5_1 = self.rfb4_1(rgb5)
        rgb_fea, rgb_map = self.agg1(rgb5_1, rgb4_1, rgb3_1) # torch.Size([1, 32, 36, 36]) torch.Size([1, 1, 36, 36])
        y_rgb = rgb_map

        tma3_1 = self.thm2_1(tma3)
        tma4_1 = self.thm3_1(tma4)
        tma5_1 = self.thm4_1(tma5)
        tma_fea, tma_map = self.thm_agg1(tma5_1, tma4_1, tma3_1) # # torch.Size([1, 32, 36, 36]) torch.Size([1, 1, 36, 36])
        y_thm = tma_map

        y = self.upsample2(rgb_fea + tma_fea)
        y = self.agant1(y)
        y = self.deconv1(y)
        y = self.agant2(y)
        y = self.deconv2(y)
        y = self.out2_conv(y)

        s_output = self.upsample(rgb_map) + self.upsample(tma_map) + y

        return s_coarse, self.upsample(rgb_map), self.upsample(tma_map), y, s_output

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes), )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes), )
        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes
        return nn.Sequential(*layers)

class aggregation_cross(nn.Module):
    def __init__(self, channel):
        super(aggregation_cross, self).__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)
        self.conv5 = nn.Conv2d(channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        x_k = self.conv4(x3_2)
        x = self.conv5(x_k)
        return x_k, x





class DB_b(nn.Module):
    """
    DB_b:
        Decoder Block, 利用融合后的底层深度特征产生边缘预测图.
    """
    def __init__(self, in_c):
        super(DB_b, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_c, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.Sigmoid())
        self.get_pred = Prediction(32)

    def forward(self, feat, up_feat):
        _, _, H, W = feat.shape
        if up_feat is not None:
            up_feat = resize(up_feat, [H, W])
            feat = torch.cat([feat, up_feat], dim=1)
        feat = self.conv(feat)
        pred = self.get_pred(feat)
        return feat, pred

class Prediction(nn.Module):
    """
    Prediction:
        将输入特征的通道压缩到1维, 然后利用sigmoid函数产生预测图.
    """
    def __init__(self, in_c):
        super(Prediction, self).__init__()
        self.pred = nn.Sequential(nn.Conv2d(in_c, 1, 1), nn.Sigmoid())

    def forward(self, input):
        pred = self.pred(input)
        return pred

def resize(input, target_size=(288, 288)):
    """
    resize:
        将tensor (shape=[N, C, H, W]) 双线性放缩到 "target_size" 大小 (默认: 288*288).
    """
    return F.interpolate(input, (target_size[0], target_size[1]), mode='bilinear', align_corners=True)

class TransBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, **kwargs):
        super(TransBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if upsample is not None and stride != 1:
            self.conv2 = nn.ConvTranspose2d(inplanes, planes,
                                            kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
        else:
            self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out += residual
        out = self.relu(out)
        return out

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()


if __name__ == '__main__':

    # x = torch.randn(1, 3, 480, 640).cuda()
    # y = torch.randn(1, 3, 480, 640).cuda()
    # z = torch.randn(1, 3, 480, 640).cuda()
    # input = torch.cat([x, y, z], dim=0)
    # model = FGCCNet().cuda()
    # model.eval()
    # print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))
    # s_coarse, rgb_map, tma_map, y, s_output = model(input)
    # print(s_coarse.shape)
    # print(s_output.shape)

    # from thop import profile
    # flops, params = profile(model, inputs=input)
    # print("Params(M): %.2f" % (params / 1e6))
    # print("FLOPs(G): %.4f" % (flops / 1e9))

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # flops = FlopCountAnalysis(model, (input,))
    # print("FLOPs(G): %.4f" % (flops.total() / 1e9))
    # print(parameter_count_table(model))

    import time

    x = torch.randn(1, 3, 480, 640).cuda()
    y = torch.randn(1, 3, 480, 640).cuda()
    z = torch.randn(1, 3, 480, 640).cuda()
    input = torch.cat([x, y, z], dim=0)
    model = FGCCNet().cuda()
    model.eval()
    #
    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(input)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(input)
            torch.cuda.synchronize()
            result.append((time.time() - st) / N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))