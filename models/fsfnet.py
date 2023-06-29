import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##---------- Spatial Attention ----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

##########################################################################
## ------ Spatial Attention --------------
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


##########################################################################
## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
##---------- Dual Attention Unit ----------
class DualAttention(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualAttention, self).__init__()
        modules_body = [conv(1, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        return res

##########################################################################
##---------- Sobel Edge Unit ----------
class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

##########################################################################
##---------- Dual Attention Edge Unit ----------
class DualEdgeAttention(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8, bias=False, act=nn.PReLU()):
        super(DualEdgeAttention, self).__init__()
        modules_body = [conv(1, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()
        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)
        ## Sobel Edge
        self.Edge = Sobelxy(n_feat)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res_att = torch.cat([sa_branch, ca_branch], dim=1)
        res_att = self.conv1x1(res_att)
        edge = self.Edge(res)
        out  = res_att + edge
        return out

##########################################################################
##------------ Feature Screening-based Fusion Network (FSFNet) ------------
class FSFNet(nn.Module):
    def __init__(self, n_feat=64, kernel_size=3):
        super(FSFNet, self).__init__()
        # Head
        self.head_1 = DualAttention(n_feat, kernel_size)
        self.head_2 = DualAttention(n_feat, kernel_size)

        # Body
        n_resblocks = 2
        m_body = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)
        # Tail
        m_tail = [conv(n_feat, 1, kernel_size)]
        self.tail = nn.Sequential(*m_tail)

        self.act = nn.ReLU(True)

    def forward(self, x, y):
        x_res = self.head_1(x)
        y_res = self.head_2(y)

        res = x_res + y_res
        res = self.body(res)
        out = self.tail(res)

        return out

##########################################################################
##---------- Edge Feature Screening-based Fusion Network (FSFNet) ----------
class AttEdgeFusionNet(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=False, act=nn.PReLU()):
        super(AttEdgeFusionNet, self).__init__()
        # DAU
        self.head_1 = DualEdgeAttention(n_feat, kernel_size)
        self.head_2 = DualEdgeAttention(n_feat, kernel_size)
        # Body
        n_resblocks = 2
        m_body = [ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        self.body = nn.Sequential(*m_body)
        # Tail
        m_tail = [conv(n_feat, 1, kernel_size)]
        self.tail = nn.Sequential(*m_tail)

        self.act = nn.ReLU(True)

    def forward(self, x, y):
        x_res = self.head_1(x)
        y_res = self.head_2(y)

        res = x_res + y_res
        res = self.body(res)
        out = self.tail(res)

        return out

def params_count(model):
  """
  Compute the number of parameters.
  Args:
      model (model): model to count the number of parameters.
  """
  return np.sum([p.numel() for p in model.parameters()]).item()



if __name__ == '__main__':
    x = torch.randn(1, 1, 640, 480).cuda()
    y = torch.randn(1, 1, 640, 480).cuda()
    model = FSFNet(64).cuda()
    outputs = model(x, y)
    model.eval()
    print("Params(M): %.2f" % (params_count(model) / (1000 ** 2)))

    # from thop import profile
    # flops, params = profile(model, inputs=[x, y])
    # print("Params(M): %.2f" % (params / 1e6))
    # print("FLOPs(G): %.4f" % (flops / 1e9))

    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # model = FSFNet(32).cuda()
    # flops = FlopCountAnalysis(model, (x, y))
    # print("FLOPs(G): %.4f" % (flops.total()/1e9))
    # print(parameter_count_table(model))

    import time
    #
    N = 10
    with torch.no_grad():
        for _ in range(N):
            out = model(x, x)

        result = []
        for _ in range(N):
            torch.cuda.synchronize()
            st = time.time()
            for _ in range(N):
                out = model(x, x)
            torch.cuda.synchronize()
            result.append((time.time() - st)/N)
        print("Running Time: {:.3f}s\n".format(np.mean(result)))