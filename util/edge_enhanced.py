import os
import kornia
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2




def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def joint_grad(rgb, t):
    img_vi = rgb
    img_ir = t
    ir_grad = torch.abs(kornia.filters.laplacian(img_ir, 7))
    vi_grad = torch.abs(kornia.filters.laplacian(img_vi, 7))
    max_grad = torch.max(ir_grad, vi_grad)
    return max_grad

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

def Edge_enhanced(sobel, rgb_root, the_root, edge_root, enhance_root):
    rgb_file_names = sorted(os.listdir(rgb_root))
    the_file_names = sorted(os.listdir(the_root))

    if not os.path.exists(edge_root):
        os.makedirs(edge_root)
    if not os.path.exists(enhance_root):
        os.makedirs(enhance_root)

    for idx, (rgb_filename, t_filename) in enumerate(zip(rgb_file_names, the_file_names)):
        rgb_path = os.path.join(rgb_root, rgb_filename)
        t_path   = os.path.join(the_root, t_filename)
        img_rgb = transforms.ToTensor()(binary_loader(rgb_path)).unsqueeze(0)
        img_t   = transforms.ToTensor()(binary_loader(t_path)).unsqueeze(0)
        max_grad = joint_grad(img_rgb, img_t)

        # rgb_grad = sobel(img_rgb)
        # t_grad   = sobel(img_t)
        # max_grad = torch.max(rgb_grad, t_grad)
        # print(max_grad.shape)
        # exit(00)


        enhance_t = img_t + 1*max_grad

        edge = max_grad.squeeze().cpu()
        edge = (kornia.utils.tensor_to_image(edge) * 255.).astype(np.uint8)  # (352, 352, 3)
        save_path = os.path.join(edge_root, rgb_filename)
        cv2.imwrite(save_path, edge)

        enhance = enhance_t.squeeze().cpu()
        enhance = (kornia.utils.tensor_to_image(enhance) * 255.).astype(np.uint8)  # (352, 352, 3)
        save_path = os.path.join(enhance_root, rgb_filename)
        cv2.imwrite(save_path, enhance)

        # print(max_grad.shape)
        # exit(00)








if __name__ == '__main__':
    rgb_root  = '/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB_RE/'
    the_root  = '/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T_RE/'
    edge_root = '/home/zongzong/WD/Datasets/RGBT/VT5000/Train/max_edge/'
    enhance_root = '/home/zongzong/WD/Datasets/RGBT/VT5000/Train/enhanced/'

    sobel = Sobelxy(1)

    Edge_enhanced(sobel, rgb_root, the_root, edge_root, enhance_root)