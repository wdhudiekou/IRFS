#!/usr/bin/python
# -*- encoding: utf-8 -*-
#-****************************************************************-#
# Author: WangDi
# Email:  diwang1211@mail.dlut.edu.cn or wangdi_1211@njust.edu.cn
#-****************************************************************-#

import sys
sys.path.append("..")
import numpy as np
from decimal import Decimal
import os
import cv2
import torch
import torch.backends.cudnn
import torch.nn.functional as F
import torch.utils.data

from dataloader.siam_fus_sod_dataloader import test_dataset
from models.fgccnet import FGCCNet

import argparse
import warnings
warnings.filterwarnings('ignore')
import time
import statistics

def hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=288, help='training dataset size')
    parser.add_argument('--load', type=str, default='../checkpoint/Fusion+SOD+Inter/IR_FSOD_230426/', help='train from checkpoints') ####
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--test_path', type=str, default='/home/zongzong/WD/Datasets/RGBT/', help='test dataset path')
    parser.add_argument('--fus_root', type=str, default='../datasets/VT5000/Test/Inter_maxGad_0.1Fused_num_sod_230426_3_10/0/',
                        help='the training gt images root')

    opt = parser.parse_args()
    return opt



def test_sod(opt, num=0):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    # todo: define network
    sodnet = FGCCNet().to(device)

    # todo: load model parameters
    load_path = os.path.join(opt.load, str(num), 'model_sod.pth')
    sodnet.load_state_dict(torch.load(load_path))
    print('===> Loading pretrained model from {} sucessfully~')
    sodnet.eval()

    # todo: testing datasets
    # test_datasets = ['VT821','VT1000', 'VT5000']
    test_datasets = ['VT5000']
    path = '../results/Fusion+SOD+Inter/IR_FSOD_230426/'
    for dataset in test_datasets:
        save_path = os.path.join(path, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rgb_root     = opt.test_path + dataset + '/Test/RGB/'
        thermal_root = opt.test_path + dataset + '/Test/T/'
        fus_root     = opt.fus_root
        gt_root      = opt.test_path + dataset + '/Test/GT/'

        test_loader = test_dataset(rgb_root, thermal_root, fus_root, gt_root, opt.testsize)
        reg_time = []
        for i in range(test_loader.size):
            rgb, thermal, fus, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            rgb     = rgb.cuda()
            thermal = thermal.cuda()
            fus     = fus.cuda()

            # fus = 0.5 * rgb + 0.5 * thermal
            sal_input = torch.cat((rgb, thermal, fus), dim=0)
            torch.cuda.synchronize() if str(device) == 'cuda' else None
            start = time.time()
            s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)
            torch.cuda.synchronize() if str(device) == 'cuda' else None
            end = time.time()
            reg_time.append(end - start)

            res = s_output
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('save img to: ', save_path + '/' + str(num) + '/' + name)
            cv2.imwrite(save_path + '/' + str(num) + '/' + name, res * 255)
        print('Test Done!')
        reg_mean = statistics.mean(reg_time[1:])
        print('fuse time (average): {:.4f}'.format(reg_mean))
        print('fps (equivalence): {:.4f}'.format(1. / reg_mean))

    pass

if __name__ == "__main__":
    opt = hyper_args()
    test_sod(opt, num=0)
