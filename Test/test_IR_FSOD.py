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
import torchvision.transforms
import torch.utils.data

from dataloader.fusion_dataloader import FuseTrainData, FuseTestData
from dataloader.siam_fus_sod_dataloader import test_dataset
from models.fsfnet import FSFNet
from models.fgccnet import FGCCNet

import kornia
import pathlib
import argparse
import warnings
import logging
warnings.filterwarnings('ignore')

def hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=288, help='training dataset size')
    parser.add_argument('--load', type=str, default='../checkpoint/Fusion+SOD+Inter/IR_FSOD_1104/', help='train from checkpoints')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--test_path', type=str, default='/home/zongzong/WD/Datasets/RGBT/', help='test dataset path')
    parser.add_argument('--fus_root', type=str, default='../datasets/VT5000/Test/Inter_Fused_num_sod_1104_3_10/',
                        help='the training gt images root')

    opt = parser.parse_args()
    return opt

def test_fusion(opt, num):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    # todo: path of trained fusion model
    checkpoint = '../checkpoint/Fusion+SOD+Inter/IR_FSOD_1104/'
    fusion_model_path = os.path.join(checkpoint, str(num), 'fusion_model.pth')

    # todo: define network and load parameters
    fusionNet = FSFNet(64).to(device)
    fusionNet.load_state_dict(torch.load(fusion_model_path))
    logging.info('loading trained fusion model done!')

    # todo: savepath of fused results
    path = '../results/Fusion+SOD+Inter/Inter_Fused_num_sod_1104_3_10/VT5000/fusion/'
    save_path = os.path.join(path, str(num))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # todo: Loading testing datasets
    test_path = '/home/zongzong/WD/Datasets/RGBT/VT5000/Test'
    val_rgb_root = test_path + '/RGB/'
    val_thermal_root = test_path + '/T/'
    path_rgb_val     = pathlib.Path(val_rgb_root)
    path_thermal_val = pathlib.Path(val_thermal_root)
    crop = torchvision.transforms.RandomResizedCrop(352)
    test_data   = FuseTestData(path_thermal_val, path_rgb_val, crop)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=False)
    for it, ((T, YCrCb), (T_path, RGB_path)) in enumerate(test_loader):
        img_ycrcb = YCrCb.cuda()
        img_t = T.cuda()
        name = T_path[0].split('/')[-1]

        img_y = img_ycrcb[:, :1, :, :]
        # fusion
        fus_y = fusionNet(img_y, img_t)
        # save images
        fus_y = torch.clamp(fus_y, 0, 1)
        fusion_ycrcb = torch.cat((fus_y, img_ycrcb[:, 1:2, :, :], img_ycrcb[:, 2:, :, :]), dim=1)
        fusion_ycrcb = fusion_ycrcb.squeeze().cpu()
        fusion_ycrcb = (kornia.utils.tensor_to_image(fusion_ycrcb) * 255.).astype(np.uint8)
        fusion_image = cv2.cvtColor(fusion_ycrcb, cv2.COLOR_YCrCb2BGR)
        file_path = os.path.join(save_path, name)
        print('Fusion Testing Results have been saved in {0} Sucessfully!'.format(file_path))
        cv2.imwrite(file_path, fusion_image)
    print('Test Done!')


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
    print('===> Loading pretrained model from {} sucessfully~'.format(load_path))
    sodnet.eval()

    # todo: testing datasets
    # test_datasets = ['VT821','VT1000', 'VT5000']
    test_datasets = ['VT5000']
    path = '../results/Fusion+SOD+Inter/Inter_Fused_num_sod_1104_3_10/'
    for dataset in test_datasets:
        save_path = os.path.join(path, dataset, str(num))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        rgb_root     = opt.test_path + dataset + '/Test/RGB/'
        thermal_root = opt.test_path + dataset + '/Test/T/'
        fus_root     = opt.fus_root  + str(num)
        gt_root      = opt.test_path + dataset + '/Test/GT/'

        test_loader = test_dataset(rgb_root, thermal_root, fus_root, gt_root, opt.testsize)
        for i in range(test_loader.size):
            rgb, thermal, fus, gt, name = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            rgb     = rgb.cuda()
            thermal = thermal.cuda()
            fus     = fus.cuda()
            sal_input = torch.cat((rgb, thermal, fus), dim=0)
            s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)

            res = s_output
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)


            print('save img to: ', save_path + '/' +name)
            cv2.imwrite(save_path + '/' + name, res * 255)
        print('Test Done!')

    pass

if __name__ == "__main__":
    opt = hyper_args()
    test_fusion(opt, num=0)
    test_sod(opt, num=0)
