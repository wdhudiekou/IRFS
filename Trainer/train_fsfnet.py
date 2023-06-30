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
import datetime
import time
import os
import cv2
import torch
import kornia
import torch.backends.cudnn
import torchvision.transforms
import torch.utils.data

from dataloader.fusion_dataloader import FuseTrainData, FuseTestData
from models.fsfnet import FSFNet
from loss import fusion_loss

import pathlib
import logging
import argparse

import warnings
warnings.filterwarnings('ignore')


def hyper_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs',     type=int,   default=100,   help='epoch number')
    parser.add_argument('--lr',          type=float, default=1e-3,  help='learning rate')
    parser.add_argument('--lr_sod',      type=float, default=5e-5,  help='learning rate for SOD')
    parser.add_argument('--batchsize',   type=int,   default=4,    help='training batch size')
    parser.add_argument('--trainsize',   type=int,   default=352,   help='training dataset size')
    parser.add_argument('--clip',        type=float, default=0.5,   help='gradient clipping margin')
    parser.add_argument('--lw',          type=float, default=0.001, help='weight')
    parser.add_argument('--lr_decay',    type=float, default=0.5,   help='decay rate of learning rate')
    parser.add_argument('--decay_rate',  type=float, default=0.1,   help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,   default=60,    help='every n epochs decay learning rate')
    parser.add_argument('--load',        type=str,   default=None,  help='train from checkpoints')
    parser.add_argument("--cuda",        action="store_false", help="Use cuda?")
    parser.add_argument('--gpu_id',      type=str,   default='0',   help='train use gpu')

    parser.add_argument('--alternate',   type=int,   default=6,   help='alternate training times')
    parser.add_argument('--checkpoint',  type=str,   default='../checkpoint/Fusion+SOD+Inter/attFus_spSod_0712/',   help='save path of model')

    parser.add_argument('--rgb_root',      type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB/',       help='the training rgb images root')
    parser.add_argument('--thermal_root',  type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T/',         help='the training depth images root')
    parser.add_argument('--gt_root',       type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/GT/',        help='the training gt images root')
    parser.add_argument('--rgb_map',       type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB_map_soft/', help='the training gt images root')
    parser.add_argument('--thermal_map',   type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T_map_soft/',   help='the training gt images root')


    parser.add_argument('--val_rgb_root',        type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB/',      help='the test rgb images root')
    parser.add_argument('--val_thermal_root',    type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T/',    help='the test depth images root')
    parser.add_argument('--val_gt_root',         type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/',       help='the test gt images root')

    parser.add_argument('--fused_test_dir',      type=str, default='../Results/Fusion/VT5000/Fused_0912_3_10/',    help='the path to save fused images')
    opt = parser.parse_args()
    return opt

def create_logger(filepath):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    console_logger = logging.getLogger('ConsoleLogger')
    file_logger = logging.getLogger('FileLogger')

    file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr_current = opt.lr * opt.lr_decay ** (epoch - 1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_current



def Train_FSFNet(opt, file_logger):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    '''Defining ckpt, dataloader, model, optimizer, loss functions for Fusion phase'''
    # todo: Creating Save Path of Checkpoints
    checkpoint = '../checkpoint/Fusion/FSFNet/'
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

    # todo: Loading datasets
    path_rgb = pathlib.Path(opt.rgb_root)
    path_thermal = pathlib.Path(opt.thermal_root)
    path_gt = pathlib.Path(opt.gt_root)
    path_rgb_map = pathlib.Path(opt.rgb_map)
    path_thermal_map = pathlib.Path(opt.thermal_map)
    crop = torchvision.transforms.RandomResizedCrop(352)
    data = FuseTrainData(path_thermal, path_rgb, path_gt, path_thermal_map, path_rgb_map, crop)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=opt.batchsize,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)
    file_logger.info("the training dataset is length:{}".format(len(train_loader)))
    train_loader.n_iter = len(train_loader)

    # todo: define network, optimizer and loss functions
    fusionNet = FSFNet(64).to(device)
    optimizer = torch.optim.Adam(params=fusionNet.parameters(), lr=opt.lr)
    criteria_fusion = fusion_loss.FusionLoss().to(device)

    # TODO: Starting training
    nEpochs = 30
    st = glob_st = time.time()
    for epoch in range(opt.start_epoch, nEpochs + 1):
        fusionNet.train()
        for it, ((T, YCrCb, GT), (T_path, RGB_path), (T_map, RGB_map)) in enumerate(train_loader):
            gt = GT.cuda()
            img_t = T.cuda()
            img_ycrcb = YCrCb.cuda()

            map_t = T_map.cuda()
            map_rgb = RGB_map.cuda()

            img_y = img_ycrcb[:, :1, :, :]
            # fusion
            fus_y = fusionNet(img_y, img_t)
            fusion_ycrcb = torch.cat((torch.clamp(fus_y, 0, 1), img_ycrcb[:, 2:, :, :], img_ycrcb[:, 1:2, :, :]), dim=1)
            fusion_image = kornia.color.ycbcr_to_rgb(fusion_ycrcb)
            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(fus_y, img_t, img_y, map_t, map_rgb)

            loss = loss_fusion

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * (epoch - 1) + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                msg = ', '.join(
                    [
                        'Epoch: {epoch}',
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    epoch=epoch,
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    time=t_intv,
                    eta=eta,
                )
                file_logger.info(msg)
                st = ed
        # save model
        fusion_model_file = os.path.join(checkpoint, 'fusion_model.pth')
        torch.save(fusionNet.state_dict(), fusion_model_file)
        print("Fusion Model Save to: {}".format(fusion_model_file), '\n')
        # save optimizer
        fusion_optim_file = os.path.join(checkpoint, 'optimizer.pt')
        torch.save(optimizer.state_dict(), fusion_optim_file)
        print("Fusion Optimizer Save to: {}".format(fusion_optim_file), '\n')



def Test_FSFNet(opt, file_logger):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    # todo: path of trained fusion model
    checkpoint = '../checkpoint/Fusion/FSFNet/'
    fusion_model_path = os.path.join(checkpoint, 'fusion_model.pth')

    # todo: savepath of fused results
    if not os.path.exists(opt.fused_test_dir):
        os.makedirs(opt.fused_test_dir)

    # todo: Loading datasets
    # for training
    path_rgb         = pathlib.Path(opt.rgb_root)
    path_thermal     = pathlib.Path(opt.thermal_root)
    path_gt          = pathlib.Path(opt.gt_root)
    path_rgb_map     = pathlib.Path(opt.rgb_map)
    path_thermal_map = pathlib.Path(opt.thermal_map)
    crop = torchvision.transforms.RandomResizedCrop(352)
    train_data = FuseTrainData(path_thermal, path_rgb, path_gt, path_thermal_map, path_rgb_map, crop)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=False)
    file_logger.info("the testing dataset is length:{}".format(len(train_loader)))
    # for testing
    path_rgb_val     = pathlib.Path(opt.val_rgb_root)
    path_thermal_val = pathlib.Path(opt.val_thermal_root)
    test_data = FuseTestData(path_thermal_val, path_rgb_val, crop)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=False)
    file_logger.info("the testing dataset is length:{}".format(len(train_loader)))

    # todo: define network and load parameters
    fusionNet = FSFNet(64).to(device)
    fusionNet.load_state_dict(torch.load(fusion_model_path))
    logging.info('loading trained fusion model done!')

    fusionNet.eval()
    with torch.no_grad():
        # todo: fuse validation dataset
        for it, ((T, YCrCb), (T_path, RGB_path)) in enumerate(test_loader):
            img_ycrcb = YCrCb.cuda()
            img_t = T.cuda()
            name = T_path[0].split('/')[-1]

            img_y = img_ycrcb[:, :1, :, :]
            # fusion
            fus_y = fusionNet(img_y, img_t)
            # save images
            fus_y = torch.clamp(fus_y, 0, 1)
            fusion_ycrcb = torch.cat((fus_y, img_ycrcb[:, 1:2, :, :], img_ycrcb[:, 2:, :, :]), dim=1)  # torch.Size([1, 3, 352, 352])
            fusion_ycrcb = fusion_ycrcb.squeeze().cpu()
            fusion_ycrcb = (kornia.utils.tensor_to_image(fusion_ycrcb) * 255.).astype(np.uint8)  # (352, 352, 3)
            fusion_image = cv2.cvtColor(fusion_ycrcb, cv2.COLOR_YCrCb2BGR)
            save_path = os.path.join(opt.fused_test_dir, name)
            cv2.imwrite(save_path, fusion_image)
        print('Fusion Testing Results have been saved in {0} Sucessfully!'.format(opt.fused_test_dir))



if __name__ == "__main__":
    opt = hyper_args()
    console_logger, file_logger = create_logger('../log/train_fsfnet_0912.log')
    Train_FSFNet(opt, file_logger)
    Test_FSFNet(opt, file_logger)