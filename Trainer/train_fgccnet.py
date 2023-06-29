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
import torch
import torch.backends.cudnn
import torch.nn.functional as F

from dataloader.siam_fus_sod_dataloader import get_loader, test_dataset
from models.fgccnet import FGCCNet
from loss import sodloss
from util.utility import clip_gradient
from warmup_scheduler.scheduler import GradualWarmupScheduler

import logging
import argparse
import warnings
warnings.filterwarnings('ignore')

def create_logger(filepath):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    console_logger = logging.getLogger('ConsoleLogger')
    file_logger = logging.getLogger('FileLogger')

    file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger

def hyper_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--nEpochs', type=int, default=10, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate for SOD')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--lw', type=float, default=0.001, help='weight')
    parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate of learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--rgb_root',     type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/RGB/', help='the training rgb images root')
    parser.add_argument('--thermal_root', type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/T/', help='the training depth images root')
    parser.add_argument('--gt_root',      type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/GT/', help='the training gt images root')
    parser.add_argument('--fus_root',     type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Train/train_fuse_rgb/', help='the training gt images root')

    parser.add_argument('--val_rgb_root',     type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB/', help='the test rgb images root')
    parser.add_argument('--val_thermal_root', type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T/', help='the test depth images root')
    parser.add_argument('--val_gt_root',      type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/', help='the test gt images root')
    parser.add_argument('--val_fus_root',     type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/test_fuse_rgb/', help='the training gt images root')

    opt = parser.parse_args()
    return opt

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

def Train_FGCCNet(opt, file_logger):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    '''Defining ckpt, dataloader, model, optimizer, loss functions for SOD phase'''
    # todo: Creating SavePath of Checkpoints
    cache = '../checkpoint/SOD/xxxxx/'
    if not os.path.exists(cache):
        os.makedirs(cache)

    # todo: dataloader of training and testing
    # training dataset
    train_loader = get_loader(opt.rgb_root, opt.thermal_root, opt.fus_root, opt.gt_root, # opt.rgb_root  opt.thermal_root
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              shuffle=True,
                              split='train'
                              )
    file_logger.info("the training dataset is length:{}".format(len(train_loader)))
    # testing dataset
    val_loader   = test_dataset(opt.val_rgb_root,
                                opt.val_thermal_root,
                                opt.val_fus_root,
                                opt.val_gt_root,
                                opt.trainsize,
                              )
    file_logger.info("the validation dataset is length:{}".format(len(val_loader)))

    # todo: define network, optimizer and loss functions
    sodnet    = FGCCNet().to(device)
    optimizer = torch.optim.Adam(sodnet.parameters(), opt.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nEpochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
    criteria_sod = sodloss.SODLoss().to(device)

    # load pretrained model
    if opt.load != None:
        load_path = os.path.join(cache, 'model_sod.pth')
        sodnet.load_state_dict(torch.load(load_path))
        file_logger.info('===> Loading pretrained model from {} sucessfully~')

    # savepath of best model
    save_path = os.path.join(cache, 'model_sod.pth')

    total_step = len(train_loader)
    best_mae   = 1
    best_epoch = 1

    for epoch in range(1, opt.nEpochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]

        sodnet.train()
        epoch_step = 0
        loss_all = 0
        for i, (rgb, thermal, fus, gt, name) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            rgb = rgb.cuda()
            thermal = thermal.cuda()
            fus = fus.cuda()
            gt = gt.cuda()

            sal_input = torch.cat((rgb, thermal, fus), dim=0)
            s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)

            gt_coarse = F.interpolate(gt, (s_coarse.shape[2], s_coarse.shape[3]), mode='bilinear', align_corners=True)
            gt_coarse = torch.cat((gt_coarse, gt_coarse, gt_coarse), dim=0)
            gt_specific = torch.cat((gt, gt), dim=0)

            loss_coarse = criteria_sod(s_coarse, gt_coarse)
            loss_final = criteria_sod(s_output, gt)
            loss_y = criteria_sod(y, gt)
            loss_specific = criteria_sod(torch.cat((rgb_map, tma_map), dim=0), gt_specific)
            loss_sod = loss_coarse + loss_final + loss_specific + loss_y

            loss = loss_sod
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                file_logger.info('Epoch [{:03d}/{:03d}], LR: {:.2e}, Step [{:04d}/{:04d}], Loss: {:.4f}'.
                                 format(epoch, opt.nEpochs, Decimal(lr), i, total_step, loss.data))
        loss_all /= epoch_step
        file_logger.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.nEpochs, loss_all))

        # todo: validation each epoch
        sodnet.eval()
        with torch.no_grad():
            mae_sum = 0
            for i in range(val_loader.size):
                rgb, thermal, fus, gt, name = val_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                rgb = rgb.cuda()
                thermal = thermal.cuda()
                fus = fus.cuda()

                sal_input = torch.cat((rgb, thermal, fus), dim=0)
                s_coarse, rgb_map, tma_map, y, s_output = sodnet(sal_input)
                res = s_output
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            mae = mae_sum / val_loader.size
            file_logger.info(
                'Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
            if epoch == 1:
                best_mae = mae
                torch.save(sodnet.state_dict(), save_path)
                file_logger.info('best epoch:{}'.format(epoch))
            else:
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    torch.save(sodnet.state_dict(), save_path)
                    file_logger.info('best epoch:{}'.format(epoch))



if __name__ == "__main__":
    opt = hyper_args()
    console_logger, file_logger = create_logger('../log/train.log')
    Train_FGCCNet(opt, file_logger)