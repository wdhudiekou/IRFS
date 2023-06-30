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
import torch.nn.functional as F
import torchvision.transforms
import torch.utils.data

from dataloader.fusion_dataloader import FuseTrainData, FuseTestData, FuseEdgeTrainData
from models.fsfnet import FSFNet
from dataloader.siam_fus_sod_dataloader import get_loader, test_dataset, get_ctd_loader, test_edge_dataset
from models.fgccnet import FGCCNet
from models.ctdnet import CTDNet
from loss import fusion_loss, fusionloss
from loss import sodloss
from util.utility import RGB2YCrCb, YCrCb2RGB, clip_gradient
from warmup_scheduler.scheduler import GradualWarmupScheduler

import pathlib
import logging
import argparse
import random
# from apex import amp

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
    parser.add_argument('--checkpoint',  type=str,   default='../checkpoint/Fusion+SOD+Inter/IR_CTD_FSOD_1110/',   help='save path of model')

    parser.add_argument('--rgb_root',      type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB/',       help='the training rgb images root')
    parser.add_argument('--thermal_root',  type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T/',         help='the training depth images root')
    parser.add_argument('--gt_root',       type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/',        help='the training gt images root')
    parser.add_argument('--gt_edge',       type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/edge/',        help='the training gt images root')
    parser.add_argument('--rgb_map',       type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB_map_soft/', help='the training gt images root')
    parser.add_argument('--thermal_map',   type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T_map_soft/',   help='the training gt images root')


    parser.add_argument('--val_rgb_root',        type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB/',      help='the test rgb images root')
    parser.add_argument('--val_thermal_root',    type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T/',    help='the test depth images root')
    parser.add_argument('--val_gt_root',         type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/',       help='the test gt images root')

    parser.add_argument('--fused_train_dir',     type=str, default='../datasets/VT5000/Train/Inter_Fused_num_sod_CTD_1110_3_30/',    help='the path to save fused images')
    parser.add_argument('--fused_test_dir',      type=str, default='../datasets/VT5000/Test/Inter_Fused_num_sod_CTD_1110_3_30/',    help='the path to save fused images')


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

def train_fusion(opt, file_logger, num=0):

    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    seed = 7
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    '''Defining ckpt, dataloader, model, optimizer, loss functions for Fusion phase'''
    # todo: Creating Save Path of Checkpoints
    checkpoint = opt.checkpoint
    cache = os.path.join(checkpoint, str(num))
    if not os.path.exists(cache):
        os.makedirs(cache)

    # todo: Loading datasets
    path_rgb         = pathlib.Path(opt.rgb_root)
    path_thermal     = pathlib.Path(opt.thermal_root)
    path_gt          = pathlib.Path(opt.gt_root)
    path_edge        = pathlib.Path(opt.gt_edge)
    path_rgb_map     = pathlib.Path(opt.rgb_map)
    path_thermal_map = pathlib.Path(opt.thermal_map)
    crop = torchvision.transforms.RandomResizedCrop(352)
    data = FuseEdgeTrainData(path_thermal, path_rgb, path_gt, path_edge, path_thermal_map, path_rgb_map, crop)
    train_loader = torch.utils.data.DataLoader(dataset=data,
                                               batch_size=opt.batchsize,
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)
    file_logger.info("the training dataset is length:{}".format(len(train_loader)))
    train_loader.n_iter = len(train_loader)

    # todo: define network, optimizer and loss functions
    fusionNet       = FSFNet(64).to(device)
    optimizer       = torch.optim.Adam(params=fusionNet.parameters(), lr=opt.lr)
    criteria_fusion = fusion_loss.FusionLoss().to(device)

    if num > 0:
        # loading pretrained model
        pre_fusion_model = os.path.join(checkpoint, str(num-1), 'fusion_model.pth')
        fusionNet.load_state_dict(torch.load(pre_fusion_model))

    '''Defining ckpt, dataloader, model, optimizer, loss functions for SOD phase'''
    if num>0:
        save_pth = os.path.join(checkpoint, str(num-1), 'model_sod.pth')
        # todo: define model and frozen parameters
        SodNet = CTDNet(mode='train').to(device)
        SodNet.load_state_dict(torch.load(save_pth))
        SodNet.eval()
        for p in SodNet.parameters():
            p.requires_grad = False
        file_logger.info('Load SOD Model {} Sucessfully~'.format(save_pth))
        # todo: define loss functions for SOD
        criteria_sod = sodloss.SODLoss().to(device)

    nEpochs = 3
    st = glob_st = time.time()
    for epoch in range(num * nEpochs + opt.start_epoch, (num + 1) * nEpochs + 1):
        fusionNet.train()

        for it, ((T, YCrCb, GT, edge), (T_path, RGB_path), (T_map, RGB_map)) in enumerate(train_loader):
            gt        = GT.cuda()
            img_t     = T.cuda()
            img_ycrcb = YCrCb.cuda()
            edge      = edge.cuda()


            map_t     = T_map.cuda()
            map_rgb   = RGB_map.cuda()

            img_y = img_ycrcb[:, :1, :, :]
            # fusion
            fus_y = fusionNet(img_y, img_t)
            fusion_ycrcb = torch.cat((torch.clamp(fus_y, 0, 1), img_ycrcb[:, 2:, :, :], img_ycrcb[:, 1:2, :, :]), dim=1)
            fusion_image = kornia.color.ycbcr_to_rgb(fusion_ycrcb) # torch.Size([4, 3, 352, 352])
            # fusion loss
            loss_fusion, loss_in, loss_grad = criteria_fusion(fus_y, img_t, img_y, map_t, map_rgb)
            # sod loss
            if num > 0:
                out1, out_edge, out2, out3, out4, out5 = SodNet(fusion_image)
                loss1 = sodloss.total_loss(out1, gt)
                loss_edge = sodloss.bce_loss(out_edge, edge)
                loss2 = sodloss.total_loss(out2, gt)
                loss3 = sodloss.total_loss(out3, gt)
                loss4 = sodloss.total_loss(out4, gt)
                loss5 = sodloss.total_loss(out5, gt)
                loss_sod = loss1 + loss_edge + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

                # loss_total = loss_fusion + num * loss_sod
                loss_total = loss_fusion + loss_sod
            else:
                loss_total = loss_fusion

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = train_loader.n_iter * (epoch - 1) + it + 1
            eta = int((train_loader.n_iter * epoch - now_it) * (glob_t_intv / (now_it)))
            eta = str(datetime.timedelta(seconds=eta))
            if now_it % 10 == 0:
                if num > 0:
                    loss_sod = loss_sod.item()
                else:
                    loss_sod = 0
                msg = ', '.join(
                    [
                        'Epoch: {epoch}',
                        'step: {it}/{max_it}',
                        'loss_total: {loss_total:.4f}',
                        'loss_in: {loss_in:.4f}',
                        'loss_grad: {loss_grad:.4f}',
                        'loss_sod: {loss_sod:.4f}',
                        'eta: {eta}',
                        'time: {time:.4f}',
                    ]
                ).format(
                    epoch=epoch,
                    it=now_it,
                    max_it=train_loader.n_iter * epoch,
                    loss_total=loss_total.item(),
                    loss_in=loss_in.item(),
                    loss_grad=loss_grad.item(),
                    loss_sod=loss_sod,
                    time=t_intv,
                    eta=eta,
                )
                file_logger.info(msg)
                st = ed
        # save model
        fusion_model_file = os.path.join(cache, 'fusion_model.pth')
        torch.save(fusionNet.state_dict(), fusion_model_file)
        print("Fusion Model Save to: {}".format(fusion_model_file), '\n')
        # save optimizer
        fusion_optim_file = os.path.join(cache, 'optimizer.pt')
        torch.save(optimizer.state_dict(),fusion_optim_file)
        print("Fusion Optimizer Save to: {}".format(fusion_optim_file), '\n')

    pass

def run_fusion(opt, file_logger, num):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    # todo: path of trained fusion model
    checkpoint = opt.checkpoint
    fusion_model_path = os.path.join(checkpoint, str(num), 'fusion_model.pth')

    # todo: savepath of fused results
    fused_train_dir = os.path.join(opt.fused_train_dir, str(num))
    if not os.path.exists(fused_train_dir):
        os.makedirs(fused_train_dir)

    fused_test_dir = os.path.join(opt.fused_test_dir, str(num))
    if not os.path.exists(fused_test_dir):
        os.makedirs(fused_test_dir)

    # todo: Loading datasets
    # for training
    path_rgb         = pathlib.Path(opt.rgb_root)
    path_thermal     = pathlib.Path(opt.thermal_root)
    path_gt          = pathlib.Path(opt.gt_root)
    path_edge        = pathlib.Path(opt.gt_edge)
    path_rgb_map     = pathlib.Path(opt.rgb_map)
    path_thermal_map = pathlib.Path(opt.thermal_map)
    crop = torchvision.transforms.RandomResizedCrop(352)
    train_data = FuseEdgeTrainData(path_thermal, path_rgb, path_gt, path_edge, path_thermal_map, path_rgb_map, crop)
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
        for it, ((T, YCrCb, GT, edge), (T_path, RGB_path), (T_map, RGB_map)) in enumerate(train_loader):
            img_t = T.cuda()
            img_ycrcb = YCrCb.cuda()
            label = GT.cuda()
            edge  = edge.cuda()
            name = T_path[0].split('/')[-1]

            img_y = img_ycrcb[:, :1, :, :]
            # fusion
            fus_y = fusionNet(img_y, img_t)
            # save images
            fus_y = torch.clamp(fus_y, 0, 1)
            fusion_ycrcb = torch.cat((fus_y, img_ycrcb[:, 1:2, :, :], img_ycrcb[:, 2:, :, :]), dim=1)  # torch.Size([4, 3, 352, 352])
            fusion_ycrcb = fusion_ycrcb.squeeze().cpu()
            fusion_ycrcb = (kornia.utils.tensor_to_image(fusion_ycrcb) * 255.).astype(np.uint8)  # (352, 352, 3)
            fusion_image = cv2.cvtColor(fusion_ycrcb, cv2.COLOR_YCrCb2BGR)  # (352, 352, 3)
            save_path = os.path.join(fused_train_dir, name)
            cv2.imwrite(save_path, fusion_image)
        print('Fusion Training Results have been saved in {0} Sucessfully!'.format(fused_train_dir))

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
            save_path = os.path.join(fused_test_dir, name)
            cv2.imwrite(save_path, fusion_image)
        print('Fusion Testing Results have been saved in {0} Sucessfully!'.format(fused_test_dir))


def train_sod(opt, file_logger, num=0):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    '''Defining ckpt, dataloader, model, optimizer, loss functions for SOD phase'''
    # todo: Creating SavePath of Checkpoints
    checkpoint = opt.checkpoint
    cache = os.path.join(checkpoint, str(num))
    if not os.path.exists(cache):
        os.makedirs(cache)

    # todo: Loading datasets
    # training dataset
    fused_train_dir = os.path.join(opt.fused_train_dir, str(num))
    gt_train_dir = '/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/'
    train_loader = get_ctd_loader(fused_train_dir,  gt_train_dir, opt.gt_edge,
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              shuffle=True,
                              split='train'
                              )
    file_logger.info("the training dataset is length:{}".format(len(train_loader)))
    # testing dataset
    fused_test_dir = os.path.join(opt.fused_test_dir, str(num))
    gt_val_dir = '/home/zongzong/WD/Datasets/RGBT/VT5000/Test/GT/'
    val_loader = test_edge_dataset(
                              fused_test_dir,
                              gt_val_dir,
                              opt.trainsize,
                              )
    file_logger.info("the validation dataset is length:{}".format(len(val_loader)))

    # todo: define network, optimizer and loss functions
    SodNet    = CTDNet(mode='train').to(device)
    ## parameter
    enc_params, dec_params = [], []
    for name, param in SodNet.named_parameters():
        if 'bkbone' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    optimizer = torch.optim.SGD([{'params': enc_params}, {'params': dec_params}], lr=0.05, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    # SodNet, optimizer = amp.initialize(SodNet, optimizer, opt_level='O2')

    # optimizer = torch.optim.Adam(SodNet.parameters(), opt.lr_sod)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nEpochs, eta_min=1e-6)
    # scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler)
    criteria_sod = sodloss.SODLoss().to(device)
    # todo: load pretrained model
    if num > 0:
        file_logger.info('=======Loading last stage pretrained SOD model!======')
        load_path = os.path.join(checkpoint, str(num - 1), 'model_sod.pth')
        SodNet.load_state_dict(torch.load(load_path))

    # todo: save path of sod model
    save_path = os.path.join(cache, 'model_sod.pth')

    # TODO: Starting training
    epoch_nums = 40
    best_mae   = 1
    best_epoch = 1
    total_step = len(train_loader)


    for epoch in range(1, epoch_nums + 1):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch) / (epoch_nums) * 2 - 1)) * 0.05 * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch) / (epoch_nums) * 2 - 1)) * 0.05
        # scheduler.step()
        # lr = scheduler.get_lr()[0]

        SodNet.train()
        epoch_step = 0
        loss_all = 0
        for i, (fus, gt, edge, name) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            fus  = fus.cuda()
            gt   = gt.cuda()
            edge = edge.cuda()

            out1, out_edge, out2, out3, out4, out5 = SodNet(fus)


            loss1 = sodloss.total_loss(out1, gt)
            loss_edge = sodloss.bce_loss(out_edge, edge)
            loss2 = sodloss.total_loss(out2, gt)
            loss3 = sodloss.total_loss(out3, gt)
            loss4 = sodloss.total_loss(out4, gt)
            loss5 = sodloss.total_loss(out5, gt)
            loss_sod = loss1 + loss_edge + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

            loss = loss_sod
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            epoch_step += 1
            loss_all += loss.data
            if i % 50 == 0 or i == total_step or i == 1:
                file_logger.info('Epoch [{:03d}/{:03d}], LR: {:.2e}, Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(epoch, epoch_nums, Decimal(optimizer.param_groups[0]['lr']), i, total_step, loss.data))
        loss_all /= epoch_step
        file_logger.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, epoch_nums, loss_all))

        # todo: validation each epoch
        SodNet.eval()
        with torch.no_grad():
            mae_sum = 0
            for i in range(val_loader.size):
                fus, gt, name = val_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                fus = fus.cuda()

                out1, out_edge, out2, out3, out4, out5 = SodNet(fus)
                res = out1
                res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            mae = mae_sum / val_loader.size
            file_logger.info('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
            if epoch == 1:
                best_mae = mae
                torch.save(SodNet.state_dict(), save_path)
                file_logger.info('best epoch:{}'.format(epoch))
            else:
                if mae < best_mae:
                    best_mae = mae
                    best_epoch = epoch
                    torch.save(SodNet.state_dict(), save_path)
                    file_logger.info('best epoch:{}'.format(epoch))


if __name__ == "__main__":
    opt = hyper_args()
    alternate = 10
    console_logger, file_logger = create_logger('../log/IR-FSOD-1110.log')

    for i in range(0, alternate):
        train_fusion(opt, file_logger, i)
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion(opt, file_logger, i)
        print("|{0} Fusion Image Sucessfully~!".format(i + 1))
        train_sod(opt, file_logger, i)
        print("|{0} Train Segmentation Model Sucessfully~!".format(i + 1))
    print("training Done!")