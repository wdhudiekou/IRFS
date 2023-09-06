#!/usr/bin/python
# -*- encoding: utf-8 -*-
#-****************************************************************-#
# Author: WangDi
# Email:  diwang1211@mail.dlut.edu.cn or wangdi_1211@njust.edu.cn
#-****************************************************************-#
import sys
sys.path.append("..")
import numpy as np
import os
import cv2
import torch
import kornia
import torch.backends.cudnn
import torchvision.transforms
import torch.utils.data

from dataloader.fusion_dataloader import FuseTrainData, FuseTestData
from models.fsfnet import FSFNet

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
    parser.add_argument('--checkpoint',  type=str,   default='../checkpoint/Fusion+SOD+Inter/IR_FSOD_230318/',   help='save path of model')

    parser.add_argument('--val_rgb_root',        type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/RGB/',      help='the test rgb images root')
    parser.add_argument('--val_thermal_root',    type=str, default='/home/zongzong/WD/Datasets/RGBT/VT5000/Test/T/',    help='the test depth images root')
    parser.add_argument('--fused_test_dir',      type=str, default='../results/Fusion+SOD+Inter/IR_FSOD_230318/Fusion_VT5000/9/',    help='the path to save fused images')
    opt = parser.parse_args()
    return opt

def create_logger(filepath):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    console_logger = logging.getLogger('ConsoleLogger')
    file_logger = logging.getLogger('FileLogger')

    file_handler = logging.FileHandler(filepath, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger


def Test_FSFNet(opt, file_logger):
    # todo: set the device for training
    cuda = opt.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    # todo: path of trained fusion model
    checkpoint = '../checkpoint/Fusion+SOD+Inter/IR_FSOD_230318/9/'
    fusion_model_path = os.path.join(checkpoint, 'fusion_model.pth')

    # todo: savepath of fused results
    if not os.path.exists(opt.fused_test_dir):
        os.makedirs(opt.fused_test_dir)

    # todo: Loading datasets
    crop = torchvision.transforms.RandomResizedCrop(352)
    path_rgb_val     = pathlib.Path(opt.val_rgb_root)
    path_thermal_val = pathlib.Path(opt.val_thermal_root)
    test_data = FuseTestData(path_thermal_val, path_rgb_val, crop)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=False)
    file_logger.info("the testing dataset is length:{}".format(len(test_loader)))

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
    console_logger, file_logger = create_logger('../log/test_fsfnet_230315.log')
    Test_FSFNet(opt, file_logger)