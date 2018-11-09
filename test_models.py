import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch, split_train_val_test
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=30, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=25,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    net = UNet(n_channels=3, n_classes=1)
    """
    checkpoint = torch.load(args.load)
    base_dict = {'.'.join(k.split('.')[7:]): v for k,v in list(checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)
    """
    state_dict =torch.load(args.load)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(args.load))
    net.cuda()
    net.eval()
    dir_img = '/research/pheng5/jzwang/data/resize_train/ISIC-2017_Test_v2_Data/'
    dir_mask = '/research/pheng5/jzwang/data/resize_train/ISIC-2017_Test_v2_Part1_GroundTruth/'
    img_scale = args.scale
    gpu=args.gpu
    ids = get_ids(dir_img)
    ids = split_ids(ids)
    iddataset = split_train_val_test(ids, 0)
    train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
    val_dice,val_jaccard = eval_net(net, train, gpu)
    print('Validation Dice Coeff: {}'.format(val_dice))
    print('Jaccard:: {}'.format(val_jaccard))
