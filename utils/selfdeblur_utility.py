from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import matplotlib
import cv2
import torch
import torch.optim
import glob
from skimage.io import imread
from skimage.io import imsave
import warnings
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM
import warnings

warnings.filterwarnings('ignore')

# save final results
def save_final_result(net, opt, imgname_now, net_kernel):
    torch.save(net, os.path.join(opt.save_path, "%s_x.pth" % imgname_now))
    torch.save(net_kernel, os.path.join(opt.save_path, "%s_k.pth" % imgname_now))
    torch.save(net_kernel.state_dict(), os.path.join(opt.save_path,'k_model_weights.pth'))
    torch.save(net.state_dict(), os.path.join(opt.save_path,'x_model_weights.pth'))

# save image and kernel at saving frequency
def save_image(source_images, which_image, step, opt, out_x, padh, img_size, padw, out_k_m):
    format = 'bmp'
    imgname_now = source_images[which_image][:-4] + "_iter_" + str(step + 1)
    save_path = os.path.join(opt.save_path, '%s_x.bmp'  % imgname_now)
    out_x_np = torch_to_np(out_x)
    out_x_np = out_x_np.squeeze()
    out_x_np = out_x_np[padh // 2:padh // 2 + img_size[1], padw // 2:padw // 2 + img_size[2]]
    imsave(save_path, out_x_np)
    save_path = os.path.join(opt.save_path, '%s_k.bmp' % imgname_now)
    out_k_np = torch_to_np(out_k_m)
    out_k_np = out_k_np.squeeze()
    out_k_np /= np.max(out_k_np)
    # out_k_np *= 255
    out_k_np.astype(np.uint8)
    imsave(save_path, out_k_np)

def parsing_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iter', type=int, default=30, help='number of epochs of training')
    parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
    parser.add_argument('--kernel_size', type=int, default=[17, 17], help='size of blur kernel [height, width]')
    parser.add_argument('--data_path', type=str, default="datasets/OTF_extraction/20_averaged/",
                        help='path to blurry image')
    parser.add_argument('--save_path', type=str, default="results/OTF_extraction/20_averaged/", help='path to save results')
    # source_images[which_image][:-4] + "/" + param[i], help='path to save results')
    parser.add_argument('--save_frequency', type=int, default=10, help='lfrequency to save results')
    parser.add_argument('--down_channel_combo',   default = [[128, 128, 128, 128, 128], [16, 32, 64, 128, 128],
                                    [8, 16, 32, 64, 128], [4, 8, 16, 32], [8, 16, 32, 64, 64], [8, 16, 32, 64, 64]])
    parser.add_argument('--up_channel_combo',   default = [[128, 128, 128, 128, 128], [16, 32, 64, 128, 128],
                                    [8, 16, 32, 64, 128], [4, 8, 16, 32], [8, 16, 32, 64, 64], [8, 16, 32, 64, 64]])
    parser.add_argument('--skip_channel_combo', default = [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4],
                                                           [4, 4, 4, 4], [4, 4, 4, 4, 4], [2, 2, 2, 2, 2]])
    return parser