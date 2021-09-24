# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 11:16:24 2021

@author: zidu
"""
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
import utils.selfdeblur_utility
from utils.selfdeblur_utility import save_image, parsing_args, save_final_result
warnings.filterwarnings('ignore')

# define model and parameter choices
param = ["model3_NL0.03_LR0.01"]
#define source images directory
imgs_directory = "datasets/OTF_extraction/20_averaged/"
def self_deblur():
    source_images = os.listdir(imgs_directory)
    print(source_images)
    ## process one input image
    for which_image in range(len(source_images)):
    # process one image at certain condition
        for i in range(len(param)):
            # preparation
            source_image = source_images[which_image]
            opt = parsing_args().parse_args()
            #torch.backends.cudnn.enabled = True
            #torch.backends.cudnn.benchmark =True
            #dtype = torch.cuda.FloatTensor
            dtype = torch.FloatTensor
            files_source = glob.glob(os.path.join(opt.data_path, source_image))
            print(os.path.join(opt.data_path, source_image))
            f = files_source[0]
            print(files_source)
            save_path = opt.save_path
            os.makedirs(save_path, exist_ok=True)
            print(save_path)

            INPUT = 'noise'
            pad = 'reflection'

            # noise level
            reg_noise_std = float(param[i][9:13])
            print("noise level: " + str(reg_noise_std))
            # learning rate
            LR = 0.01
            print("learning rate: " + str(LR))
            # iteration
            num_iter = opt.num_iter
            # model index
            index = int(param[i][5])
            print("model: " + str(index))
            path_to_image = f
            imgname = os.path.basename(f)
            imgname = os.path.splitext(imgname)[0]

            _, imgs = get_image(path_to_image, -1) # load image and convert to np.
            y = np_to_torch(imgs).type(dtype)
            img_size = imgs.shape
            # ######################################################################
            padh, padw = opt.kernel_size[0]-1, opt.kernel_size[1]-1
            opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw
            '''
            x_net:
            '''
            input_depth = 4
            net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

            ## architectures for the generative network
            up_channels_combos = opt.up_channel_combo
            down_channels_combos = opt.down_channel_combo
            skip_channels_combos = opt.skip_channel_combo

            # architecture can be tuned
            net = skip( input_depth, 1,
                        num_channels_down = down_channels_combos[index],
                        num_channels_up   = up_channels_combos[index],
                        num_channels_skip = skip_channels_combos[index],
                        upsample_mode='bilinear', filter_size_down= 3, filter_size_up= 3,
                        need_sigmoid=True, need_relu = False, need_bias=True, pad=pad, act_fun='LeakyReLU')

            print("down/up channel combos: " + str(down_channels_combos[index]) + "\n")
            print("skip channels combos: " + str(skip_channels_combos[index]) + '\n')
            net = net.type(dtype)

            '''
            k_net:
            '''
            #n_k = 200
            n_k = 50
            ## original: kernel initial input is pure noise
            net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
            net_input_kernel.squeeze_()

            net_kernel = fcn(n_k, opt.kernel_size[0]*opt.kernel_size[1])
            net_kernel = net_kernel.type(dtype)

            # Losses: can be tuned or restructured
            mse = torch.nn.MSELoss().type(dtype)
            ssim = SSIM().type(dtype)

            # optimizer
            optimizer = torch.optim.Adam([{'params':net.parameters()},{'params':net_kernel.parameters(),'lr':1e-4}], lr=LR)
            scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma= 1.5)  # learning rates

            # initilization inputs
            net_input_saved = net_input.detach().clone()
            net_input_kernel_saved = net_input_kernel.detach().clone()

            ### start SelfDeblur
            for step in tqdm(range(num_iter)):
                # input regularization
                net_input = net_input_saved + reg_noise_std*torch.zeros(net_input_saved.shape).type_as(net_input_saved.data).normal_()
                # change the learning rate
                scheduler.step(step)
                optimizer.zero_grad()
                # get the network output
                out_x = net(net_input)
                out_k = net_kernel(net_input_kernel)
                out_k_m = out_k.view(-1,1,opt.kernel_size[0],opt.kernel_size[1])
                # print(out_k_m)
                out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

                if step < 3000:
                    total_loss = mse(out_y,y)
                else:
                    total_loss = 1-ssim(out_y, y)
                print("total loss: " + str(total_loss))
                total_loss.backward()
                optimizer.step()

                if (step + 1) % opt.save_frequency == 0:
                    save_image(source_images, which_image, step, opt, out_x, padh, img_size, padw, out_k_m)
                if step == num_iter - 1:
                    imgname_now = source_images[which_image][:-4]
                    save_final_result(net, opt, imgname_now, net_kernel)




start_time = time.time()
self_deblur()
print("Deblurring Execution Time: " + str(time.time() - start_time))