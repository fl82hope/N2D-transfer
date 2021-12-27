import argparse
import os
import torch
import numpy as np
import cv2

import utils
import dataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Saving, and loading parameters
    parser.add_argument('--save_name', type = str, default = './results', help = 'save the generated with certain epoch')
    parser.add_argument('--load_name', type = str, default = './models/KPN_single_image_epoch120_bs16_mu0_sigma30.pth', help = 'load the pre-trained model with certain epoch')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'size of the batches')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'number of workers')
    # Initialization parameters
    parser.add_argument('--color', type = bool, default = True, help = 'input type')
    parser.add_argument('--burst_length', type = int, default = 1, help = 'number of photos used in burst setting')
    parser.add_argument('--blind_est', type = bool, default = True, help = 'variance map')
    parser.add_argument('--kernel_size', type = list, default = [5], help = 'kernel size')
    parser.add_argument('--sep_conv', type = bool, default = False, help = 'simple output type')
    parser.add_argument('--channel_att', type = bool, default = False, help = 'channel wise attention')
    parser.add_argument('--spatial_att', type = bool, default = False, help = 'spatial wise attention')
    parser.add_argument('--upMode', type = str, default = 'bilinear', help = 'upMode')
    parser.add_argument('--core_bias', type = bool, default = False, help = 'core_bias')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'initialization type of generator')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, \
        default = 'E:\\Deblur\\Short-Long RGB to RGB Mapping\\data\\mobile_phone\\logn8_short1_20200601\\2019_07_27_15_18_27_360', \
            help = 'images baseroot')
    parser.add_argument('--crop', type = bool, default = True, help = 'whether to crop input images')
    parser.add_argument('--crop_size', type = int, default = 512, help = 'single patch size')
    parser.add_argument('--geometry_aug', type = bool, default = False, help = 'geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type = bool, default = False, help = 'geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type = float, default = 1, help = 'min scaling factor')
    parser.add_argument('--scale_max', type = float, default = 1, help = 'max scaling factor')
    parser.add_argument('--add_noise', type = bool, default = False, help = 'whether to add noise to input images')
    parser.add_argument('--mu', type = int, default = 0, help = 'Gaussian noise mean')
    parser.add_argument('--sigma', type = int, default = 30, help = 'Gaussian noise variance: 30 | 50 | 70')
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    test_dataset = dataset.DenoisingValDataset(opt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    sample_folder = opt.save_name
    utils.check_path(sample_folder)

    # forward
    for i, (true_input, true_target) in enumerate(test_loader):

        # To device
        true_input = true_input.cuda()
        true_target = true_target.cuda()

        # Forward propagation
        with torch.no_grad():
            fake_target = generator(true_input, true_input)
            fake_target = fake_target.unsqueeze(0)

        # Save
        print('The %d-th iteration' % (i))
        img_list = [true_input, fake_target, true_target]
        name_list = ['in', 'pred', 'gt']
        utils.save_sample_png(sample_folder = sample_folder, sample_name = '%d' % (i + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        
