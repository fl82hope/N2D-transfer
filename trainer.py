# -*- coding:utf-8 -*-
import sys

sys.path.append('./det_faster/lib/')

import time
import datetime
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import dataset
import utils
import torch.nn.functional as F
import tqdm
from transfer import WCT2, get_all_transfer
from utils_WCT.io import Timer, open_image
from PIL import Image
from torchvision.utils import make_grid, save_image
import cv2
from dataset import RandomCropTensor
from model.utils.config import cfg_from_file, cfg, cfg_from_list
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.faster_rcnn.vgg16 import vgg16
# from faster_rcnn.lib.model.nms.nms_wrapper import nms
from model.utils.config import cfg
import torchvision
from model.rpn.bbox_transform import bbox_overlaps_batch
from model.roi_layers import nms
from dataset import readxml

def get_dice_loss(gt_score, pred_score):
    inter = torch.sum(gt_score * pred_score)
    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    return 1. - (2 * inter / union)


def generate_mask(input, H, W):
    bbx1 = input[..., :4]
    bbx1 = np.array([bbx1[..., 0], bbx1[..., 1], bbx1[..., 2], bbx1[..., 1], bbx1[..., 2], bbx1[..., 3],
                     bbx1[..., 0], bbx1[..., 3]])
    bbx1 = bbx1.transpose(1, 0).reshape(-1, 4, 2).astype(np.int)
    scoremap = np.zeros((H, W))
    cv2.fillPoly(scoremap, bbx1, 1)
    return torch.from_numpy(scoremap)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

            # bboxes1 = bboxes1.cpu().detach().numpy()
            # bboxes2 = bboxes2.cpu().detach().numpy()
            # tl = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])
            # br = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
            #
            # iw = (br - tl + 1)[:, :, 0]
            # ih = (br - tl + 1)[:, :, 1]
            # iw[iw < 0] = 0
            # ih[ih < 0] = 0
            # overlaps = iw * ih
            #
            # area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            #     bboxes1[:, 3] - bboxes1[:, 1] + 1)
            #
            # if mode == 'iou':
            #     area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
            #         bboxes2[:, 3] - bboxes2[:, 1] + 1)
            #     ious = overlaps / (area1[:, None] + area2 - overlaps)
            # else:
            #     ious = overlaps / (area1[:, None])
            # ious = torch.from_numpy(ious).cuda()

    return ious


class IoU_loss(object):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, pred, target):
        ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=self.eps)
        loss = -ious.log()
        return loss.mean()


class Smooth_l1_loss(object):
    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, pred, target):
        assert self.beta > 0
        assert pred.size() == target.size() and target.numel() > 0
        pred_widths = pred[:, 2] - pred[:, 0] + 1.0
        pred_heights = pred[:, 3] - pred[:, 1] + 1.0
        pred_ctr_x = pred[:, 0] + 0.5 * pred_widths
        pred_ctr_y = pred[:, 1] + 0.5 * pred_heights

        target_widths = target[:, 2] - target[:, 0] + 1.0
        target_heights = target[:, 3] - target[:, 1] + 1.0
        target_ctr_x = target[:, 0] + 0.5 * target_widths
        target_ctr_y = target[:, 1] + 0.5 * target_heights

        # targets_dx=(target_ctr_x-pred_ctr_x)/pred_widths
        # target_dy=(target_ctr_y-pred_ctr_y)/pred_heights
        # target_dw=torch.log(target_widths/pred_widths)
        # target_dh=torch.log(target_heights/pred_heights)
        #
        # diff=torch.cat(target_ctr_x,target_ctr_y,target_dw,target_dh)

        diff = torch.abs(pred - target)
        diff[:, 0] = diff[:, 0] / target_widths
        diff[:, 2] = diff[:, 2] / target_widths
        diff[:, 1] = diff[:, 1] / target_heights
        diff[:, 3] = diff[:, 3] / target_heights
        loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                           diff - 0.5 * self.beta)
        return loss.mean()


def detection_task(img, fasterRCNN):
    empty_array = np.transpose(np.array([[0], [0], [0], [0], [0]]), (1, 0)).astype(np.float32)
    gt_boxes = torch.FloatTensor([[1, 1, 1, 1, 1]]).cuda()
    num_boxes = torch.FloatTensor([0]).cuda()
    im_info = torch.FloatTensor([[img.shape[2], img.shape[3], 1]]).cuda()  # ---第三维应该是batch_size
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = fasterRCNN(img, im_info, gt_boxes, num_boxes)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    box_deltas = bbox_pred.data
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                     + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * 2)

    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    pred_boxes /= im_info[0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    thresh = 0.05
    j = 1
    inds = torch.nonzero(scores[:, j] > thresh).view(-1)
    # if there is det
    if inds.numel() > 0:
        cls_scores = scores[:, j][inds]
        _, order = torch.sort(cls_scores, 0, True)

        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
        cls_dets = cls_dets[order]
        keep = nms(cls_dets[..., :4], cls_dets[..., -1], cfg.TEST.NMS)
        if keep.shape[0] == 0:
            all_boxes = empty_array
        else:
            cls_dets = cls_dets[keep.view(-1).long()]
            all_boxes = cls_dets.cpu().numpy()
    else:
        all_boxes = empty_array

    image_scores = np.hstack([all_boxes[:, -1]])
    max_per_image = 100
    if len(image_scores) > max_per_image:
        image_thresh = np.sort(image_scores)[-max_per_image]
        keep = np.where(all_boxes[:, -1] >= image_thresh)[0]
        all_boxes = all_boxes[keep, :]

    return torch.from_numpy(all_boxes).cuda()


# -----------pixel-wise augmix for tensors
def Aug_and_Mix_names(content, opt, device):
    transfer_at = set()
    if opt.transfer_at_encoder:
        transfer_at.add('encoder')
    if opt.transfer_at_decoder:
        transfer_at.add('decoder')
    if opt.transfer_at_skip:
        transfer_at.add('skip')

    fname_style = os.listdir(opt.basestyle)
    h, w = content.shape[1], content.shape[2]
    # ws = np.zeros((opt.mixture_width, h, w))
    # for m in range(h):
    #     for n in range(w):
    #         ws[:, m, n] = np.float32(np.random.dirichlet([1] * opt.mixture_width))
    ws = np.random.rand(h * w, opt.mixture_width)
    a = ws / np.sum(ws, 1)[:, np.newaxis]
    ws = a.reshape(h, w, opt.mixture_width).transpose(2, 0, 1)  # test x = np.sum(ws, 0)

    mix = np.zeros((3, h, w))
    for j in range(opt.mixture_width):  # opt.mixture_width
        depth = opt.mixture_depth if opt.mixture_depth > 0 else np.random.randint(1, 3)
        name_list = ''
        input = content
        for _ in range(depth):
            index = np.random.choice(len(fname_style))
            style_name = fname_style[index]
            name_list = name_list + style_name
            style = open_image(os.path.join(opt.basestyle, style_name), opt.image_size).to(device)

            # with Timer('Elapsed time in whole WCT: {}', opt.verbose):
            wct2 = WCT2(transfer_at=transfer_at, option_unpool=opt.option_unpool, device=device,
                        verbose=opt.verbose)
            with torch.no_grad():
                img_out = wct2.transfer(input.unsqueeze(0), style.unsqueeze(0), np.asarray([]), np.asarray([]),
                                        alpha=opt.alpha)
                # save_image(img_out.clamp_(0, 1), name_list, padding=0)
            input = img_out.squeeze()
        mix += np.multiply(np.stack((ws[j], ws[j], ws[j])), img_out.squeeze().to('cpu').numpy())

    # im = Image.fromarray(mix.transpose(1, 2, 0).astype(np.uint8))
    # im.save('mixed.png')
    return torch.from_numpy(mix).to(device).unsqueeze(0).float()


def Aug_and_Mix(content, opt, device):
    transfer_at = set()
    if opt.transfer_at_encoder:
        transfer_at.add('encoder')
    if opt.transfer_at_decoder:
        transfer_at.add('decoder')
    if opt.transfer_at_skip:
        transfer_at.add('skip')

    fname_style = os.listdir(opt.basestyle)
    h, w = content.shape[1], content.shape[2]
    # ws = np.zeros((opt.mixture_width, h, w))
    # for m in range(h):
    #     for n in range(w):
    #         ws[:, m, n] = np.float32(np.random.dirichlet([1] * opt.mixture_width))
    ws = np.random.rand(h * w, opt.mixture_width)
    a = ws / np.sum(ws, 1)[:, np.newaxis]
    ws = a.reshape(h, w, opt.mixture_width).transpose(2, 0, 1)  # test x = np.sum(ws, 0)

    mix = np.zeros((3, h, w))
    for j in range(opt.mixture_width):  # opt.mixture_width
        depth = opt.mixture_depth if opt.mixture_depth > 0 else np.random.randint(1, 3)
        name_list = ''
        input = content
        for _ in range(depth):
            index = np.random.choice(len(fname_style))
            style_name = fname_style[index]
            name_list = name_list + style_name
            style = open_image(os.path.join(opt.basestyle, style_name), opt.image_size).to(device)

            # with Timer('Elapsed time in whole WCT: {}', opt.verbose):
            wct2 = WCT2(transfer_at=transfer_at, option_unpool=opt.option_unpool, device=device,
                        verbose=opt.verbose)
            with torch.no_grad():
                img_out = wct2.transfer(input.unsqueeze(0), style.unsqueeze(0), np.asarray([]), np.asarray([]),
                                        alpha=opt.alpha)
                # save_image(img_out.clamp_(0, 1), name_list, padding=0)
            input = img_out.squeeze()
        mix += np.multiply(np.stack((ws[j], ws[j], ws[j])), img_out.squeeze().to('cpu').numpy())

    # im = Image.fromarray(mix.transpose(1, 2, 0).astype(np.uint8))
    # im.save('mixed.png')
    return torch.from_numpy(mix).to(device).unsqueeze(0).float()

def Pre_train_unpair(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    if opt.loss_det == 'L1':
        loss_det = Smooth_l1_loss()
    if opt.loss_det == 'iou':
        loss_det = IoU_loss()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'KPN_single_image_epoch%d_bs%d_mu%d_sigma%d.pth' % (
                epoch, opt.train_batch_size, opt.mu, opt.sigma)
        if opt.save_mode == 'iter':
            model_name = 'KPN_single_image_iter%d_bs%d_mu%d_sigma%d.pth' % (
                iteration, opt.train_batch_size, opt.mu, opt.sigma)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    # opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.UnpairDataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers,
                              pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------
    device = torch.device('cuda:0')

    # Count start time
    prev_time = time.time()

    if opt.lamda != 0:
        opt.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 24, 48]', 'ANCHOR_RATIOS', '[0.5,1,2]']
        opt.cfg_file = "./det_faster/cfgs/{}.yml".format('vgg16')
        cfg_from_file(opt.cfg_file)
        cfg_from_list(opt.set_cfgs)
        load_name = './det_faster/faster_rcnn_1_40_249.pth'  # './faster_rcnn/vgg16_caffe.pth'
        classes = ('__background__', 'car')

        fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=False)
        fasterRCNN.create_architecture()

        checkpoint = torch.load(load_name)
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        cfg.CUDA = True
        fasterRCNN.cuda()
        fasterRCNN.eval()
        thresh = 0.05

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input, true_target, xml_name) in enumerate(train_loader):
            # To device
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Train Generator
            optimizer_G.zero_grad()
            # 关键的部分，为啥输入都是true_input，都是有噪声的
            fake_target = generator(true_input, true_input)

            # L1 Loss
            Pixellevel_L1_Loss = criterion_L1(fake_target, true_target.squeeze())

            loss_dets = torch.FloatTensor([0]).cuda()
            if opt.lamda != 0:
                with torch.no_grad():
                    for k in range(opt.train_batch_size):
                        fake_input1 = fake_target[k, ...]  # 将batch取出
                        bbox = torch.from_numpy(readxml(xml_name[k])).cuda()

                        fake_input1 = torch.clamp(fake_input1 * 255, 0, 255).unsqueeze(0)
                        assert fake_input1.shape[0] == 1
                        det1 = detection_task(fake_input1,
                                              fasterRCNN)  # fake_target1[k, ...].unsqueeze(0) [1,3,720,1280]

                        if torch.nonzero(det1).view(-1).numel() == 0:
                            det1 = det1.expand(bbox.shape[0], 5).contiguous()
                            loss_det1 = loss_det(det1[:, :4], bbox)
                        else:
                            overlaps = bbox_overlaps_batch(det1[:, :4], bbox.unsqueeze(0)).squeeze(0)  # shape[N,K]
                            max_overlaps, gt_assignment = torch.max(overlaps, 1)
                            inds = torch.nonzero(max_overlaps >= 0.5).view(-1)
                            det1_ = det1[inds]
                            gt_bbx_ = bbox[gt_assignment[inds]]
                            if gt_bbx_.numel() <= 0:
                                loss_det1 = torch.FloatTensor([0]).cuda()
                            else:
                                loss_det1 = loss_det(det1_[:, :4], gt_bbx_)
                        loss_dets += loss_det1

            # Overall Loss and optimize
            loss_detect = loss_dets / opt.train_batch_size
            loss = Pixellevel_L1_Loss + opt.lamda * loss_detect
            loss.backward()
            optimizer_G.step()

            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [Det Loss: %.4f] Time_left: %s" %
                  ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), loss_detect, time_left))

            # print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [JS Loss: %.4f] Time_left: %s" %
            #       ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), loss_JS.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder=sample_folder, sample_name='train_epoch%d' % (epoch + 1),
                                  img_list=img_list, name_list=name_list, pixel_max_cnt=255)

def Pre_train(opt):
    # ----------------------------------------
    #       Network training parameters
    # ----------------------------------------

    # cudnn benchmark
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    utils.check_path(save_folder)
    utils.check_path(sample_folder)

    # Loss functions
    criterion_L1 = torch.nn.L1Loss().cuda()

    if opt.loss_det == 'L1':
        loss_det = Smooth_l1_loss()
    if opt.loss_det == 'iou':
        loss_det = IoU_loss()

    # Initialize Generator
    generator = utils.create_generator(opt)

    # To device
    if opt.multi_gpu:
        generator = nn.DataParallel(generator)
        generator = generator.cuda()
    else:
        generator = generator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2),
                                   weight_decay=opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(opt, epoch, optimizer):
        target_epoch = opt.epochs - opt.lr_decrease_epoch
        remain_epoch = opt.epochs - epoch
        if epoch >= opt.lr_decrease_epoch:
            lr = opt.lr_g * remain_epoch / target_epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    # Save the model if pre_train == True
    def save_model(opt, epoch, iteration, len_dataset, generator):
        """Save the model at "checkpoint_interval" and its multiple"""
        # Define the name of trained model
        if opt.save_mode == 'epoch':
            model_name = 'KPN_single_image_epoch%d_bs%d_mu%d_sigma%d.pth' % (
                epoch, opt.train_batch_size, opt.mu, opt.sigma)
        if opt.save_mode == 'iter':
            model_name = 'KPN_single_image_iter%d_bs%d_mu%d_sigma%d.pth' % (
                iteration, opt.train_batch_size, opt.mu, opt.sigma)
        save_model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.module.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))
        else:
            if opt.save_mode == 'epoch':
                if (epoch % opt.save_by_epoch == 0) and (iteration % len_dataset == 0):
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at epoch %d' % (epoch))
            if opt.save_mode == 'iter':
                if iteration % opt.save_by_iter == 0:
                    torch.save(generator.state_dict(), save_model_path)
                    print('The trained model is successfully saved at iteration %d' % (iteration))

    # ----------------------------------------
    #             Network dataset
    # ----------------------------------------

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.train_batch_size *= gpu_num
    # opt.val_batch_size *= gpu_num
    opt.num_workers *= gpu_num

    # Define the dataset
    trainset = dataset.AugPixelDataset(opt)
    print('The overall number of training images:', len(trainset))

    # Define the dataloader
    train_loader = DataLoader(trainset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers,
                              pin_memory=True)

    # ----------------------------------------
    #                 Training
    # ----------------------------------------
    device = torch.device('cuda:0')

    # Count start time
    prev_time = time.time()

    if opt.lamda != 0:
       opt.set_cfgs = ['ANCHOR_SCALES', '[2, 4, 8, 16, 24, 48]', 'ANCHOR_RATIOS', '[0.5,1,2]']
       opt.cfg_file = "./det_faster/cfgs/{}.yml".format('vgg16')
       cfg_from_file(opt.cfg_file)
       cfg_from_list(opt.set_cfgs)
       load_name = './det_faster/faster_rcnn_1_40_249.pth'  # './faster_rcnn/vgg16_caffe.pth'
       classes = ('__background__', 'car')

       fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=False)
       fasterRCNN.create_architecture()

       checkpoint = torch.load(load_name)
       fasterRCNN.load_state_dict(checkpoint['model'])
       if 'pooling_mode' in checkpoint.keys():
           cfg.POOLING_MODE = checkpoint['pooling_mode']
       cfg.CUDA = True
       fasterRCNN.cuda()
       fasterRCNN.eval()
       thresh = 0.05

    # For loop training
    for epoch in range(opt.epochs):
        for i, (true_input1, true_input2, true_target, xml_name) in enumerate(train_loader):
            # To device
            true_input1 = true_input1.cuda()
            true_input2 = true_input2.cuda()
            true_target = true_target.cuda()
            # Train Generator
            optimizer_G.zero_grad()
            # 关键的部分，为啥输入都是true_input，都是有噪声的
            fake_target1 = generator(true_input1, true_input1)
            fake_target2 = generator(true_input2, true_input2)
            '''
            img_copy = fake_target1.cpu().detach().numpy().transpose(1, 2, 0) * 255
            img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
            cv2.imwrite('fake_target1_' + str(i) + '.jpg', img)
            '''
            if len(fake_target1.shape)==3:
               fake_target1=fake_target1.view(opt.train_batch_size, -1, fake_target1.shape[1], fake_target1.shape[2])
               fake_target2=fake_target2.view(opt.train_batch_size, -1, fake_target2.shape[1], fake_target2.shape[2])
            # L1 Loss
            Pixellevel_L1_Loss = 0.5 * (
            criterion_L1(fake_target1, true_target.squeeze()) + criterion_L1(fake_target2, true_target.squeeze()))

            loss_dets = torch.FloatTensor([0]).cuda()
            if opt.lamda != 0:
                with torch.no_grad():
                    for k in range(opt.train_batch_size):
                        fake_input1 = fake_target1[k, ...]  # 将batch取出
                        fake_input2 = fake_target2[k, ...]  # 将batch取出
                        bbox = torch.from_numpy(readxml(xml_name[k])).cuda()

                        fake_input1 = torch.clamp(fake_input1 * 255, 0, 255).unsqueeze(0)
                        fake_input2 = torch.clamp(fake_input2 * 255, 0, 255).unsqueeze(0)
                        assert fake_input1.shape[0] == 1
                        det1 = detection_task(fake_input1,
                                              fasterRCNN)  # fake_target1[k, ...].unsqueeze(0) [1,3,720,1280]
                        det2 = detection_task(fake_input2,
                                              fasterRCNN)  # fake_target1[k, ...].unsqueeze(0) [1,3,720,1280]

                        if torch.nonzero(det1).view(-1).numel() == 0:
                            det1 = det1.expand(bbox.shape[0], 5).contiguous()
                            loss_det1 = loss_det(det1[:, :4], bbox)
                        else:
                            overlaps = bbox_overlaps_batch(det1[:, :4], bbox.unsqueeze(0)).squeeze(0)  # shape[N,K]
                            max_overlaps, gt_assignment = torch.max(overlaps, 1)
                            inds = torch.nonzero(max_overlaps >= 0.5).view(-1)
                            det1_ = det1[inds]
                            gt_bbx_ = bbox[gt_assignment[inds]]
                            if gt_bbx_.numel()<=0:
                               loss_det1=torch.FloatTensor([0]).cuda()
                            else:
                               loss_det1 = loss_det(det1_[:, :4], gt_bbx_)

                        if torch.nonzero(det2).view(-1).numel() == 0:
                            det2 = det2.expand(bbox.shape[0], 5).contiguous()
                            loss_det2 = loss_det(det2[:, :4], bbox)
                        else:
                            overlaps = bbox_overlaps_batch(det2[:, :4], bbox.unsqueeze(0)).squeeze(0)  # shape[N,K]
                            max_overlaps, gt_assignment = torch.max(overlaps, 1)
                            inds = torch.nonzero(max_overlaps >= 0.5).view(-1)
                            det2_ = det2[inds]
                            gt_bbx_ = bbox[gt_assignment[inds]]
                            if gt_bbx_.numel()<=0:
                               loss_det2=torch.FloatTensor([0]).cuda()
                            else:
                               loss_det2 = loss_det(det2_[:, :4], gt_bbx_)
                        loss_dets += (loss_det1 + loss_det2) * 0.5

            # Overall Loss and optimize
            loss_detect = loss_dets / opt.train_batch_size
            loss = Pixellevel_L1_Loss + opt.lamda * loss_detect
            loss.backward()
            optimizer_G.step()


            # Determine approximate time left
            iters_done = epoch * len(train_loader) + i
            iters_left = opt.epochs * len(train_loader) - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [Det Loss: %.4f] Time_left: %s" %
                  ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), loss_detect, time_left))

            # print("\r[Epoch %d/%d] [Batch %d/%d] [Pixellevel L1 Loss: %.4f] [JS Loss: %.4f] Time_left: %s" %
            #       ((epoch + 1), opt.epochs, i, len(train_loader), Pixellevel_L1_Loss.item(), loss_JS.item(), time_left))

            # Save model at certain epochs or iterations
            save_model(opt, (epoch + 1), (iters_done + 1), len(train_loader), generator)

            # Learning rate decrease at certain epochs
            adjust_learning_rate(opt, (epoch + 1), optimizer_G)

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input1, fake_target1, true_input2, fake_target2, true_target]
            name_list = ['in1', 'pred1', 'in2', 'pred2', 'gt']
            utils.save_sample_png(sample_folder=sample_folder, sample_name='train_epoch%d' % (epoch + 1),
                                  img_list=img_list, name_list=name_list, pixel_max_cnt=255)

        '''#-------------将WCT2和KPN结合起来训练
        for i, (true_input) in enumerate(train_loader):
            # To device
            true_input = true_input.cuda()
            true_target = true_input

            with torch.no_grad():
                cropper = RandomCropTensor(true_target.shape[2:4], (opt.crop_size, opt.crop_size))
                true_input1s = torch.zeros_like(true_input)
                true_input2s = torch.zeros_like(true_input)
                for j in range(opt.train_batch_size):
                    true_input1 = Aug_and_Mix(true_input[j, ...], opt, device)
                    true_input1s[j, ...] = true_input1
                    true_input2 = Aug_and_Mix(true_input[j, ...], opt, device)
                    true_input2s[j, ...] = true_input2
                true_input1s = cropper(true_input1s)
                true_input2s = cropper(true_input2s)
                true_target = cropper(true_target)
        '''
        '''
        ### Validation
        val_PSNR = 0
        num_of_val_image = 0

        for j, (true_input, true_target) in enumerate(val_loader):
            
            # To device
            # A is for input image, B is for target image
            true_input = true_input.cuda()
            true_target = true_target.cuda()

            # Forward propagation
            with torch.no_grad():
                fake_target = generator(true_input)

            # Accumulate num of image and val_PSNR
            num_of_val_image += true_input.shape[0]
            val_PSNR += utils.psnr(fake_target, true_target, 1) * true_input.shape[0]
        val_PSNR = val_PSNR / num_of_val_image

        ### Sample data every epoch
        if (epoch + 1) % 1 == 0:
            img_list = [true_input, fake_target, true_target]
            name_list = ['in', 'pred', 'gt']
            utils.save_sample_png(sample_folder = sample_folder, sample_name = 'val_epoch%d' % (epoch + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)

        # Record average PSNR
        print('PSNR at epoch %d: %.4f' % ((epoch + 1), val_PSNR))
        '''

        '''div loss
        p_clean, p_aug1, p_aug2 = F.softmax(
            true_target, dim=1), F.softmax(
            true_input1, dim=1), F.softmax(
            true_input2, dim=1)
        mix_ = (true_target + fake_target1 + fake_target2) / 3.
        p_mixture = torch.clamp(mix_, 1e-7, 1)
        # loss_JS = 12 * (F.kl_div(p_mixture, true_target, reduction='batchmean') +
        #                 F.kl_div(p_mixture, fake_target1, reduction='batchmean') +
        #                 F.kl_div(p_mixture, fake_target2, reduction='batchmean')) / 3.
        for i in range(opt.train_batch_size):
            x = p_mixture[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            y = true_target[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0)
            a = ssim(x, y)
            a = ssim(p_mixture[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0),
                     true_target[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0))
            b = a
            loss_JS += 1 - (
                               ssim(p_mixture[i, ...].squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1),
                                    true_target[i, ...].squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1))
                               + ssim(p_mixture[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0),
                                      fake_target1[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0))
                               + ssim(p_mixture[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2, 0),
                                      fake_target2[i, ...].squeeze().cpu().detach().numpy().transpose(1, 2,
                                                                                                      0))) / 3.
        a = ssim(p_mixture.squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1),
                 true_target.squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1))
        loss_JS = 1 - (
                          ssim(p_mixture.squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1),
                               true_target.squeeze().cpu().detach().numpy().transpose(0, 2, 3, 1))
                          + ssim(p_mixture.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
                                 fake_target1.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
                          + ssim(p_mixture.squeeze().cpu().detach().numpy().transpose(1, 2, 0),
                                 fake_target2.squeeze().cpu().detach().numpy().transpose(1, 2, 0))) / 3.
        '''

        '''
        det_target = detection_task(true_target[k, ...].unsqueeze(0), fasterRCNN)
        score1 = generate_mask(det1, H, W)
        score2 = generate_mask(det2, H, W)
        score_target = generate_mask(det_target, H, W)
        cv2.imwrite('mask1.jpg', score1.numpy() * 255)
        cv2.imwrite('mask2.jpg', score2.numpy() * 255)
        cv2.imwrite('target_mask.jpg', score_target.numpy() * 255)
        loss_detection = (get_dice_loss(gt_mask[k, ...].squeeze().double(), score1) + get_dice_loss(
            gt_mask[k, ...].squeeze().double(), score2)) / 2.0
        '''
