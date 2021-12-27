# -*- coding:utf-8 -*-
import os
import random
import numpy as np
import cv2
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transfer import WCT2, get_all_transfer
from utils_WCT.io import Timer, open_image
from PIL import Image
from torchvision.utils import make_grid
from torchvision.transforms import RandomHorizontalFlip
import utils
import xml.etree.ElementTree as ET

'''读取xml_file'''


# Make pixel indexes 0-based
# x1 = int(float(bbox.find('xmin').text) * image_size / H - 7)
# y1 = int(float(bbox.find('ymin').text) * image_size / H)
# x2 = int(float(bbox.find('xmax').text) * image_size / H - 7)
# y2 = int(float(bbox.find('ymax').text) * image_size / H)
# cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0))
# cv2.imwrite('retang.jpg', img)

def readxml(xml_name):
    tree = ET.parse(xml_name)
    objs = tree.findall('object')
    bbx = []
    for i, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        # poly = np.array([x1, y1, x2, y1, x2, y2, x1, y2]).reshape((4, 2))
        poly = np.array([x1, y1, x2, y2])
        bbx.append(poly)
    bbx = np.array(bbx).astype(np.float32)
    return bbx


def aug(opt, img):
    H_in = img[0].shape[0]
    W_in = img[0].shape[1]
    sc = np.random.uniform(opt.scale_min, opt.scale_max)
    H_out = int(math.floor(H_in * sc))
    W_out = int(math.floor(W_in * sc))
    # scaled size should be greater than opts.crop_size
    if H_out < W_out:
        if H_out < opt.crop_size:
            H_out = opt.crop_size
            W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
    else:  # W_out < H_out
        if W_out < opt.crop_size:
            W_out = opt.crop_size
            H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
    img = cv2.resize(img, (W_out, H_out))
    return img


def rotate(img):
    rotate = random.randint(0, 3)
    if rotate != 0:
        img = np.rot90(img, rotate)
    if np.random.random() >= 0.5:
        img = cv2.flip(img, flipCode=0)
    return img


# ---------image level fusion
def augment_and_mix(img, aug_imgs, opt):
    ws = np.float32(np.random.dirichlet([1] * opt.mixture_width))
    m = np.float32(np.random.beta(1, 1))
    mix = np.zeros_like(img.astype(np.float32))

    for i in range(opt.mixture_width):
        # 深度
        # depth = self.opt.mixture_depth if self.opt.mixture_depth > 0 else np.random.randint(
        #     1, 4)
        # for _ in range(depth):
        #     image_aug = np.random.choice(aug_imgs)
        mix += ws[i] * aug_imgs[i].astype(np.float32)

    # mixed = (1 - m) * img.astype(np.float32) + m * mix
    mixed = mix
    return mixed


styles = ['in00', 'in01', 'in14', 'in18', 'in20']
styles1 = ['in00', 'in01', 'in02', 'in03', 'in04']
styles2 = ['in03', 'in04', 'in06', 'in08', 'in18']
# ---------pixel level fusion
def augment_and_mix_pixel(img, aug_imgs, img_name, aug_names, opt):
    if 'aug2' in opt.baseaug:
       style_list = styles2
    elif 'aug1' in opt.baseaug:
       style_list=styles1
    else:
       style_list=styles
    h, w = img.shape[0], img.shape[1]
    ws = np.random.rand(h * w, opt.mixture_width)
    a = ws / np.sum(ws, 1)[:, np.newaxis]
    ws = a.reshape(h, w, opt.mixture_width)  # test
    x = np.sum(ws, 2)
    mix = np.zeros((h, w, 3))
    for j in range(opt.mixture_width):  # opt.mixture_width
        depth = opt.mixture_depth if opt.mixture_depth > 0 else np.random.randint(0, 2)
        name_1 = np.random.choice(style_list)
        style_ = name_1
        for _ in range(depth):
            while style_ == name_1:
                style_ = np.random.choice(style_list)
            name_1 += '_' + style_
        _, ext = os.path.splitext(img_name)
        index = aug_names.index(img_name.replace(opt.baseroot, opt.baseaug).replace(ext, '') + '_' + name_1 + '.png')
        x = np.stack((ws[..., j], ws[..., j], ws[..., j]))
        mix += np.multiply(x.transpose(1, 2, 0), aug_imgs[index])  # 这里reshape和transpose没有太大差别
    
    #im = cv2.cvtColor(mix.astype(np.float32), cv2.COLOR_RGB2GRAY)
    #im = im.astype(np.uint8)
    #blur = cv2.GaussianBlur(im, (5, 5), 0)
    #wide = cv2.Canny(blur, 50, 200)
    #mix = np.concatenate((mix, wide[:,:, np.newaxis]), axis = 2)
  
    img = cv2.imread('./background.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    m = np.float32(np.random.beta(1, 1))
    mixed = (1 - m) * img.astype(np.float32) + m * mix
    mixed = mix
    # cv2.imwrite('ori.jpg', img)
    # cv2.imwrite('mixed.jpg', mixed)
    return mixed


# ---------patch level fusion
def augment_and_mix_patch(img, aug_imgs, img_name, aug_names, opt):
    h, w = img.shape[0], img.shape[1]
    kernel_size = 5
    weights = np.zeros((3, h, w))
    for i in range(h // kernel_size):
        for j in range(h // kernel_size):
            ws = np.float32(np.random.dirichlet([1] * opt.mixture_width))
            ws = ws[:, np.newaxis, np.newaxis]
            ws = np.concatenate((ws, ws, ws, ws, ws), 1)
            ws = np.concatenate((ws, ws, ws, ws, ws), 2)  # shape 3*5*5
            weights[:, (i * 5): (i * 5 + 5), (j * 5):(j * 5 + 5)] = ws

    # cv2.imwrite('ws.jpg', weights.transpose(1, 2, 0) * 255)

    mix = np.zeros((h, w, 3))
    for j in range(opt.mixture_width):  # opt.mixture_width
        depth = opt.mixture_depth if opt.mixture_depth > 0 else np.random.randint(0, 2)
        name_1 = np.random.choice(style_list)
        style_ = name_1
        for _ in range(depth):
            while style_ == name_1:
                style_ = np.random.choice(style_list)
            name_1 += '_' + style_
        _, ext = os.path.splitext(img_name)
        index = aug_names.index(img_name.replace(opt.baseroot, opt.baseaug).replace(ext, '') + '_' + name_1 + '.png')
        x = np.stack((weights[j, ...], weights[j, ...], weights[j, ...]))
        mix += np.multiply(x.transpose(1, 2, 0), aug_imgs[index])

    # m = np.float32(np.random.beta(1, 1))
    # mixed = (1 - m) * img.astype(np.float32) + m * mix
    mixed = mix
    # cv2.imwrite('ori.jpg', img)
    # cv2.imwrite('mixed.jpg', mixed)
    return mixed


class RandomCropTensor(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 4:
            return img[:, :, self.h1: self.h2, self.w1: self.w2]
        else:
            return img[:, self.h1: self.h2, self.w1: self.w2]


class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if img is None:
            return [self.h1, self.w1]
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


# ------------------将WCT2和KPN结合起来
class AugMixDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        self.stylelist = utils.get_files(opt.basestyle)

    def __getitem__(self, index):
        ## read an image
        img_name = self.imglist[index]
        content = open_image(img_name, self.opt.image_size)
        return content

    def __len__(self):
        return len(self.imglist)

# ------------unpair night to day for KPN training
class UnpairDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)  # 训练用的白天的数据
        self.auglist = utils.get_files(opt.baseaug)  # 训练用的晚上的数据
        self.xmllist = utils.get_files(opt.basexml)  # 晚上的Ground truth

    def __getitem__(self, index):
        ## read an image
        img_name = self.imglist[index]  # img_name = './datasets/Cars/2317.jpg'
        img = Image.open(img_name)
        randomFlip = RandomHorizontalFlip(p=0.5)
        img = randomFlip(img)
        img = np.asarray(img)      
  
        night_name = self.auglist[index]
        night_img = cv2.imread(night_name)
        night_img = cv2.cvtColor(night_img, cv2.COLOR_BGR2RGB) #晚上的图像，当作输入

        # ----------读取xml文件，并且transform对应的坐标
        xml_name = night_name.replace(self.opt.baseaug, self.opt.basexml).replace('jpg', 'xml')

        if self.opt.geometry_aug:
            img = aug(self.opt, img)
            night_img = aug(self.opt, night_img)

        if self.opt.angle_aug:
            img = rotate(img)
            night_img = rotate(night_img)

        img = img.astype(np.float32)  # RGB image in range [0, 255]
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        night_img = night_img.astype(np.float32)  # RGB image in range [0, 255]
        night_img = night_img / 255.0
        night_img = torch.from_numpy(night_img.transpose(2, 0, 1)).contiguous()

        return night_img, img, xml_name  # mixed2, img,

    def __len__(self):
        return len(self.imglist)

# ------------------pixel-level 图像相加， 先把图像生成，再结合
class AugPixelDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)  # 训练用的白天的数据
        self.auglist = utils.get_files(opt.baseaug)
        self.xmllist = utils.get_files(opt.basexml)

    def __getitem__(self, index):
        ## read an image
        img_name = self.imglist[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ----------读取xml文件，并且transform对应的坐标
        xml_name = img_name.replace(self.opt.baseroot, self.opt.basexml).replace('jpg', 'xml')

        aug_names = [x for x in self.auglist if img_name.replace(self.opt.baseroot + '/', '')[:4] in x]
        aug_imgs = []
        for item in aug_names:
            aug_img = cv2.imread(item)
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            aug_imgs.append(aug_img)

        if self.opt.geometry_aug:
            img = aug(self.opt, img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = aug(self.opt, aug_imgs[i])

        if self.opt.angle_aug:
            img = rotate(img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = rotate(aug_imgs[i])

        mixed1 = augment_and_mix_pixel(img, aug_imgs, img_name, aug_names, self.opt).astype(
            np.float32)  # cv2.imwrite('mixed.png', cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR))
        '''
        style_list =styles
        name_1 = np.random.choice(style_list)
        _, ext = os.path.splitext(img_name)
        index = aug_names.index(
            img_name.replace(self.opt.baseroot, self.opt.baseaug).replace(ext, '') + '_' + name_1 + '.png')
        mixed1 = aug_imgs[index].astype(np.float32)
        im = cv2.cvtColor(mixed1, cv2.COLOR_RGB2GRAY)
        im = im.astype(np.uint8)
        blur = cv2.GaussianBlur(im, (5, 5), 0)
        wide = cv2.Canny(blur, 50, 200)
        mixed1 = np.concatenate((mixed1, wide[:, :, np.newaxis]), axis=2)
        '''
        mixed1 = mixed1 / 255.0  # cv2.imwrite('mixed1.jpg', mixed1 * 255)
        mixed1 = torch.from_numpy(mixed1.transpose(2, 0, 1)).contiguous()
        
        mixed2 = augment_and_mix_pixel(img, aug_imgs, img_name, aug_names, self.opt).astype(np.float32)
        '''
        name_2 = np.random.choice(style_list)
        _, ext = os.path.splitext(img_name)
        index = aug_names.index(
            img_name.replace(self.opt.baseroot, self.opt.baseaug).replace(ext, '') + '_' + name_2 + '.png')
        mixed2 = aug_imgs[index].astype(np.float32)
        im = cv2.cvtColor(mixed2, cv2.COLOR_RGB2GRAY)
        im = im.astype(np.uint8)
        blur = cv2.GaussianBlur(im, (5, 5), 0)
        wide = cv2.Canny(blur, 50, 200)
        mixed2 = np.concatenate((mixed2, wide[:, :, np.newaxis]), axis=2)
        '''
        mixed2 = mixed2 / 255.0
        mixed2 = torch.from_numpy(mixed2.transpose(2, 0, 1)).contiguous()
        
        img = img.astype(np.float32)  # RGB image in range [0, 255]
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        return mixed1, mixed2, img, xml_name  # mixed2, img,

    def __len__(self):
        return len(self.imglist)


# ------------------pixel-level 图像相加， 先把图像生成，再结合
class AugPixelDataset_crop(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)  # 训练用的白天的数据
        self.auglist = utils.get_files(opt.baseaug)
        self.xmllist = utils.get_files(opt.basexml)

    def __getitem__(self, index):
        ## read an image
        # img_name = self.imglist[index]
        img_name = './datasets/Cars/2317.jpg'
        image = Image.open(img_name)
        # W, H = image.size
        # image = transforms.Resize(self.opt.image_size)(image)  # 这种Resize方法不会跟aug的图片有shift
        # w, h = image.size
        # img = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))(image)
        img = np.asarray(image)

        bbx = np.array([0])
        # ----------读取xml文件，并且transform对应的坐标
        # xml_name = img_name.replace(self.opt.baseroot, self.opt.basexml).replace('jpg', 'xml')
        # bbx = readxml(xml_name)  # get all bbox [x1,y1,x2,y2]
        '''deal with mask
        # score_map = np.zeros((H, W), np.float32)
        # cv2.fillPoly(score_map, bbx, 1)
        # score_map = Image.fromarray(score_map)
        # mask = transforms.Resize(self.opt.image_size)(score_map)  # 这种Resize方法不会跟aug的图片有shift
        # w, h = mask.size
        # mask = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))(mask)
        # mask = np.asarray(mask)
        '''

        aug_names = [x for x in self.auglist if img_name.replace(self.opt.baseroot + '/', '')[:4] in x]
        aug_imgs = []
        for item in aug_names:
            aug_img = cv2.imread(item)
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            aug_imgs.append(aug_img)

        if self.opt.geometry_aug:
            img = aug(self.opt, img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = aug(self.opt, aug_imgs[i])

        # crop
        '''
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        # cv2.imwrite('ori.png', img)
        for i in range(len(aug_imgs)):
            aug_imgs[i] = cropper(aug_imgs[i])
        #     # cv2.imwrite(str(i)+'aug.png',aug_imgs[i] )
        # mask = cropper(mask)
        # cv2.imwrite('score_map.jpg', mask * 255)
        '''

        if self.opt.angle_aug:
            img = rotate(img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = rotate(aug_imgs[i])

        mixed1 = augment_and_mix_pixel(img, aug_imgs, img_name, aug_names, self.opt).astype(np.float32)
        # cv2.imwrite('mixed.png', cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR))
        mixed1 = mixed1 / 255.0
        # cv2.imwrite('mixed1.jpg', mixed1 * 255)
        mixed1 = torch.from_numpy(mixed1.transpose(2, 0, 1)).contiguous()

        mixed2 = augment_and_mix_pixel(img, aug_imgs, img_name, aug_names, self.opt).astype(np.float32)
        # cv2.imwrite('mixed.png', cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR))
        mixed2 = mixed2 / 255.0
        # cv2.imwrite('mixed2.jpg', mixed2 * 255)
        mixed2 = torch.from_numpy(mixed2.transpose(2, 0, 1)).contiguous()

        img = img.astype(np.float32)  # RGB image in range [0, 255]
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        bbx = torch.from_numpy(bbx)
        return mixed1, bbx  # mixed2, img,

    def __len__(self):
        return len(self.imglist)


# ------------------image-level 图像相加
class AugDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        self.auglist = utils.get_files('./datasets/Augs')

    def __getitem__(self, index):
        ## read an image
        img_name = self.imglist[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug_names = [x for x in self.auglist if img_name.replace('./datasets/Cars/', '')[:4] in x]
        aug_imgs = []
        for item in aug_names:
            aug_img = cv2.imread(item)
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
            aug_imgs.append(aug_img)

        # plt.imshow(img)
        # plt.show()
        img = cv2.resize(img, (aug_imgs[0].shape[1], aug_imgs[0].shape[0]), interpolation=cv2.INTER_LINEAR)
        # plt.imshow(img)
        # plt.show()
        # cv2.imwrite('ori_resize.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if self.opt.geometry_aug:
            img = aug(self.opt, img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = aug(self.opt, aug_imgs[i])

        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        # cv2.imwrite('ori.png',img)
        for i in range(len(aug_imgs)):
            aug_imgs[i] = cropper(aug_imgs[i])
            # cv2.imwrite(str(i)+'aug.png',aug_imgs[i] )

        if self.opt.angle_aug:
            img = rotate(img)
            for i in range(len(aug_imgs)):
                aug_imgs[i] = rotate(aug_imgs[i])

        mixed1 = augment_and_mix(img, aug_imgs, self.opt)
        # cv2.imwrite('mixed.png', cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR))
        mixed1 = mixed1 / 255.0
        # plt.imshow(mixed1)
        # plt.show()
        mixed1 = torch.from_numpy(mixed1.transpose(2, 0, 1)).contiguous()

        mixed2 = augment_and_mix(img, aug_imgs, self.opt)
        # cv2.imwrite('mixed.png', cv2.cvtColor(mixed, cv2.COLOR_RGB2BGR))
        mixed2 = mixed2 / 255.0
        # plt.imshow(mixed2)
        # plt.show()
        mixed2 = torch.from_numpy(mixed2.transpose(2, 0, 1)).contiguous()

        img = img.astype(np.float32)  # RGB image in range [0, 255]
        img = img / 255.0
        # plt.imshow(img)
        # plt.show()
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()

        # p_img = F.softmax(img, dim=0)
        # img = p_img.numpy().transpose(1, 2, 0)
        # # plt.imshow(img)
        # # plt.show()
        #
        # p_img = F.softmax(mixed1, dim=0)
        # img = p_img.numpy().transpose(1, 2, 0)
        # plt.imshow(img)
        # plt.show()
        return mixed1, mixed2, img

    def __len__(self):
        return len(self.imglist)


class DenoisingDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)

    def __getitem__(self, index):
        ## read an image
        img = cv2.imread(self.imglist[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ## data augmentation
        # random scale
        if self.opt.geometry_aug:
            H_in = img[0].shape[0]
            W_in = img[0].shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else:  # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        # random crop
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode=0)

        # add noise
        img = img.astype(np.float32)  # RGB image in range [0, 255]
        # 生成高斯噪声
        noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
        noisy_img = img + noise

        # normalization
        img = img / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
        noisy_img = noisy_img / 255.0
        noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).contiguous()

        return noisy_img, img

    def __len__(self):
        return len(self.imglist)


class DenoisingValDataset(Dataset):
    def __init__(self, opt):  # root: list ; transform: torch transform
        self.opt = opt
        self.imglist = utils.get_files(opt.baseroot)
        # self.xmllist = utils.get_files(opt.basexml)

    def __getitem__(self, index):
        img_name = self.imglist[index].replace(self.opt.baseroot + '/', '').replace('.jpg', '')
        ## read an image
        img_path = self.imglist[index]
        image = Image.open(img_path)
        image = transforms.Resize(512)(image)  # 这种Resize方法不会跟aug的图片有shift
        w, h = image.size
        img = transforms.CenterCrop((h // 16 * 16, w // 16 * 16))(image)
        img = np.asarray(img)

        # aug_names = [x for x in self.imglist if img_name[:9] in x]
        # aug_imgs = []
        # for item in aug_names:
        #     aug_img = cv2.imread(item)
        #     aug_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
        #     aug_imgs.append(aug_img)
        # mixed1 = augment_and_mix(img, aug_imgs, self.opt)
        # img = mixed1

        ## data augmentation
        # random scale
        if self.opt.geometry_aug:
            H_in = img[0].shape[0]
            W_in = img[0].shape[1]
            sc = np.random.uniform(self.opt.scale_min, self.opt.scale_max)
            H_out = int(math.floor(H_in * sc))
            W_out = int(math.floor(W_in * sc))
            # scaled size should be greater than opts.crop_size
            if H_out < W_out:
                if H_out < self.opt.crop_size:
                    H_out = self.opt.crop_size
                    W_out = int(math.floor(W_in * float(H_out) / float(H_in)))
            else:  # W_out < H_out
                if W_out < self.opt.crop_size:
                    W_out = self.opt.crop_size
                    H_out = int(math.floor(H_in * float(W_out) / float(W_in)))
            img = cv2.resize(img, (W_out, H_out))
        # random crop
        # if self.opt.crop:
        #     cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        #     img = cropper(img)
        img_t = img
        # 完整的图片分成6份
        h, w, c = img.shape
        # img1 = img[:512, :512, :]
        # img2 = img[:512, 512:1024, :]
        # img3 = img[:512, (w - 512):w, :]
        # img4 = img[(h - 512):, :512, :]
        # img5 = img[(h - 512):, 512:1024, :]
        # img6 = img[(h - 512):, (w - 512):w, :]
        # img_list = [img1, img2, img3, img4, img5, img6]

        # for transfered 512, 896
        img1 = img[:self.opt.crop_size, :self.opt.crop_size, :]
        img2 = img[:self.opt.crop_size, self.opt.crop_size:self.opt.crop_size * 2, :]
        img3 = img[:self.opt.crop_size, self.opt.crop_size * 2:self.opt.crop_size * 3, :]
        img4 = img[:self.opt.crop_size, w - self.opt.crop_size:w, :]
        img5 = img[self.opt.crop_size:, :self.opt.crop_size, :]
        img6 = img[self.opt.crop_size:, self.opt.crop_size:self.opt.crop_size * 2, :]
        img7 = img[self.opt.crop_size:, self.opt.crop_size * 2:self.opt.crop_size * 3, :]
        img8 = img[self.opt.crop_size:, w - self.opt.crop_size:w, :]
        img_list = [img1, img2, img3, img4, img5, img6, img7, img8]

        # random rotate and horizontal flip
        # according to paper, these two data augmentation methods are recommended
        if self.opt.angle_aug:
            rotate = random.randint(0, 3)
            if rotate != 0:
                img = np.rot90(img, rotate)
            if np.random.random() >= 0.5:
                img = cv2.flip(img, flipCode=0)

        # add noise
        if self.opt.add_noise:
            img = img.astype(np.float32)  # RGB image in range [0, 255]
            noise = np.random.normal(self.opt.mu, self.opt.sigma, img.shape).astype(np.float32)
            noisy_img = img + noise
            # normalization
            img = img / 255.0
            img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
            noisy_img = noisy_img / 255.0
            noisy_img = torch.from_numpy(noisy_img.transpose(2, 0, 1)).contiguous()
        else:
            noises = []
            for i in range(len(img_list)):
                img = img_list[i].astype(np.float32)  # RGB image in range [0, 255]
                # normalization
                img = img / 255.0
                img = torch.from_numpy(img.transpose(2, 0, 1)).contiguous()
                noisy_img = img
                noises.append(noisy_img)

        img_t = img_t.astype(np.float32)
        img_t = img_t / 255.0
        img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).contiguous()

        return noises, img_t, img_name

    def __len__(self):
        return len(self.imglist)

        ''' ---crop the bounding boxes and get score map
        # h1, w1 = cropper(None)
        # bbxes = np.array(
        #     [item for item in bbx if item[0] >= w1 and item[2] < (w1 + self.opt.crop_size) and item[1] >= h1 and item[
        #         3] < (h1 + self.opt.crop_size)])
        # bbxes[:, 0] = bbxes[:, 0] - w1
        # bbxes[:, 2] = bbxes[:, 2] - w1
        # bbxes[:, 1] = bbxes[:, 1] - h1
        # bbxes[:, 3] = bbxes[:, 3] - h1
        # bbxes = np.array(
        #     [bbxes[:, 0], bbxes[:, 1], bbxes[:, 2], bbxes[:, 1], bbxes[:, 2], bbxes[:, 3], bbxes[:, 0], bbxes[:, 3]])
        # bbx = bbxes.reshape((4, 2, -1)).transpose(2, 0, 1)
        # score_map = np.zeros((img.shape[0], img.shape[1]), np.float32)
        # cv2.fillPoly(score_map, bbx, 1)
        '''
