"""
Copyright (c) 2019 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import os
import tqdm
import argparse

import torch
from torchvision.utils import save_image

from model_kpn import WaveEncoder, WaveDecoder

from utils_WCT.core import feature_wct
from utils_WCT.io import Timer, open_image, load_segment, compute_label_info
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT2:
    def __init__(self, model_path='./model_checkpoints', transfer_at=['encoder', 'skip', 'decoder'],
                 option_unpool='cat5', device='cuda:0', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
            transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, content_segment, style_segment, alpha=1):
        label_set, label_indicator = compute_label_info(content_segment, style_segment)
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct2_enc_level = [1, 2, 3, 4]
        wct2_dec_level = [1, 2, 3, 4]
        wct2_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct2_enc_level:
                content_feat = feature_wct(content_feat, style_feats['encoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                # self.print_('transfer at encoder {}'.format(level))
        if 'skip' in self.transfer_at:
            for skip_level in wct2_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = feature_wct(content_skips[skip_level][component],
                                                                       style_skips[skip_level][component],
                                                                       content_segment, style_segment,
                                                                       label_set, label_indicator,
                                                                       alpha=alpha, device=self.device)
                    # self.print_('transfer at skip {}'.format(skip_level))

        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct2_dec_level:
                content_feat = feature_wct(content_feat, style_feats['decoder'][level],
                                           content_segment, style_segment,
                                           label_set, label_indicator,
                                           alpha=alpha, device=self.device)
                # self.print_('transfer at decoder {}'.format(level))
            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat


def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret


def run_bulk(config):
    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    transfer_at = set()
    if config.transfer_at_encoder:
        transfer_at.add('encoder')
    if config.transfer_at_decoder:
        transfer_at.add('decoder')
    if config.transfer_at_skip:
        transfer_at.add('skip')

    # The filenames of the content and style pair should match
    print(set(os.listdir(config.content)))
    print(set(os.listdir(config.style)))
    fnames_content = set(os.listdir(config.content))
    fnames_style = set(os.listdir(config.style))

    if config.content_segment and config.style_segment:
        fnames_content &= set(os.listdir(config.content_segment))
        fnames_style &= set(os.listdir(config.style_segment))

    for fname in tqdm.tqdm(fnames_content):
        if not is_image_file(fname):
            print('invalid file (is not image), ', fname)
            continue
        _content = os.path.join(config.content, fname)
        _content_segment = os.path.join(config.content_segment, fname) if config.content_segment else None
        content = open_image(_content, config.image_size).to(device)
        content_segment = load_segment(_content_segment, config.image_size)

        for style_name in tqdm.tqdm(fnames_style):
            _style = os.path.join(config.style, style_name)
            _style_segment = os.path.join(config.style_segment, style_name) if config.style_segment else None
            _, ext = os.path.splitext(fname)
            _output = os.path.join(config.output, fname.replace(ext, '') + '_' + style_name)

            style = open_image(_style, config.image_size).to(device)
            style_segment = load_segment(_style_segment, config.image_size)
            _, ext = os.path.splitext(style_name)

            postfix = '_'.join(sorted(list(transfer_at)))
            # fname_output = _output.replace(ext, '_{}_{}{}'.format(config.option_unpool, postfix, ext))
            print('------ transfer:', _output)
            wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device,
                        verbose=config.verbose)
            with torch.no_grad():
                img = wct2.transfer(content.unsqueeze(0), style.unsqueeze(0), content_segment, style_segment,
                                    alpha=config.alpha)
            save_image(img.clamp_(0, 1), _output, padding=0)

            content_2 = img
            for style_2 in tqdm.tqdm(fnames_style):
                if style_2 == style_name:
                    continue
                _style = os.path.join(config.style, style_2)
                _style_segment = os.path.join(config.style_segment, style_2) if config.style_segment else None
                _, ext = os.path.splitext(fname)
                _, ext_style = os.path.splitext(style_2)
                _output = os.path.join(config.output, fname.replace(ext, '') + '_' + style_name.replace(ext_style, '') + '_' + style_2)

                style = open_image(_style, config.image_size).to(device)
                style_segment = load_segment(_style_segment, config.image_size)
                _, ext = os.path.splitext(style_2)


                postfix = '_'.join(sorted(list(transfer_at)))
                fname_output = _output.replace(ext, '_{}_{}{}'.format(config.option_unpool, postfix, ext))
                print('------ transfer:', _output)
                wct2 = WCT2(transfer_at=transfer_at, option_unpool=config.option_unpool, device=device,
                            verbose=config.verbose)
                with torch.no_grad():
                    img = wct2.transfer(content_2, style.unsqueeze(0), content_segment, style_segment,
                                        alpha=config.alpha)
                save_image(img.clamp_(0, 1), _output, padding=0)


import cv2
import numpy as np

if __name__ == '__main__':

    # file_path = '/home/fulan/352/train_epoch200_gt.png'
    # base_path = '/home/fulan/352/train_epoch200_pred.png'
    #
    # img1 = cv2.cvtColor(cv2.imread(file_path),cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(cv2.imread(base_path),cv2.COLOR_BGR2GRAY)
    # print(ssim(torch.from_numpy(img1.copy()), torch.from_numpy(img2.copy())))
    #
    # file_path = '/home/fulan/WCT2/examples/style/in18.png'
    # base_path = '/home/fulan/WCT2/examples/style/in00.png'
    # i = 23
    # name = 'in' + str("%02d" % i)
    # img = cv2.imread(file_path)
    # img_base = cv2.imread(base_path)
    # img_new = np.hstack((img[:, :1024, :], img_base[:, 1024:, :]))
    # print(img_new.shape)
    # print(img_base.shape)
    #
    # cv2.imwrite('/home/fulan/WCT2/examples/style/' + name + '.png', img_new)

    # file_path = '/home/fulan/WCT2/examples/content/in00.png'
    # # rootpath = os.listdir(file_path)
    # i = 1
    # for path in range(22):
    #     name = 'in'+ str("%02d" % i)
    #     img = cv2.imread(file_path)
    #     cv2.imwrite('/home/fulan/WCT2/examples/content/' + name + '.png', img)
    #     i = i + 1
    #
    # file_path = '/home/fulan/WCT2/examples/style/night_style'
    # rootpath = os.listdir(file_path)
    # i = 1
    # for path in rootpath:
    #     name = 'in'+ str("%02d" % i)
    #     img = cv2.imread(file_path + '/' + path)
    #     cv2.imwrite(file_path + '/' + name + '.png', img)
    #     i = i + 1

    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, default='./datasets/Cars')
    parser.add_argument('--content_segment', type=str, default=None)  # './examples/content_segment'
    parser.add_argument('--style', type=str, default='./datasets/Style')
    parser.add_argument('--style_segment', type=str, default=None)  # './examples/style_segment/'
    parser.add_argument('--output', type=str, default='./datasets/Cars_aug')
    parser.add_argument('--image_size', type=int, default=None)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='sum', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_false')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_false')
    parser.add_argument('-s', '--transfer_at_skip', action='store_false')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_false')
    config = parser.parse_args()

    print(config)

    if not os.path.exists(os.path.join(config.output)):
        os.makedirs(os.path.join(config.output))

    '''
    CUDA_VISIBLE_DEVICES=6 python transfer.py --content ./examples/content --style ./examples/style --content_segment ./examples/content_segment --style_segment ./examples/style_segment/ --output ./outputs/ --verbose --image_size 512 -a
    '''
    run_bulk(config)
