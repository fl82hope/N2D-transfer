import argparse
import os

import trainer

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # Pre-train, saving, and loading parameters
    parser.add_argument('--save_path', type=str, default='./models', help='saving path that is a folder')
    parser.add_argument('--sample_path', type=str, default='./samples', help='training samples path that is a folder')
    parser.add_argument('--save_mode', type=str, default='epoch',
                        help='saving mode, and by_epoch saving is recommended')
    parser.add_argument('--save_by_epoch', type=int, default=10, help='interval between model checkpoints (by epochs)')
    parser.add_argument('--save_by_iter', type=int, default=100000,
                        help='interval between model checkpoints (by iterations)')
    parser.add_argument('--load_name', type=str, default='', help='load the pre-trained model with certain epoch')
    # GPU parameters
    parser.add_argument('--multi_gpu', type=bool, default=True, help='True for more than 1 GPU')
    parser.add_argument('--gpu_ids', type=str, default='0, 1', help='gpu_ids: e.g. 0  0,1  0,1,2  use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='True for unchanged input data type')
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs of training')
    parser.add_argument('--train_batch_size', type=int, default=16, help='size of the batches')
    parser.add_argument('--lr_g', type=float, default=0.0002, help='Adam: learning rate for G / D')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=8, help='weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type=int, default=20,
                        help='lr decrease at certain epoch and its multiple')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='number of cpu threads to use during batch generation')
    # Initialization parameters
    parser.add_argument('--color', type=bool, default=True, help='input type')
    parser.add_argument('--burst_length', type=int, default=1, help='number of photos used in burst setting')
    parser.add_argument('--blind_est', type=bool, default=True, help='variance map')
    parser.add_argument('--kernel_size', type=list, default=[5], help='kernel size')
    parser.add_argument('--sep_conv', type=bool, default=False, help='simple output type')
    parser.add_argument('--channel_att', type=bool, default=False, help='channel wise attention')
    parser.add_argument('--spatial_att', type=bool, default=False, help='spatial wise attention')
    parser.add_argument('--upMode', type=str, default='bilinear', help='upMode')
    parser.add_argument('--core_bias', type=bool, default=False, help='core_bias')
    parser.add_argument('--init_type', type=str, default='xavier', help='initialization type of generator')
    parser.add_argument('--init_gain', type=float, default=0.02, help='initialization gain of generator')
    # Dataset parameters
    parser.add_argument('--baseroot', type=str, default='./datasets/Cars', help='images baseroot')
    parser.add_argument('--baseaug', type=str, default='./datasets/Cars_aug', help='images baseroot')
    parser.add_argument('--basexml', type=str, default='./datasets/xml', help='images baseroot')
    parser.add_argument('--crop_size', type=int, default=720, help='single patch size')
    parser.add_argument('--geometry_aug', type=bool, default=False, help='geometry augmentation (scaling)')
    parser.add_argument('--angle_aug', type=bool, default=False, help='geometry augmentation (rotation, flipping)')
    parser.add_argument('--scale_min', type=float, default=1, help='min scaling factor')
    parser.add_argument('--scale_max', type=float, default=1, help='max scaling factor')
    parser.add_argument('--mu', type=int, default=0, help='Gaussian noise mean')
    parser.add_argument('--sigma', type=int, default=30, help='Gaussian noise variance: 30 | 50 | 70')

    # augmix style parameters
    parser.add_argument('--basestyle', type=str, default='./datasets/Style', help='images basestyle')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--option_unpool', type=str, default='sum', choices=['sum', 'cat5'])
    parser.add_argument('-e', '--transfer_at_encoder', action='store_false')
    parser.add_argument('-d', '--transfer_at_decoder', action='store_false')
    parser.add_argument('-s', '--transfer_at_skip', action='store_false')
    parser.add_argument('-a', '--transfer_all', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_false')
    parser.add_argument('--mixture_width', default=3, type=int,
                        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument('--mixture_depth', default=-1, type=int,
                        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')

    parser.add_argument('--lamda', default=1, type=int,
                        help='balancle L1 and SSIM')
    parser.add_argument('--loss_det', default='L1', type=str,
                        help='balancle L1 and SSIM')

    parser.add_argument('--unpair', default=False, type=bool,
                        help='balancle L1 and SSIM')
    opt = parser.parse_args()
    print(opt)

    '''
    # ----------------------------------------
    #        Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
        print('Multi-GPU mode, %s GPUs are used' % (opt.gpu_ids))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('Single-GPU mode')
    '''

    # ----------------------------------------
    #       Choose pre / continue train
    # ----------------------------------------
    if opt.unpair:
        trainer.Pre_train_unpair(opt)
    else:
        trainer.Pre_train(opt)
