"""
Image Variational Autoencoding
"""

import sys
import os
import argparse
import numpy as np
from image_vae import ImageVAE

parser = argparse.ArgumentParser(description='')

parser.add_argument('--data_dir', type=str, default='data', help='input data directory (in train subfolder)')
parser.add_argument('--save_dir', type=str, default='save', help='save directory')
parser.add_argument('--phase', type=str, default='train', help='train, load, or dice')
parser.add_argument('--checkpoint', type=str, default='NA', help='checkpoint weight file')

parser.add_argument('--image_size', type=int, default=64, help='image size')
parser.add_argument('--image_channel', type=int, default=3, help='image channels')
parser.add_argument('--image_res', type=int, default=8, help='image resolution (8 or 16)')

parser.add_argument('--latent_dim', type=int, default=2, help='latent dimension')
parser.add_argument('--inter_dim', type=int, default=64, help='intermediate dimension')
parser.add_argument('--num_conv', type=int, default=3, help='number of convolutions')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=2, help='training epochs')
parser.add_argument('--nfilters', type=int, default=64, help='num convolution filters')
parser.add_argument('--learn_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--epsilon_std', type=float, default=1.0, help='epsilon width')
parser.add_argument('--latent_samp', type=int, default=10, help='number of latent samples')
parser.add_argument('--num_save', type=int, default=8, help='number of reconstructed images to save')
parser.add_argument('--verbose', type=int, default=2, help='1=verbose, 2=quiet')
parser.add_argument('--steps_per_epoch', type=int, default=0, help='steps per epoch')

parser.add_argument('--is_numpy', default=False, action='store_true',
                    help='images are represented as numpy arrays')
parser.add_argument('--channel_first', default=False, action='store_true',
                    help='specify if images are channel first')
parser.add_argument('--channels_to_use', type=str, default='all',
                    help='specify channels to use if numpy arrays are used')
parser.add_argument('--save_individual', default=False, action='store_true',
                    help='if numpy, specify to save individual image reconstruction')
parser.add_argument('--channel_labels', type=str, default='NA',
                    help='specify channel labels if numpy arrays are used')
args = parser.parse_args()


def main():
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
    if not args.is_numpy:
        os.makedirs(os.path.join(args.save_dir, 'latent_walk'), exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'reconstructed'), exist_ok=True)
    if args.save_individual:
        os.makedirs(os.path.join(args.save_dir, 'individual_cells'), exist_ok=True)

    if args.is_numpy:
        if args.channels_to_use != 'all':
            channels = np.array(args.channels_to_use.split(',')).astype(int)
            if len(channels) != args.image_channel:
                sys.exit('Number of specified channels does not image_channel argument!')
        else:
            args.channels_to_use = ','.join(str(i) for i in list(range(args.image_channel)))
        if args.channel_labels != 'NA':
            ch_names = np.array(args.channels_to_use.split(',')).astype(int)
            if len(ch_names) != args.image_channel:
                sys.exit('Number of channel names does not image_channel argument!')
        else:
            args.channel_labels = ','.join(str(i) for i in list(range(args.image_channel)))

    if args.phase == 'train':
        model = ImageVAE(args)
        model.train()

    if args.phase == 'load':
        if args.checkpoint == 'NA':
            sys.exit('No checkpoint file provided')

        model = ImageVAE(args)
        model.vae.load_weights(args.checkpoint)
        model.train()

    if args.phase == 'dice':
        if args.checkpoint == 'NA':
            sys.exit('No checkpoint file provided')

        model = ImageVAE(args)
        model.vae.load_weights(args.checkpoint)
        model.dice()


if __name__ == '__main__':
    main()
