# Original code
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# reference paper: https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf


import os
import numpy as np
import argparse
import tensorflow as tf

from model import PCGAN
from trainer import dataset

def main(config):

    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    model = PCGAN(config)

    if config.is_train:
        model.train()
    else:
        model.load()
        model.visualize()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', type=bool, default=True, help='training->true')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='adam optimizer learning rate')
    parser.add_argument('--beta1', type=float, default=0.0, help='adam optimizer beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='training batch_size')
    parser.add_argument('--epoch', type=int, default=25, help='training epoch')
    parser.add_argument('--z_dim', type=int, default=100, help='generating z')
    parser.add_argument('--c_dim', type=int, default=3, help='data color dim')
    parser.add_argument('--data_height', type=int, default=64, help='data height')
    parser.add_argument('--data_width', type=int, default=64, help='data width')
    parser.add_argument('--gfc_dim', type=int, default=1024, help='linear matrix dim in generator')
    parser.add_argument('--dfc_dim', type=int, default=512, help='linear matrix dim in discriminator')  # ??
    parser.add_argument('--default_batch_size', type=int, default=16, help='default_training batch_size')


    parser.add_argument('--data_dir', type=dir, default='celebahd', help='dir name for dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='dir name for saving or restoring')
    parser.add_argument('--summary_dir', type=str, default='summary', help='dir name of summary')
    parser.add_argument('--visualize_dir', type=str, default='visualize', help='dir name of visualization')

    config, unparsed = parser.parse_known_args()
    tf.app.run(main(config))