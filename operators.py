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
import sys
import numpy as np
import tensorflow as tf

def lerp(a,b, t):
    with tf.name_scope('lerp'): #just affect to operation name
        return a + (b-a)*t


def generator_block(input, config, reuse=tf.AUTO_REUSE):
    #input:  [none, self.z_dim]
    #output: phase 1:  [none, 512, 4, 4] -> [none, 3, 1024] by rgb + upscale transform
    #        phase 2:  [none, 512, 8, 8] -> [none, 3, 1024]
    #         ...
    #        phase 10: [3, 1024, 1024]   -> [none, 3, 1024]
    with tf.variable_scope("generator", reuse=reuse) as scope:
        input.set_shape([None, config.z_dim])


    g_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")

    return img, g_variables

def discriminator_block(input, config, reuse=tf.AUTO_REUSE):
    #input: phase 1:   [none, 3, 1024, 1024] -> [none, 512, 4, 4]
    #       phase 2:   [none, 3, 1024, 1024] -> [none, 512, 8, 8]
    #         ...
    #       phase 10:  [none, 3, 1024, 1024] -> [none, 3, 1024, 1024]
    #output: [none, 1]
    with tf.variable_scope("discriminator", reuse=reuse) as scope:
        input.set_shape([None, config.c_dim, config.final_resolution, config.final_resolution])


    d_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

    return img, d_variables
def import_function(func_name):


    return module

class Network(object):
    def __init__(self, type, config, *args):

        self.config = config
        self.trainable_vars = None
        self._build_func = import_function('generator_block' if type=='G' else 'discriminator_block')


    def get_output(self, input):
        img, self.trainable_vars = self._build_func(input, self.config)
        return img

    def set_variable(self, var_name, var_value):

    def find_var(self, var_name):



