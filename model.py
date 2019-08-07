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
import tensorflow as tf

from operators import *
from trainer import *




# Adjust minibatch size in case the training happens to run out of memory.
def adjust_minibatch(res, minibatch_size, minibatch_limits):
    limit = (minibatch_size - (minibatch_size - 1) // 16 - 1) // config.num_gpus
    limit = min(limit, minibatch_limits.get(res // 2, limit))
    group = config.D.get('mbstd_group_size', 4)
    if limit > group:
        limit = (limit // group) * group
    print('OUT OF MEMORY -- trying minibatch_limits[%d] = %d' % (res, limit))
    assert limit >= 1
    minibatch_limits = dict(minibatch_limits)
    minibatch_limits[res] = limit
    return minibatch_limits





def process_real(real_images, lod_in):
    #[batch, 3, 1024, 1024] -> [batch, 3, 1024,1024]
    c_dim, images_w, images_h = real_images.shape[1:]
    re_images = tf.reshape(real_images, [None, c_dim, lod_in, int(images_w/lod_in), lod_in, int(images_w/lod_in)])
    re_images = tf.reduce_mean(re_images, axis=[3, 5], keep_dims=True)
    ti_images = tf.tile(re_images, [1,1,1,int(images_w/lod_in),1, int(images_h/lod_in)])
    pro_images = tf.reshape(ti_images, [None, c_dim, images_w, images_h])
    return pro_images

def training_loss(g_class, d_class, real_images, latents_in,
                  wgan_target=1.0,
                  wgan_lambda=10.0,
                  wgan_epsilon=0.001):
    g_loss = []; d_loss = []
    with tf.name_scope("g_fake"):
        fake_images = g_class.get_output(latents_in)
    with tf.name_scope("d_real"):
        real_score = d_class.get_output(real_images)
    with tf.name_scope("d_fake"):
        fake_score = d_class.get_output(fake_images)

    with tf.name_scope('Mix'):
        mixing_factors = tf.random_uniform([tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
        mixed_images = lerp(tf.cast(real_images, tf.float32), fake_images, mixing_factors)
    with tf.name_scope('D_mixed'):
        mixed_score = d_class.get_output(mixed_images)
    with tf.name_scope('gradient_penalty'):
        mixed_grads = tf.gradients(tf.reduce_sum(mixed_score), mixed_images)
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads),axis=[1,2,3]))
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tf.square(real_score)
    g_loss += [(fake_score, -1.0)]
    d_loss += [(fake_score, 1.0)]
    d_loss += [(real_score, -1.0)]
    d_loss += [(gradient_penalty, wgan_lambda)]
    d_loss += [(epsilon_penalty, wgan_epsilon)]

    g_loss = tf.add_n([tf.reduce_mean((val*weight)) for val, weight in g_loss])
    d_loss = tf.add_n([tf.reduce_mean((val*weight)) for val, weight in d_loss])

    return g_loss, d_loss



class PCGAN(object):
    def __init__(self, config):
        self.config = config
        self.is_train = config.is_train
        self.learning_rate = config.learning_rate
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epoch = config.epoch
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.data_height = config.data_height
        self.data_width = config.data_width
        self.gfc_dim = config.gfc_dim
        self.dfc_dim = config.dfc_dim

        self.data_dir = config.data_dir
        self.checkpoint_dir = config.checkpoint_dir
        self.summary_dir = config.summary_dir
        self.visualize_dir = config.visualize_dir
        self.dataset = TFRecordDataset(data_dir=self.data_dir)
        self.cur_nimg = 0
        self.total_kimg = int(self.dataset.total_kimg / 1000)

        #direct input to placeholder
        self.cur_batch_num = config.default_batch_size
        self.cur_resolution = 4

        self.sess = tf.Session()

        self.build_model()
        self.saver = tf.train.Saver()

    def build_model(self):
        self.lod_in       = tf.placeholder(tf.float32, name='lod_in', shape=[])       #level of resolution
        self.minibatch_in = tf.placeholder(tf.int32, name='minibatch_in', shape=[])

        self.g_class = Network('G', self.config)
        self.d_class = Network('D', self.config)

        with tf.name_scope('inputs'):
            real_images = process_real(self.dataset.get_minibatch_tf(), self.lod_in)
            fake_latent = tf.random_normal([self.minibatch_in, self.z_dim])
            assign_ops  = [tf.assign(self.g_class.find_var('lod'), self.lod_in),tf.assign(self.d_class.find_var('lod'), self.lod_in)]

        with tf.control_dependencies(assign_ops):
            self.g_loss, self.d_loss = training_loss(self.g_class, self.d_class, real_images, fake_latent)
            optimizer_kwargs = {'learning_rate':self.learning_rate, 'beta1':self.beta1, 'beta2':self.beta2}
            self.d_trainer = tf.train.AdamOptimizer(**optimizer_kwargs).minimize(loss=self.d_loss, var_list=self.d_class.trainable_vars)
            self.g_trainer = tf.train.AdamOptimizer(**optimizer_kwargs).minimize(loss=self.g_loss, var_list=self.g_class.trainable_vars)


    def train(self):

        self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.epoch):
            while self.cur_nimg < self.total_kimg * 1000:
                self.dataset.configure(self.cur_batch_num)
                #...get parameters(self.cur_resolution)
                #

            #real trainging ops!
            try:
                self.sess.run([self.d_trainer, self.d_loss], feed_dict={self.lod_in:self.cur_resolution, self.minibatch_in:self.cur_batch_num})
                self.sess.run([self.g_trainer, self.g_loss], feed_dict={self.lod_in:self.cur_resolution, self.minibatch_in:self.cur_batch_num})
                self.cur_nimg += self.cur_batch_num

            except tf.errors.ResourceExhaustedError:
                self.cur_batch_num = adjust_minibatch(self.cur_batch_num, self.cur_resolution)

            #log progress -> resolution 바뀌는 nimg마다 5번씩 print하자

            #save snapshots -> log progress랑 지점 비슷하게!

        #write final results -> network저장하자!






    def load(self):



    def visualize(self):


        #self.generator     = generator(self.z)
        #self.discriminator = discriminator(self.input)
        #->이 방식은 gen, dis안의 내부 함수들을 이 pcgan class에 구현해야한다는의미
        #->pcgan의 함수 구조를 모든 gan에 통합하고 싶을시에 비효율적




