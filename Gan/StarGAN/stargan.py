from __future__ import print_function, division
import numpy as np
import os
import cv2
from PIL import Image
import random
from functools import partial
from utils import *

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D,BatchNormalization,LeakyReLU,ReLU,UpSampling2D,Reshape,Dropout,concatenate,Lambda,Multiply,Add,Flatten,Dense
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class StarGAN(object):

    def __init__(self,args):

        self.c_dim = args.c_dim
        self.image_size = args.image_size
        self.g_conv_dim = args.g_conv_dim
        self.d_conv_dim = args.d_conv_dim
        self.g_repeat_num = args.g_repeat_num
        self.d_repeat_num = args.d_repeat_num
        self.lambda_class = args.lambda_cls
        self.lambda_rec = args.lambda_rec
        self.lambda_gp = args.lambda_gp

        #Train Config
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_iters_decay = args.num_iters_decay
        self.g_lr = args.g_lr
        self.d_lr = args.d_lr
        self.n_critic = args.n_critic
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.selected_attrs = args.selected_attrs

        # Test configurations.
        self.test_iters = args.test_iters

        # Miscellaneous.
        self.mode = args.mode 

        # Directories.
        self.data_dir = args.data_dir
        self.sample_dir = args.sample_dir
        self.model_save_dir = args.model_save_dir
        self.result_dir = args.result_dir

        # Step size.
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.lr_update_step = args.lr_update_step

        # Custom image
        self.custom_image_name = args.custom_image_name
        self.custom_image_label = args.custom_image_label

    
    def residual_Block(self,input_layer,dim_out):
        x = ZeroPadding2D(padding=1)(input_layer)
        x = Conv2D(filters=dim_out,kernel_size=3,strides=1,padding='valid',use_bias = False)(x)
        x = InstanceNormalization(axis=-1)(x)
        x = ReLU()(x)
        x = ZeroPadding2D(padding=1)(x)
        x = Conv2D(filters=dim_out,kernel_size=3,strides=1,padding='valid',use_bias=False)(x)
        x = InstanceNormalization(axis=-1)(x)
        return Add()([input_layer,x])

    def build_generator(self):

        inp_c = Input(shape = (self.c_dim, ))
        inp_img = Input(shape = (self.image_size, self.image_size, 3))
    
        # Replicate spatially and concatenate domain information
        c = Lambda(lambda x: K.repeat(x, self.image_size**2))(inp_c)
        c = Reshape((self.image_size, self.image_size, self.c_dim))(c)
        x = concatenate([inp_img, c])
    
        # First Conv2D
        x = Conv2D(filters = self.g_conv_dim, kernel_size = 7, strides = 1, padding = 'same', use_bias = False)(x)
        x = InstanceNormalization(axis = -1)(x)
        x = ReLU()(x)
    
        # Down-sampling layers
        curr_dim = self.g_conv_dim
        for i in range(2):
            x = ZeroPadding2D(padding = 1)(x)
            x = Conv2D(filters = curr_dim*2, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False)(x)
            x = InstanceNormalization(axis = -1)(x)
            x = ReLU()(x)
            curr_dim = curr_dim * 2
        
        # Bottleneck layers.
        for i in range(self.g_repeat_num):
            x = self.residual_Block(x, curr_dim)
        
        # Up-sampling layers
        for i in range(2):
            x = UpSampling2D(size = 2)(x)       
            x = Conv2D(filters = curr_dim // 2, kernel_size = 4, strides = 1, padding = 'same', use_bias = False)(x)
            x = InstanceNormalization(axis = -1)(x)
            x = ReLU()(x)        
            curr_dim = curr_dim // 2
    
        # Last Conv2D
        x = ZeroPadding2D(padding = 3)(x)
        out = Conv2D(filters = 3, kernel_size = 7, strides = 1, padding = 'valid', activation = 'tanh', use_bias = False)(x)
    
        return Model(inputs = [inp_img, inp_c], outputs = out) 
    
    def build_discriminator(self):

        inp_img = Input(shape = (self.image_size, self.image_size, 3))
        x = ZeroPadding2D(padding = 1)(inp_img)
        x = Conv2D(filters = self.d_conv_dim, kernel_size = 4, strides = 2, padding = 'valid', use_bias = False)(x)
        x = LeakyReLU(0.01)(x)
    
        curr_dim = self.d_conv_dim
        for i in range(1, self.d_repeat_num):
            x = ZeroPadding2D(padding = 1)(x)
            x = Conv2D(filters = curr_dim*2, kernel_size = 4, strides = 2, padding = 'valid')(x)
            x = LeakyReLU(0.01)(x)
            curr_dim = curr_dim * 2
    
        kernel_size = int(self.image_size / np.power(2, self.d_repeat_num))
    
        out_src = ZeroPadding2D(padding = 1)(x)
        out_src = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'valid', use_bias = False)(out_src)
    
        out_cls = Conv2D(filters = self.c_dim, kernel_size = kernel_size, strides = 1, padding = 'valid', use_bias = False)(x)
        out_cls = Reshape((self.c_dim, ))(out_cls)
    
        return Model(inp_img, [out_src, out_cls]) 

    def classification_loss(self,y_true,y_pred):
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    def w_loss(self,y_true,y_pred):
        return K.mean(y_true * y_pred)
    
    def reconstruct_loss(self,y_true,y_pred):
        return K.mean(K.abs(y_true - y_pred))
    
    def train_d(self,)