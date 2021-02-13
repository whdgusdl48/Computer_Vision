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
        
        #build model
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.G_optimizer = tf.keras.optimizers.Adam(lr = self.g_lr,beta_1=self.beta1,beta_2 = self.beta2)
        self.D_optimizer = tf.keras.optimizers.Adam(lr = self.d_lr,beta_1=self.beta1,beta_2 = self.beta2)
        self.G.summary()
        self.D.summary()
        self.Image_data_class = ImageData(data_dir=self.data_dir, select_attrs=self.selected_attrs)
        self.Image_data_class.preprocess()

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
        x = LeakyReLU(0.2)(x)
    
        curr_dim = self.d_conv_dim
        for i in range(1, self.d_repeat_num):
            x = ZeroPadding2D(padding = 1)(x)
            x = Conv2D(filters = curr_dim*2, kernel_size = 4, strides = 2, padding = 'valid')(x)
            x = LeakyReLU(0.2)(x)
            curr_dim = curr_dim * 2
    
        kernel_size = int(self.image_size / np.power(2, self.d_repeat_num))
    
        out_src = ZeroPadding2D(padding = 1)(x)
        out_src = Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = 'valid',use_bias = False)(out_src)
    
        out_cls = Conv2D(filters = self.c_dim, kernel_size = kernel_size, strides = 1, padding = 'valid', use_bias = False)(x)
        out_cls = Reshape((self.c_dim, ))(out_cls)
    
        return Model(inp_img, [out_src, out_cls]) 
 

    def gradient_penalty(self,f,real,fake):
        
        in_shape = K.shape(real)
        shape = K.concatenate([in_shape[0:1], K.ones_like(in_shape[1:], dtype='int32')], axis=0)
        alpha = K.random_uniform(shape)
        inter =  (alpha * real) + ((1 - alpha) * fake)
        with tf.GradientTape() as t:
            t.watch(inter)
            gp_src,pred = f(inter)
        grad = t.gradient(pred,[inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad)))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    def classification_loss(self, Y_true, Y_pred) :
        return -tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))

    def wasserstein_loss(self, r_logit, f_logit):
        f_loss = tf.reduce_mean(f_logit)
        r_loss = tf.reduce_mean(r_logit)
        return f_loss - r_loss

    def reconstruction_loss(self, Y_true, Y_pred):
        return K.mean(K.abs(Y_true - Y_pred))

    def train_g(self,x_real,labels,orig_labels):
        # self.train_G = Model([real_x, org_label, trg_label], [fake_out_src, fake_out_cls, x_reconst])
        # G_loss = self.train_G.train_on_batch(x = [imgs, orig_labels, target_labels], y = [valid, target_labels, imgs])
        with tf.GradientTape() as t:
            valid = np.ones((self.batch_size, 2, 2, 1))
            fake =  -np.ones((self.batch_size, 2, 2, 1))

            x_fake = self.G([x_real,labels],training=True)
            real_src, real_domain = self.D(x_real,training=True)
            fake_src, fake_domain = self.D(x_fake,training=True)

            src_loss = self.wasserstein_loss(real_src,fake_src)

            fake_cls_loss = self.classification_loss(fake_domain,labels)

            x_recon = self.G([x_fake,orig_labels],training=True)
            rec_loss = self.reconstruction_loss(x_real,x_recon)
            loss =  src_loss + fake_cls_loss + 10 * rec_loss
        grad = t.gradient(loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad, self.G.trainable_variables))
        return [loss,src_loss,fake_cls_loss,rec_loss]
    # train loss
    def train_d(self,x_real,labels,orig_labels):
        # D_loss = self.train_D.train_on_batch(x = [imgs, target_labels], y = [valid, orig_labels, fake, dummy])
        # self.train_D = Model([x_real, label_trg], [out_src_real, out_cls_real, out_src_fake, out_src])
        with tf.GradientTape() as t:
            x_fake = self.G([x_real,labels],training=True)

            real_src, real_domain = self.D(x_real,training=True)
            fake_src, fake_domain = self.D(x_fake,training=True)
            
            src_loss = self.wasserstein_loss(real_src,fake_src)

            real_cls_loss = self.classification_loss(real_domain,orig_labels)
            fake_cls_loss = self.classification_loss(fake_domain,labels)
            
            gp = self.gradient_penalty(partial(self.D,training=True),x_real,x_fake) 
            loss = -src_loss + real_cls_loss + 10 * gp
        grad = t.gradient(loss,self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grad,self.D.trainable_variables))
        return [loss,src_loss,real_cls_loss,gp]

    def train(self):
        data_iter = get_loader(self.Image_data_class.train_dataset, self.Image_data_class.train_dataset_label, self.Image_data_class.train_dataset_fix_label, 
                               image_size=self.image_size, batch_size=self.batch_size, mode=self.mode)
        
        for epoch in range(self.num_iters):
            imgs,orig_labels,target_labels,fix_labels, _ = next(data_iter)

            G_loss = self.train_g(imgs,target_labels,orig_labels)

            if (epoch + 1) % self.n_critic == 0:
                D_loss = self.train_d(imgs,target_labels,orig_labels)
            
            if (epoch + 1) % self.log_step == 0:
                print(f"Iteration: [{epoch + 1}/{self.num_iters}]")
                print(f"\tD/_loss = [{D_loss[1]:.4f}], D/loss_cls =  [{D_loss[2]:.4f}], D/loss_gp = [{D_loss[-1]:.4f}]")
                print(f"\tG/loss_fake = [{G_loss[1]:.4f}], G/loss_rec = [{G_loss[3]:.4f}], G/loss_cls = [{G_loss[2]:.4f}]") 
            
            if (epoch + 1) % self.model_save_step == 0:  
                self.G.save_weights(os.path.join(self.model_save_dir, 'G_weights.hdf5'))
                self.D.save_weights(os.path.join(self.model_save_dir, 'D_weights.hdf5'))
                self.train_D.save_weights(os.path.join(self.model_save_dir, 'train_D_weights.hdf5'))
                self.train_G.save_weights(os.path.join(self.model_save_dir, 'train_G_weights.hdf5')) 

    def test(self):
        G_weights_dir = os.path.join(self.model_save_dir, 'G_weights.hdf5')
        if not os.path.isfile(G_weights_dir):
            print("Don't find weight's generator model")
        else:
            self.G.load_weights(G_weights_dir)

        data_iter = get_loader(self.Image_data_class.test_dataset, self.Image_data_class.test_dataset_label, self.Image_data_class.test_dataset_fix_label, 
                               image_size=self.image_size, batch_size=self.batch_size, mode=self.mode)        
        n_batches = int(len(self.sample_step) / self.batch_size)
        total_samples = n_batches * self.batch_size

        for i in range(n_batches):
            imgs, orig_labels, target_labels, fix_labels, names = next(data_iter)
            for j in range(self.batch_size):
                preds = self.G.predict([np.repeat(np.expand_dims(imgs[j], axis = 0), len(self.selected_attrs), axis = 0), fix_labels[j]])
                for k in range(len(self.selected_attrs)):                    
                    Image.fromarray((preds[k]*127.5 + 127.5).astype(np.uint8)).save(os.path.join(self.result_dir, names[j].split(os.path.sep)[-1].split('.')[0] + f'_{k + 1}.png'))