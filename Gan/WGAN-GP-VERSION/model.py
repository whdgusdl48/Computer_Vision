import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from  tensorflow.keras.layers import Dropout,Input,Conv2D,BatchNormalization,LeakyReLU,Dense,Reshape,Conv2DTranspose,Flatten
from tensorflow.keras.models import Sequential, Model
import os
import time
import matplotlib.pyplot as plt
from functools import partial
from tensorflow import reduce_mean
from tqdm.autonotebook import tqdm
from tensorflow.python.keras import metrics
import shutil
import random
class WGANGP():
    def __init__(self,
                 input_shape,
                 generator_filters,
                 critic_filters,
                 generator_strides,
                 critic_strides,
                 batch_size):
        self.input_shape = input_shape
        self.generator_filters = generator_filters
        self.critic_filters = critic_filters
        self.generator_strides = generator_strides
        self.critic_strides = critic_strides
        self.k_dim = 0
        self.z_dim = 200
        self.epochs = 500
        self.batch_size = batch_size
        # self.checkpoint_dir = checkpoint_dir
        self.generator_optimizer = tf.keras.optimizers.Adam(0.00002,0.5)
        self.critic_optimizer = tf.keras.optimizers.Adam(0.00002,0.5)
        self.n_critic = 5
        self.D = self.build_critic()
        self.G = self.build_generator()

        self.D.summary()

        self.G.summary()

    def train(self,dataset):
        z = np.random.normal(0,1,(self.batch_size,self.z_dim))
        g_train_loss = metrics.Mean()
        d_train_loss = metrics.Mean()

        for epoch in range(self.epochs):
            bar = self.pbar(2000,self.batch_size,epoch,self.epochs)
            for batch in dataset:
               
                for _ in range(self.n_critic):
                    self.train_d(batch)
                    d_loss = self.train_d(batch)
                    
                    d_train_loss(d_loss)
                
                g_loss = self.train_g()
                g_train_loss(g_loss)
                self.train_g()

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.update(self.batch_size)

            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

            sample = self.G(z,training=False)
            
            fig = plt.figure(figsize=(4,4))

            for i in range(sample.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(sample[i, :, :, :], cmap='gray_r')
                plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/WGAN-GP/image/image_at_epoch_{:04d}.png'.format(epoch))

    def pbar(self,total_images, batch_size, epoch, epochs):
        bar = tqdm(total=(total_images // batch_size) * batch_size,
               ncols=int(120 * .9),
               desc=tqdm.write(f'Epoch {epoch + 1}/{epochs}'),
               postfix={
                   'g_loss': f'{0:6.3f}',
                   'd_loss': f'{0:6.3f}',
                   1: 1
               },
               bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
               'ETA: {remaining}  Elapsed Time: {elapsed}  '
               'G Loss: {postfix[g_loss]}  D Loss: {postfix['
               'd_loss]}',
               unit=' images',
               miniters=10)
        return bar

    def d_loss_fn(self,f_logit, r_logit):
        f_loss = reduce_mean(f_logit)
        r_loss = reduce_mean(r_logit)
        return f_loss - r_loss

    def g_loss_fn(self,f_logit):
        f_loss = -reduce_mean(f_logit)
        return f_loss

    def train_g(self):
        z = np.random.normal(0,1,(self.batch_size,self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z,training=True)
            fake_logits = self.D(x_fake,training=True)
            loss = self.g_loss_fn(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss 

    def train_d(self,x_real):
        z = np.random.normal(0,1,(self.batch_size,self.z_dim))
        with tf.GradientTape() as t:
            x_fake = self.G(z,training=True)
            fake_logits = self.D(x_fake,training=True)
            real_logits = self.D(x_real,training=True)
            cost = self.d_loss_fn(fake_logits,real_logits)
            gp = self.gradient_penalty(partial(self.D,training=True),x_real,x_fake)
            cost += 5 * gp
        grad = t.gradient(cost,self.D.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grad,self.D.trainable_variables))
        return cost

    def gradient_penalty(self,f,real,fake):
        
        in_shape = K.shape(real)
        shape = K.concatenate([in_shape[0:1], K.ones_like(in_shape[1:], dtype='int32')], axis=0)
        alpha = K.random_uniform(shape)
        inter =  (alpha * real) + ((1 - alpha) * fake)
        with tf.GradientTape() as t:
            t.watch(inter)
            pred = f(inter)
        grad = t.gradient(pred,[inter])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad)))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp    


    def build_critic(self):
    
        input = Input(shape=self.input_shape)
        x = input
        for i in range(len(self.critic_filters)):
            x = Conv2D(filters=self.critic_filters[i],
                       kernel_size = (5,5),
                       strides=self.critic_strides[i],
                       padding='same')(x)
            x = LeakyReLU()(x)
        self.k_dim = K.int_shape(x)[1:]
        flatten = Flatten()(x)
        output = Dense(1)(flatten)
        model = Model(input,output)
        return model

    def build_generator(self):    
        
        stride = [2,2,2,2,2]
        input = Input(shape=(self.z_dim))
        x = input
        x = Dense(np.prod(self.k_dim))(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        x = Reshape(self.k_dim)(x)
        for i in range(len(self.generator_filters)):
            if i != len(self.generator_filters) -1:
                x = Conv2DTranspose(filters=self.generator_filters[i],
                                    kernel_size=(5,5),
                                    strides=self.generator_strides[i],
                                    padding='same')(x)
                x = BatchNormalization(momentum=0.9)(x)
                x = LeakyReLU(0.2)(x)
            else:
                x = Conv2DTranspose(filters=self.generator_filters[i],
                                    kernel_size=(5,5),
                                    strides=self.generator_strides[i],
                                    padding='same',
                                    )(x)
                x = LeakyReLU(0.2)(x)
                

        model = Model(input,x)
        return model