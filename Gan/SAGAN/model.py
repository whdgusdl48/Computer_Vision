import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Dense, BatchNormalization,Input,ReLU,UpSampling2D,InputSpec, Add, LeakyReLU,AveragePooling2D
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential, Model
import os
from functools import partial

class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')

class SAGAN():

    def __init__(self,image_size,z_dim,checkpoint_dir,batch_size):
        self.image_size = image_size
        self.z_dim = z_dim
        self.channel = 1024
        self.layer_num = int(np.log2(self.image_size[0])) - 3
        self.batch_size = batch_size
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.G_optimizer = tf.keras.optimizers.Adam(lr = 0.0001,beta_1=0,beta_2 = 0.9)
        self.D_optimizer = tf.keras.optimizers.Adam(lr = 0.0004,beta_1=0,beta_2 = 0.9)
        self.checkpoint_dir = checkpoint_dir
        self.epochs = 500

    def discriminator_loss(loss_func, real, fake):
        real_loss = 0
        fake_loss = 0

        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

        loss = real_loss + fake_loss

        return loss

    def generator_loss(loss_func, fake):

        fake_loss = 0
        fake_loss = -tf.reduce_mean(fake)
        loss = fake_loss
        return loss

    def train_g(self):
        z =tf.random.uniform([self.batch_size, self.z_dim],-1.,1.)
        with tf.GradientTape() as t:
            x_fake = self.G(z,training=True)
            fake_logits = self.D(x_fake,training=True)
            loss = self.generator_loss(fake_logits)
        grad = t.gradient(loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(grad, self.G.trainable_variables))
        return loss 

    def train_d(self,x_real):
        z = tf.random.uniform([self.batch_size, self.z_dim],-1.,1.)
        with tf.GradientTape() as t:
            x_fake = self.G(z,training=True)
            fake_logits = self.D(x_fake,training=True)
            real_logits = self.D(x_real,training=True)
            cost = self.discriminator_loss(real_logits,fake_logits)
            gp = self.gradient_penalty(partial(self.D,training=True),x_real,x_fake)
            cost += 10 * gp
        grad = t.gradient(cost,self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(grad,self.D.trainable_variables))
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

    def train(self,dataset):
        
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.G_optimizer,
                                 discriminator_optimizer=self.D_optimizer,
                                 generator=self.G,
                                 discriminator=self.D
                                )

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            for image_batch in dataset:
                d_loss = self.train_d(image_batch)
                g_loss = self.train_g()
                # print ('d_losds {:.5f} g_loss {:.5f}'.format(d_loss,g_loss))
            seed = tf.random.uniform([self.batch_size, self.z_dim],-1.,1.)
            predictions = self.G(seed)
            predictions = np.array(predictions)
            predictions = (predictions + 1) * 127.5
            predictions = predictions.astype(np.uint8)
            # print(predictions[0])
            fig = plt.figure(figsize=(4,4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(predictions[i, :, :, :], cmap='gray_r')
                plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/SAGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec d_loss {:.5f} g_loss {:.5f}'.format(epoch + 1, time.time()-start,d_loss,g_loss))

        seed = tf.random.uniform([self.batch_size, self.z_dim],-1.,1.)
        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, :], cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/SAGAN/image/image_at_epoch_{:04d}.png'.format(epoch))



    def up_resblock(self,layer,filters,use_bias=True):
        # resblock1
        x = BatchNormalization()(layer)
        x = ReLU()(x)
        x = UpSampling2D((2,2))(x)

        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=True)(x)
        x = ReLU()(x)

        layer = UpSampling2D((2,2))(layer)
        layer = Conv2D(filters=filters,kernel_size=1,strides=1,activation='relu',use_bias=False)(layer)

        return Add()([x,layer])

    def init_down_resblock(self,layer,filters):
        x = ReflectionPadding2D()(layer)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=True)(x)
        x = LeakyReLU(0.2)(x)

        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=True)(x)
        x = AveragePooling2D((2,2))(x)

        layer = AveragePooling2D((2,2))(layer)
        layer = Conv2D(filters=filters,kernel_size=1,strides=1,use_bias=True)(layer)

        return Add()([layer,x])
    def down_resblock(self,layer,filters,to_down=True):
        init_channel = K.int_shape(layer)[-1]
        x = LeakyReLU(0.2)(layer)
        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=True)(x)
       
        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,use_bias=True)(x)

        if to_down:
            x = AveragePooling2D((2,2))(x)
        
        if to_down or init_channel != filters:
            layer = Conv2D(filters=filters,kernel_size=1,strides=1,use_bias=True)(layer)
            if to_down:
                layer = AveragePooling2D((2,2))(layer)

        return Add()([layer,x])

    def build_discriminator(self):
        
        ch = 64
        input = Input(shape=self.image_size)
        x = input
        x = self.init_down_resblock(x,ch)

        x = self.down_resblock(x,ch * 2)
        x = self.attention_layer(x,ch * 2)

        ch = ch * 2

        for i in range(self.layer_num) :
            if i == self.layer_num - 1 :
                x = self.down_resblock(x, ch,to_down=False)
            else :
                x = self.down_resblock(x, ch * 2)

            ch = ch * 2

        x = LeakyReLU(0.2)(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        x = Dense(1)(x)

        model = Model(input,x)
        return model

    def attention_layer(self,x,channels):
        
        f = Conv2D(filters=channels//8,kernel_size=1,strides=1,padding='same')(x)
        g = Conv2D(filters=channels//8,kernel_size=1,strides=1,padding='same')(x)
        h = Conv2D(filters=channels,kernel_size=1,strides=1,padding='same')(x)

        f_shape = K.int_shape(f)[1:]
        g_shape = K.int_shape(g)[1:]
        h_shape = K.int_shape(h)[1:]

        f = Reshape((f_shape[0] * f_shape[1], f_shape[2]))(f)
        g = Reshape((g_shape[0] * g_shape[1], g_shape[2]))(g)
        h = Reshape((h_shape[0] * h_shape[1], h_shape[2]))(h)
        s = tf.matmul(f,g,transpose_b = True)
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta,h)
        
        o = Reshape(x.shape[1:])(o)

        o = Conv2D(channels,kernel_size=1,strides=1)(o)

        
        x = Add()([o,x])

        return x

    def build_generator(self): 
        input = Input(shape=(self.z_dim))
        ch = self.channel
        x = input
        x = Dense(np.prod((4,4,self.channel)))(x)
        x = Reshape((4,4,self.channel))(x)

        x = self.up_resblock(x,self.channel)

        for i in range(self.layer_num // 2) :
            x = self.up_resblock(x,ch // 2)
            ch = ch // 2

        x = self.attention_layer(x,ch)

        for i in range(self.layer_num//2,self.layer_num):
            x = self.up_resblock(x,ch // 2)
            ch = ch // 2
        
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = ReflectionPadding2D()(x)
        x = Conv2D(filters=3,kernel_size=3,strides=1,activation='tanh')(x)
        
        model = Model(input,x)
        return model

