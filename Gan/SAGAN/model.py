import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from  tensorflow.keras.layers import Dropout,Add,Input,ReLU,Conv2D,Activation,BatchNormalization,LeakyReLU,Dense,Reshape,Conv2DTranspose,Flatten
from tensorflow.keras.models import Sequential, Model
import os
import time
import matplotlib.pyplot as plt
import partial

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

    def __init__(self,
                 generator_layer_num,
                 discriminator_layer_num,
                 image_size,
                 batch_size,
                 checkpoint_dir,
                 epochs):
        self.z_dim = 128
        self.generator_layer_num = generator_layer_num
        self.discriminator_layer_num = discriminator_layer_num
        self.image_size = image_size
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.0004,0.0)
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0001,0.9)
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.epochs = epochs

    def train(self,dataset):
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator
                                )

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            z = np.random.normal(0,1,(self.batch_size,self.z_dim))
            for image_batch in dataset:
                d_loss = self.train_d(image_batch)
                g_loss = self.train_g()
            
            seed = tf.random.uniform([1, self.z_dim],-1.,1.)
            predictions = self.generator(seed)
            predictions = np.array(predictions)
            predictions = (predictions + 1) * 127.5
            predictions = predictions.astype(np.uint8)
            plt.imshow(predictions [0,:, :, :], cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/PGGAN/image/image_at_epoch_{:04d}.png'.format(epoch))

            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec d_loss {:.5f} g_loss {:.5f}'.format(epoch + 1, time.time()-start,d_loss,g_loss))

        predictions = self.generator(seed)
        predictions = np.array(predictions)
        predictions = (predictions + 1) * 127.5
        predictions = predictions.astype(np.uint8)
        plt.imshow(predictions [0,:, :, :], cmap='gray_r')
        plt.axis('off')

        plt.savefig('/home/ubuntu/bjh/Gan/PGGAN/image/image_at_epoch_{:04d}.png'.format(epoch))   

    def gradient_penalty(self,real,fake):
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

    def train_d(self,real):
        z = np.random.normal(0,1,(self.batch_size,self.z_dim))

        with tf.GradientTape() as t:
            x_fake = self.generator(z,training=True)
            fake_logits = self.discriminator(x_fake,training=True)
            real_logits = self.discriminator(real,training=True)    
            f_loss = tf.reduce_mean(fake_logits)
            r_loss = tf.reduce_mean(real_logits)
            loss = f_loss + r_loss
            gp = self.gradient_penalty(partial(self.discriminator,training=True),real,x_fake)
            loss += 5 * gp

        grad = t.gradient(cost,self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(grad,self.discriminator.trainable_variable))

        return loss

    def train_g(self):
        z = np.random.normal(0,1,(self.batch_size,self.z_dim))

        with tf.GradientTape() as t:
            x_fake = self.generator(z,training=True)
            fake_logits = self.discriminator(x_fake,training=True)
            loss = -tf.reduce_mean(fake_logits)
        
        grad = t.gradient(loss,self.generator.trainable_variable)
        self.generator_optimizer.apply_gradients(zip(grad,self.generator_optimizer))

        return loss

    def UpBlock(self,layers,filters):
        
        x = BatchNormalization()(layers)
        x = ReLU()(x)
        x = UpSampling2D()(x)
        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)

        add_x = layers
        add_x = UpSampling2D()(x)
        add_x = Conv2D(filters=filters,kernel=1,strides=1,padding='same')(x)

        add = Add()([x,add_x])

        return add

    def DownBlock(self,layers,filters,to_down=True):
        
        init_channel = layers.shape.as_list()[:-1]

        x = LeakyReLU(alpha=0.2)(layers)
        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)

        x = LeakyReLU(alpha=0.2)(layers)
        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)
        
        if to_down or init_channel != filters:
            layers = Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(layers)
            if to_down:
                layers = tf.keras.layers.AveragePooling2D(2,2)(layers)

        add = Add()([x,layers])

        return add
    
    def init_down_resblock(self,layers):

        x = ReflectionPadding2D(padding=(1,1))(layers)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)

        x = ReflectionPadding2D(padding=(1,1))(layers)
        x = Conv2D(filters=filters,kernel_size=3,strides=1,padding='valid')(x)

        add_x = tf.keras.layers.AveragePooling2D(2,2)(layers)
        add_x = Conv2D(filters=filters,kernel=1,strides=1,padding='same')(add_x)

        add = Add()([x,add_x])

        return add 

    def Attention(self,layers,filters):

        f = Conv2D(filters=filters//8,kernel_size=1,strides=1,padding='same')(layers)
        g = Conv2D(filters=filters//8,kernel_size=1,strides=1,padding='same')(layers)
        h = Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(layers)

        beta = tf.kears.layers.Multiply()([f,g])
        activaton1 = Activation('softmax')(beta)

        gamma = tf.keras.Multiply()(activaton1,h)

        x = Conv2D(filters=filters,kernel_size=1,strides=1,padding='same')(gamma)

        return x
    
    def build_generator(self)
        input = Input(shape=(self.z_dim))
        init_filters = 1024
        x = input
        x = Dense(4*4*init_filters)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU()(x)
        x = Reshape((4,4,init_filters))(x)

        x = self.UpBlock(x,init_filters)

        for i in range(0,self.generator_layer_num//2):
            x = self.UpBlock(x,filters=init_filters // 2)
            init_filters = init_filters // 2

        x = self.Attention(x,filters=init_filters)

        for i in range(self.generator_layer_num//2,self.generator_layer_num):    
            x = self.UpBlock(x,filters=init_filters // 2)
            init_filters = init_filters // 2

        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = ReflectionPadding2D(padding=(1,1))(x)
        x = Conv2D(filters=3,kernel_size=3,strides=1,padding='valid')(x)
        x = Activation('tanh')(x)

        model = Model(input,x)
        return model

    def build_discriminator(self):

        init_filters= 64
        input = Input(shape(self.image_size))
        x = self.init_down_resblock(input)
        x = self.DownBlock(x,init_filters * 2)
        x = self.Attention(x,init_filters * 2)

        init_filters = init_filters * 2

        for i in range(self.discriminator_layer_num):
            if i == self.discriminator_layer_num - 1:
                x = self.DownBlock(x,init_filters)
            else:
                x = self.DownBlock(x,init_filters * 2)

            init_filters = init_filters * 2

        x = LeakyReLU(0.2)(x)
        x = tf.keras.layers.GlobalAveragePooling2D(2,2)(x)
        x = Flatten()(x)
        x = Dense(1)

        model = Model(input,x)
        return model

                