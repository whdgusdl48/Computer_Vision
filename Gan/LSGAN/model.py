import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from  tensorflow.keras.layers import Dropout,Input,ReLU,Conv2D,Activation,BatchNormalization,LeakyReLU,Dense,Reshape,Conv2DTranspose,Flatten
from tensorflow.keras.models import Sequential, Model
import os
import time
import matplotlib.pyplot as plt

class LSGAN():

    def __init__(self,
                 input_shape,
                 discriminator_filters,
                 discriminator_strides,
                 generator_filters,
                 generator_strides,
                 batch_size,
                 checkpoint_dir):
        self.input_shape = input_shape
        self.z_dim = 1024
        self.epochs = 500
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.discriminator_filters = discriminator_filters
        self.discriminator_strides = discriminator_strides
        self.generator_filters = generator_filters
        self.generator_strides = generator_strides
        self.k_dim = (7,7,256)
        self.G = self.build_generator()
        self.D = self.build_discriminator()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-3,0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-3,0.5)
    
    def discriminator_loss(self,real,fake):
        mse = tf.losses.MeanSquaredError()
        r_loss = mse(tf.ones_like(real),real)
        f_loss = mse(tf.zeros_like(fake),fake)
        d_loss = r_loss + f_loss
        return d_loss

    def generator_loss(self,fake):
        mse = tf.losses.MeanSquaredError()
        f_loss = mse(tf.ones_like(fake),fake)
        return f_loss

    def train_step(self,images):
       
        batch_size = self.batch_size
        noise = tf.random.normal([batch_size,self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_images = self.G(noise,training=True)

            real_output = self.D(images,training=True)
            fake_output = self.D(generator_images,training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.G.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.D.trainable_variables)

        g_loss = self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        d_loss = self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))  
        return gen_loss,disc_loss

    def train(self,dataset):
        
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.G,
                                 discriminator=self.D
                                )

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            for image_batch in dataset:
                d_loss,g_loss = self.train_step(image_batch)
                # print ('d_losds {:.5f} g_loss {:.5f}'.format(d_loss,g_loss))
            seed = tf.random.normal([16, self.z_dim])
            predictions = self.G(seed)
            predictions = np.array(predictions) 
            print(predictions[0])
            fig = plt.figure(figsize=(4,4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(predictions[i, :, :, :], cmap='gray_r')
                plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/LSGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec d_loss {:.5f} g_loss {:.5f}'.format(epoch + 1, time.time()-start,d_loss,g_loss))

        seed = tf.random.normal([16, self.z_dim])
        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, :], cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/LSGAN/image/image_at_epoch_{:04d}.png'.format(epoch))


    def build_generator(self):
        input = Input(shape=(self.z_dim))
        x = input
        x = Dense(np.prod(self.k_dim))(x)
        x = Flatten()(x)
        x = Reshape(self.k_dim)(x)
        x = BatchNormalization()(x)

        for i in range(len(self.generator_strides)):
            x = Conv2DTranspose(filters=self.generator_filters[i],
                                kernel_size=(3,3),
                                strides=self.generator_strides[i],
                                padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

        x = Conv2DTranspose(filters=3,kernel_size=(3,3),strides=1,padding='same')(x)
        x = Activation('tanh')(x)
        model = Model(input,x)
        return model   

    def build_discriminator(self):
        input = Input(shape=self.input_shape)
        x = input
        for i in range(len(self.discriminator_strides)):
            x = Conv2D(filters=self.discriminator_filters[i],
                       kernel_size=(5,5),
                       strides=self.discriminator_strides[i],
                       padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        x = Dense(1)(x)
        model = Model(input,x)
        return model