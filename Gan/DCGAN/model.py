import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from  tensorflow.keras.layers import Dropout,Input,Conv2D,BatchNormalization,LeakyReLU,Dense,Reshape,Conv2DTranspose,Flatten
from tensorflow.keras.models import Sequential, Model
import os
import time
import matplotlib.pyplot as plt

class DCGAN():

    def __init__(self,
                 input_shape,
                 generator_filters,
                 discriminator_filters,
                 generator_strides,
                 discriminator_strides,
                 checkpoint_dir,
                 batch_size):

        self.input_shape = input_shape
        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters
        self.generator_strides = generator_strides
        self.discriminator_strides = discriminator_strides
        self.k_dim = 0
        self.z_dim = 200
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.epochs = 100
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.generator_optimizer = tf.keras.optimizers.Adam(0.0002,0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(0.00005,0.5)

        self.build_discriminator()
        self.build_generator()

    def discriminator_loss(self,real_output,fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output),real_output)
        fake_loss = cross_entropy(tf.ones_like(fake_output),fake_output)
        total_loss = real_loss + fake_loss
        return total_loss 
    
    def generator_loss(self,fake_out):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_out),fake_out)

    def train_step(self,images):
       
        batch_size = self.batch_size
        noise = tf.random.normal([batch_size,self.z_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_images = self.generator(noise,training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generator_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        g_loss = self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        d_loss = self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))  
        return g_loss,d_loss

    def train(self,dataset):
        
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator)

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            for image_batch in dataset:
                d_loss,g_loss = self.train_step(image_batch)
                
            seed = tf.random.normal([16, self.z_dim])
            predictions = self.generator(seed)
            print(predictions[0] * 127.5 - 1)
            fig = plt.figure(figsize=(4,4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(predictions[i, :, :, :]* 127.5 - 1, cmap='gray_r')
                plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/DCGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

        seed = tf.random.normal([16, self.z_dim])
        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, :]* 127.5 - 1, cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/DCGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
           

    def build_generator(self):
        input = Input(shape=(self.z_dim,))
        x = input
        x = Dense(np.prod(self.k_dim))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

        x = Reshape((self.k_dim))(x)
        for i in range(len(self.generator_filters)):
            if i != len(self.generator_filters)-1:
                x = Conv2DTranspose(filters=self.generator_filters[i],
                                 kernel_size=(5,5),
                                 strides=self.generator_strides[i],
                                 padding='same')(x)
                x = BatchNormalization()(x)
                x = LeakyReLU()(x)
            else:
                x = Conv2DTranspose(filters=self.generator_filters[i],
                                 kernel_size=(5,5),
                                 strides=self.generator_strides[i],
                                 padding='same',
                                 activation='tanh')(x)
                   

        self.generator = Model(input,x)
        print(self.generator.summary())
    def build_discriminator(self):
        input = Input(shape=self.input_shape)
        x = input
        for i in range(len(self.discriminator_filters)):
            x = Conv2D(filters=self.discriminator_filters[i],
                       kernel_size=(5,5),
                       strides=self.discriminator_strides[i],
                       padding='same')(x)
            x = LeakyReLU()(x)
            x = Dropout(0.3)(x)
        self.k_dim = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(1,activation='sigmoid')(x)

        self.discriminator = Model(input,x)
        print(self.discriminator.summary())
        
