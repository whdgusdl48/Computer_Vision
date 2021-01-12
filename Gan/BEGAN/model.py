import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from  tensorflow.keras.layers import Dropout,Input,ELU,Conv2D,AveragePooling2D,Activation,BatchNormalization,LeakyReLU,Dense,Reshape,Conv2DTranspose,Flatten,UpSampling2D
from tensorflow.keras.models import Sequential, Model
import os
import time
import matplotlib.pyplot as plt


class BEGAN():

    def __init__(self,input_shape,filter_number,batch_size,gamma,lamda,checkpoint_dir):
        self.input_shape = input_shape
        self.hidden_num = 128
        self.z_dim = 128
        self.k_dim = 0
        self.filter_number = filter_number
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.generator = self.build_generator()
        self.kt = 0.
        self.batch_size = batch_size
        self.gamma = gamma
        self.lamda = lamda
        self.epochs = 500
        self.checkpoint_dir = checkpoint_dir
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4,0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4,0.5)

    def l1_loss(self,x,y):
        return tf.reduce_mean(tf.abs(x - y))


    def train_step(self,images):
       
        batch_size = self.batch_size
        noise =tf.random.normal((self.batch_size,self.z_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generator_images = self.generator(noise,training=True)

            real_output = self.decoder(self.encoder(images,training=True),training=True)
            fake_output = self.decoder(self.encoder(generator_images,training=True),training=True)
            d_real_loss = self.l1_loss(images,real_output)
            d_fake_loss = self.l1_loss(generator_images,fake_output)
            disc_loss = d_real_loss - self.kt * d_fake_loss
            gen_loss = -disc_loss
            m_global_loss = disc_loss + np.abs(self.gamma * d_real_loss - d_fake_loss)
            self.kt = np.maximum(np.minimum(1., self.kt + self.lamda * (self.gamma * d_real_loss - d_fake_loss)), 0.)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.encoder.trainable_variables)

        g_loss = self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        d_loss = self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.encoder.trainable_variables))  
        return gen_loss,disc_loss

    def train(self,dataset):
        
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.decoder
                                )

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            for image_batch in dataset:
                d_loss,g_loss = self.train_step(image_batch)
                # print ('d_losds {:.5f} g_loss {:.5f}'.format(d_loss,g_loss))
            seed = tf.random.normal([16, self.z_dim])
            predictions = self.generator(seed)
            predictions = np.array(predictions)
            print(predictions[0])
            fig = plt.figure(figsize=(4,4))

            for i in range(predictions.shape[0]):
                plt.subplot(4, 4, i+1)
                plt.imshow(predictions[i, :, :, :], cmap='gray_r')
                plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/BEGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            
            if (epoch + 1) % 15 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time for epoch {} is {} sec d_loss {:.5f} g_loss {:.5f}'.format(epoch + 1, time.time()-start,d_loss,g_loss))

        seed = np.random.normal(0,1,(self.batch_size,self.z_dim))
        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, :], cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/DCGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
    def build_generator(self):
        input = Input(shape=(self.z_dim,))
        x = input
        x = Dense(np.prod((8,8,self.hidden_num )))(x)
        x = Reshape((8,8,self.hidden_num ))(x)
        x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
        x = Activation('elu')(x)

        if self.input_shape[0] == 128:
            for i in range(4):
                x = UpSampling2D((2,2),interpolation='nearest')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
        else:
            for i in range(3):
                x = UpSampling2D((2,2),interpolation='nearest')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)

        x = Conv2D(filters = 3, kernel_size = 3, strides = 1,padding='same')(x)

        model = Model(input,x)
        return model
        
    def build_encoder(self):
        input = Input(shape=self.input_shape) 
        x = input
        
        x = Conv2D(filters=self.filter_number,kernel_size=3,strides=1,padding='same')(x)
        x = Activation('elu')(x)

        if self.input_shape[0] == 128:
            for i in range(4):
                x = Conv2D(filters=self.filter_number * (i + 1),kernel_size=3,strides=1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.filter_number * (i + 1),kernel_size=3,strides=1,padding='same')(x)
                x = Activation('elu')(x)
                x = AveragePooling2D((2,2))(x)
        else:
            for i in range(3):
                x = Conv2D(filters=self.filter_number * (i + 1),kernel_size=3,strides=1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.filter_number * (i + 1),kernel_size=3,strides=1,padding='same')(x)
                x = Activation('elu')(x)
                x = AveragePooling2D((2,2))(x)
        self.k_dim = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.z_dim)(x)
        model = Model(input,x)
        return model
    
    def build_decoder(self):
        input = Input(shape=(self.z_dim,))
        x = input
        x = Dense(np.prod((8,8,self.hidden_num )))(x)
        x = Reshape((8,8,self.hidden_num))(x)
        x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
        x = Activation('elu')(x)
        x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
        x = Activation('elu')(x)

        if self.input_shape[0] == 128:
            for i in range(4):
                x = UpSampling2D((2,2),interpolation='nearest')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
        else:
            for i in range(3):
                x = UpSampling2D((2,2),interpolation='nearest')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)
                x = Conv2D(filters=self.hidden_num,kernel_size=3,strides = 1,padding='same')(x)
                x = Activation('elu')(x)

        x = Conv2D(filters = 3, kernel_size = 3, strides = 1,padding='same')(x)

        model = Model(input,x)
        return model