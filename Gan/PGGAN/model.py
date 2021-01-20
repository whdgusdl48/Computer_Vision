import time
import math
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import clear_output
import os
import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec, Conv2D, Conv2DTranspose, Activation, Reshape, LayerNormalization, BatchNormalization, UpSampling2D
from tensorflow.keras.layers import Input, UpSampling2D, Dropout, Concatenate, Add, Dense, Multiply, LeakyReLU, Flatten, AveragePooling2D, Multiply
from tensorflow.keras import initializers, regularizers, constraints, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import to_categorical, plot_model
from data_loader import image_loader_celeba,image_processing
from EqualizeLearningRate import EqualizeLearningRate
from PixelNormalization import PixelNoramlization
from MinibatchstdDev import MinibatchstdDev

class PGGAN():
    
    def __init__(self,batch_size,image_size,epochs,target_size,checkpoint_dir):
        self.batch_size = batch_size,
        self.image_size = image_size
        self.start_size = 4
        self.z_dim = 512
        self.lamda = 10
        self.epochs = epochs
        self.kernel_initializer = 'he_normal'
        self.output_activation = tf.nn.tanh
        self.target_size = target_size
        self.generator = None
        self.discriminator = None
        self.checkpoint_dir = checkpoint_dir
        self.generator_optimizer = Adam(learning_rate=1e-3,beta_1 = 0., beta_2 = 0.99,epsilon=1e-8)
        self.discriminator_optimizer=Adam(learning_rate=1e-3,beta_1 = 0., beta_2 = 0.99,epsilon=1e-8)

        self.build_model()
        self.CURRENT_EPOCH = 1
    def learning_rate_decay(self,current_lr=1,decay_factor=1.00004):
        new_lr = max(current_lr / decay_factor, MIN_LR)
        return new_lr

    def set_learning_rate(self,new_lr):
        K.set_value(self.discriminator_optimizer.lr,new_lr)
        K.set_value(self.generator_optimizer.lr,new_lr)
    
    def generate_and_save_images(self,model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    # Test input is a list include noise and label
        predictions = model.predict(test_input)
        fig = plt.figure(figsize=figure_size)
        for i in range(predictions.shape[0]):
            axs = plt.subplot(subplot[0], subplot[1], i+1)
            plt.imshow(predictions[i] * 0.5 + 0.5)
            plt.axis('off')
        if save:
            plt.savefig(os.path.join('/home/ubuntu/bjh/Gan/PGGAN/image', '{}x{}_image_at_epoch_{:04d}.png'.format(predictions.shape[1], predictions.shape[2], epoch)))

    def train_d(self,real_image,alpha):
        noise = tf.random.normal([4,self.z_dim])
        epsilon = tf.random.normal(shape=[4,1,1,1])
        # print(alpha)
        with tf.GradientTape() as d_tape:
            with tf.GradientTape() as gp_tape:
                fake = self.generator([noise,alpha],training=True)
                fake_mixed = epsilon * tf.dtypes.cast(real_image,tf.float32) + ((1 - epsilon) * fake)
                fake_mixed_predict = self.discriminator([fake_mixed,alpha], training=True)
        
            grads = gp_tape.gradient(fake_mixed_predict,fake_mixed)
            grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads),axis=[1,2,3]))
            gradient_penalty = tf.reduce_mean(tf.square(grads_norm - 1))

            fake_pred = self.discriminator([fake,alpha],training=True)
            # print(real_image.shape)
            real_pred = self.discriminator([real_image,alpha],training=True)

            D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + self.lamda * gradient_penalty

        D_gradients = d_tape.gradient(D_loss,self.discriminator.trainable_variables)

        self.discriminator_optimizer.apply_gradients(zip(D_gradients,self.discriminator.trainable_variables))
        return D_loss

    def train_g(self,alpha):
        noise = tf.random.normal([4,self.z_dim])

        with tf.GradientTape() as g_tape:
            fake_image = self.generator([noise,alpha],training=True)
            fake_pred = self.discriminator([fake_image,alpha],training=True)
            G_loss = -tf.reduce_mean(fake_pred)
        
        G_gradient = g_tape.gradient(G_loss,self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(G_gradient,self.generator.trainable_variables))
        return G_loss

    
    def train(self,dataset):
        current_learning_rate = 1e-3
        training_steps = math.ceil(4000 / 4)
        # Fade in half of switch_res_every_n_epoch epoch, and stablize another half
        alpha_increment = 1. / (40 / 2 * training_steps)
        alpha = min(1., (self.CURRENT_EPOCH - 1) % 40 * training_steps *  alpha_increment)
        checkpoint_dir = self.checkpoint_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                 discriminator_optimizer=self.discriminator_optimizer,
                                 generator=self.generator,
                                 discriminator=self.discriminator
                                )

        for epoch in range(self.CURRENT_EPOCH,self.epochs + 1):
            start = time.time()
            print('Start of epoch %d' % (epoch,))
            print('Current alpha: %f' % (alpha,))
            print('Current resolution: ',self.image_size)

            for image_batch in dataset:

                alpha_tensor = tf.constant(np.repeat(alpha,4).reshape(4,1),dtype=tf.float32)
                d_loss = self.train_d(image_batch,alpha_tensor)
                g_loss = self.train_g(alpha_tensor)
                
                alpha = min(1.,alpha + alpha_increment)
            if self.image_size[0] == 4:
                seed = tf.random.uniform([2, 512],-1.,1.)
                
                predictions = self.generator(seed)
                predictions = np.array(predictions)
                predictions = (predictions + 1) * 127.5
                predictions = predictions.astype(np.uint8)
            # print(predictions[0])
                fig = plt.figure(figsize=(4,4))

                for i in range(predictions.shape[0]):
                    plt.subplot(2, 2, i+1)
                    plt.imshow(predictions[i, :, :, :], cmap='gray_r')
                    plt.axis('off')

                plt.savefig('/home/ubuntu/bjh/Gan/PGGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            else:
                seed = tf.random.uniform([4, 512],-1.,1.)
                alpha_tensor = tf.constant(np.repeat(alpha,4).reshape(4,1),dtype=tf.float32)    
                predictions = self.generator([seed,alpha_tensor])
                predictions = np.array(predictions)
                predictions = (predictions + 1) * 127.5
                predictions = predictions.astype(np.uint8)
            # print(predictions[0])
                fig = plt.figure(figsize=(4,4))

                for i in range(predictions.shape[0]):
                    plt.subplot(2, 2, i+1)
                    plt.imshow(predictions[i, :, :, :], cmap='gray_r')
                    plt.axis('off')

                plt.savefig('/home/ubuntu/bjh/Gan/PGGAN/image/image_at_epoch_{:04d}.png'.format(epoch))
            
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
                self.generator.save_weights(os.path.join('/home/ubuntu/bjh/Gan/PGGAN/model', '{}x{}_generator.h5'.format(self.image_size[0], self.image_size[1])))
                self.discriminator.save_weights(os.path.join('/home/ubuntu/bjh/Gan/PGGAN/model', '{}x{}_discriminator.h5'.format(self.image_size[0], self.image_size[1])))
                alpha = 0
                previous_image_size = int(self.image_size[0])
                self.image_size = (previous_image_size * 2,previous_image_size * 2,3)
                print(self.image_size)
                if self.image_size[0] > self.target_size:
                    print('finish')
                    break
                print('creating {} model'.format(self.image_size))
                print(previous_image_size)
                self.build_model()
                self.generator.load_weights(os.path.join('/home/ubuntu/bjh/Gan/PGGAN/model', '{}x{}_generator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
                self.discriminator.load_weights(os.path.join('/home/ubuntu/bjh/Gan/PGGAN/model', '{}x{}_discriminator.h5'.format(previous_image_size, previous_image_size)), by_name=True)
                print(self.generator.summary())
                img_path = os.path.join('/home/ubuntu/bjh/Gan/archive/img_align_celeba/img_align_celeba')
                img_list = os.listdir(img_path)
                dataset = image_loader_celeba(img_path,(previous_image_size * 2,previous_image_size*2))
                dataset = image_processing(dataset)
                dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(len(dataset)).batch(4)
                alpha_increment = 1. / (30 / 2 * training_steps)
            print ('Time for epoch {} is {} sec d_loss {:.5f} g_loss {:.5f}'.format(epoch + 1, time.time()-start,d_loss,g_loss))

        seed = tf.random.uniform([4, self.z_dim],-1.,1.)
        predictions = self.generator(seed)

        fig = plt.figure(figsize=(4,4))

        for i in range(predictions.shape[0]):
            plt.subplot(2, 2, i+1)
            plt.imshow(predictions[i, :, :, :], cmap='gray_r')
            plt.axis('off')

            plt.savefig('/home/ubuntu/bjh/Gan/PGGAN/image/image_at_epoch_{:04d}.png'.format(epoch))

    def build_model(self):
        if self.image_size[0] == 4:
            self.generator = self.build_4_generator()
            self.discriminator = self.build_4_discriminator()
        elif self.image_size[0] == 8:
            self.generator = self.build_8_generator()
            self.discriminator = self.build_8_discriminator()
        elif self.image_size[0] == 16:
            self.generator = self.build_16_generator()
            self.discriminator = self.build_16_discriminator()
        elif self.image_size[0] == 32:
            self.generator = self.build_32_generator()
            self.discriminator = self.build_32_discriminator()
        elif self.image_size[0] == 64:
            self.generator = self.build_64_generator()
            self.discriminator = self.build_64_discriminator()
        elif self.image_size[0] == 128:
            self.generator = self.build_128_generator()
            self.discriminator = self.build_128_discriminator()
            self.generator.summary()
            self.discriminator.summary()
        elif self.image_size[0]== 256:
            self.generator = self.build_256_generator()
            self.discriminator = self.build_256_discriminator()        
        elif self.image_size[0]== 512:
            self.generator = self.build_512_generator()
            self.discriminator = self.build_512_discriminator()
            self.generator.summary()
            self.discriminator.summary()
        else:
            print('not target_size')

    def upsample(self,x,in_filters,filters,
                 kernel_size=3,strides=1,padding='valid',
                 activation=tf.nn.leaky_relu,name=''):
        upsample = UpSampling2D(size=2,interpolation='nearest')(x)
        # resdiual Block => x 
        upsample_x = EqualizeLearningRate(Conv2D(filters, kernel_size, strides, padding=padding,
                   kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(upsample)
        x = PixelNoramlization()(upsample_x)
        x = Activation(activation)(x)
        x = EqualizeLearningRate(Conv2D(filters, kernel_size, strides, padding=padding,
                   kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(upsample)
        x = PixelNoramlization()(x)
        x = Activation(activation)(x)
        return x , upsample 
    
    def downsample(self,x, filters1, filters2, 
                         kernel_size=3, strides=1, padding='valid', 
                         activation=tf.nn.leaky_relu, name=''):
  
        x = EqualizeLearningRate(Conv2D(filters1, kernel_size, strides, padding=padding,
               kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_1')(x)
        x = Activation(activation)(x)
        x = EqualizeLearningRate(Conv2D(filters2, kernel_size, strides, padding=padding,
               kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name=name+'_conv2d_2')(x)
        x = Activation(activation)(x)
        downsample = AveragePooling2D(pool_size=2)(x)

        return downsample

    def generator_input_block(self,x):
        x = EqualizeLearningRate(Dense(4*4*self.z_dim,kernel_initializer=self.kernel_initializer,
                                       bias_initializer='zero'),name='input')(x)
        x = PixelNoramlization()(x)
        x = LeakyReLU()(x)
        x = Reshape((4,4,self.z_dim))(x)
        x = EqualizeLearningRate(Conv2D(512,3,strides=1,padding='same',
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer='zeros'),name='input_conv2D')(x)
        x = PixelNoramlization()(x)
        x = LeakyReLU()(x)

        return x

    def build_4_generator(self):
        input = Input(shape=(self.z_dim))
        x = self.generator_input_block(input)
        alpha = Input((1),name='alpha')

        to_RGB = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(4,4))
        rgb_out = to_RGB(x)
        model = Model(inputs=[input,alpha],outputs=rgb_out)
        return model

    def build_8_generator(self):
        inputs = Input(self.z_dim)
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha') 

        ### block

        x, up_x = self.upsample(x, in_filters=512, filters=512, kernel_size=3, strides=1,
                                         padding='same', activation=tf.nn.leaky_relu, name='Up_{}x{}'.format(8, 8))
    
    
        previous_to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=self.output_activation,
                    kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(4, 4))
        to_rgb = EqualizeLearningRate(Conv2D(3, kernel_size=1, strides=1,  padding='same', activation=self.output_activation,
                    kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='to_rgb_{}x{}'.format(8, 8))

        l_x = to_rgb(x)
        r_x = previous_to_rgb(up_x)

        l_x = Multiply()([1 - alpha, l_x])

        r_x = Multiply()([alpha, r_x])
        combined = Add()([l_x, r_x])
    
        model = Model(inputs=[inputs, alpha], outputs=combined)
        return model                                     

    def build_16_generator(self):
        
        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, up_x = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
                                                                    
        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(8,8))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(16,16))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def build_32_generator(self):

        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
        x, up_x = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(32,32))  
                                                        
        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(16,16))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(32,32))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def build_64_generator(self):
        
        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(32,32))  
        x,up_x = self.upsample(x,in_filters=self.z_dim,filters=256,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(64,64))                                                             
        

        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(32,32))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(64,64))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def build_128_generator(self):
        
       # Initial block
        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(32,32))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=256,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(64,64))                         
        x,up_x = self.upsample(x,in_filters=256,filters=128,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(128,128))                                                             
        

        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(64,64))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(128,128))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def build_256_generator(self):
        
        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(32,32))  
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=256,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(64,64))
        x, _ = self.upsample(x,in_filters=256,filters=128,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(128,128))
        x, up_x = self.upsample(x,in_filters=128,filters=64,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(256,256))                                                                
        

        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(128,128))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(256,256))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def build_512_generator(self):
        
        inputs = Input(shape=(self.z_dim))
        x = self.generator_input_block(inputs)
        alpha = Input((1),name='alpha')

        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(8,8))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(16,16))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(32,32))  
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=256,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(64,64))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=128,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(128,128))
        x, _ = self.upsample(x,in_filters=self.z_dim,filters=64,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(256,256))                                                                
        x, up_x = self.upsample(x,in_filters=self.z_dim,filters=32,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='upsample_{}x{}'.format(512,512))

        previous = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(256,256))
        to_rgb = EqualizeLearningRate(Conv2D(3,kernel_size=1,strides=1,
                                             padding='same',activation = self.output_activation,
                                             kernel_initializer=self.kernel_initializer,
                                             bias_initializer = 'zeros'),name='to_RGB_{}_{}'.format(512,512))

        prev_x = to_rgb(x)
        res_x = previous(up_x)

        prev_x = Multiply()([1-alpha,prev_x])
        res_x = Multiply()([alpha,res_x])
        comb = Add()([prev_x,res_x])

        model = Model(inputs=[inputs,alpha],outputs=comb)
        return model

    def discriminator_output_block(self,x):
        x = MinibatchstdDev()(x)
        
        x = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=3,strides=1,
                                        padding='same',kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='output_conv2d_1')(x)
        x = LeakyReLU()(x)
        x = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=4,strides=1,
                                        padding='valid',kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='output_conv2d_2')(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        x = EqualizeLearningRate(Dense(1,kernel_initializer=self.kernel_initializer,bias_initializer='zeros'),name='output_dense')(x)

        return x
    
    def build_4_discriminator(self):
        inputs = Input((4,4,3))
        alpha = Input((1), name='input_alpha')
    
        from_rgb = EqualizeLearningRate(Conv2D(self.z_dim, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
        x = from_rgb(inputs)
        x = EqualizeLearningRate(Conv2D(self.z_dim, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='conv2d_up_channel')(x)
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model

    def build_8_discriminator(self):
        fade_in_channel = 512
        inputs = Input((8,8,3))
        alpha = Input((1), name='input_alpha')
        downsample = AveragePooling2D(pool_size=2)
        previous_from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(4, 4))
        l_x = previous_from_rgb(downsample(inputs))
        l_x = Multiply()([1 - alpha, l_x])
        from_rgb = EqualizeLearningRate(Conv2D(512, kernel_size=1, strides=1, padding='same', activation=tf.nn.leaky_relu,
                      kernel_initializer=self.kernel_initializer, bias_initializer='zeros'), name='from_rgb_{}x{}'.format(8, 8))
        r_x = from_rgb(inputs)
        r_x = self.downsample(r_x, filters1=512, filters2=fade_in_channel, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        r_x = Multiply()([alpha, r_x])
        x = Add()([l_x, r_x])
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model

    def build_16_discriminator(self):
        inputs = Input((16,16,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(8,8))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(16,16))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=self.z_dim,filters2=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(512,512))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model   
        return model
    
    def build_32_discriminator(self):
        inputs = Input((32,32,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(16,16))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(32,32))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=self.z_dim,filters2=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(512,512))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model   

    def build_64_discriminator(self):
        inputs = Input((64,64,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(self.z_dim,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(32,32))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(256,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(64,64))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=256,filters2=self.z_dim,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(64,64))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model   

    def build_128_discriminator(self):
        inputs = Input((128,128,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(256,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(64,64))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(128,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(128,128))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=128,filters2=256,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(128,128))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])                                
        x = self.downsample(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
      
        return model

    def build_256_discriminator(self):
        inputs = Input((256,256,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(128,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(128,128))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(64,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(256,256))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=64,filters2=128,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(256,256))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])

        x = self.downsample(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))                                    
        x = self.downsample(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model   

    def build_512_discriminator(self):
        inputs = Input((512,512,3))
        alpha = Input((1), name='input_alpha')
        down = AveragePooling2D(pool_size=2)
        previous_rgb = EqualizeLearningRate(Conv2D(64,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(256,256))
        prev_x = previous_rgb(down(inputs))
        prev_x = Multiply()([1-alpha,prev_x])
        from_rgb = EqualizeLearningRate(Conv2D(32,kernel_size=1,strides=1,
                                        padding='same',activation=tf.nn.leaky_relu,
                                        kernel_initializer=self.kernel_initializer,
                                        bias_initializer = 'zeros'),name='from_rgb_{}x{}'.format(512,512))
        after_x = from_rgb(inputs)
        after_x = self.downsample(after_x,filters1=32,filters2=64,
                             kernel_size=3,strides=1,padding='same',activation=tf.nn.leaky_relu,
                             name='Down_{}x{}'.format(512,512))
        after_x = Multiply()([alpha,after_x])
        x = Add()([prev_x,after_x])
        x = self.downsample(x, filters1=64, filters2=128, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(256,256))
        x = self.downsample(x, filters1=128, filters2=256, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(128,128))                                    
        x = self.downsample(x, filters1=256, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(64,64))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(32,32))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(16,16))
        x = self.downsample(x, filters1=512, filters2=512, kernel_size=3, strides=1,
                                            padding='same', activation=tf.nn.leaky_relu, name='Down_{}x{}'.format(8,8))
        x = self.discriminator_output_block(x)
        model = Model(inputs=[inputs, alpha], outputs=x)
        return model             


