import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense,LeakyReLU,Conv2D,Reshape,Conv2DTranspose,Lambda,BatchNormalization,Dropout,ReLU,Concatenate,UpSampling2D,Add,Activation
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Layer,InputSpec
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(1)
class ReflectionPadding2D(Layer):
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


## img load

img_path = os.path.join('./Monet2Photh/')
img_list = os.listdir(img_path)
print(img_list)

data_A = []
data_B = []
arr = ['trainA','trainB']
for i in range(len(arr)):
    path = img_path + arr[i]
    img_dir = os.listdir(path)
    for j in range(len(img_dir)):
        img = cv2.imread(path + "/" + img_dir[j])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(256,256),interpolation = cv2.INTER_CUBIC)
        if i == 0:
            data_A.append(img)
        else:
            data_B.append(img)

data_A = np.array(data_A)
data_B = np.array(data_B)
data_A = data_A / 255.0
data_B = data_B / 255.0

print(data_A.shape, data_B.shape)

## make A,B, G_AB, G_BA

def build_descriminator(input_shape = (256,256,3)):
    input = Input(shape=input_shape)
    x = input
    stride = [2,2,2,1]
    filter = [32 * 2,64 * 2,128 * 2,256 * 2]
    for i in range(len(stride)):
        if i == 0:
            x = Conv2D(filters=filter[i],strides=stride[i],kernel_size=4, padding='same')(x)
            x = LeakyReLU(0.2)(x)
        else:
            x = Conv2D(filters=filter[i],strides=stride[i],kernel_size=4, padding='same')(x)
            x = InstanceNormalization(axis = -1, center = False, scale = False)(x)
            x = LeakyReLU(0.2)(x)
    output = Conv2D(filters=1,kernel_size=4,strides=1,padding='same')(x)
    model = Model(input,output)
    return model            

def build_generator_resnet(input_shape = (256,256,3)):

    def conv7s1(layer_input, filters, final):
        y = ReflectionPadding2D(padding =(3,3))(layer_input)
        y = Conv2D(filters, kernel_size=(7,7), strides=1, padding='valid')(y)
        if final:
            y = Activation('tanh')(y)
        else:
            y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
            y = Activation('relu')(y)
        return y

    def downsample(layer_input,filters):
        y = Conv2D(filters, kernel_size=(3,3), strides=2, padding='same')(layer_input)
        y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
        y = Activation('relu')(y)
        return y

    def residual(layer_input, filters):
        shortcut = layer_input
        y = ReflectionPadding2D(padding =(1,1))(layer_input)
        y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid')(y)
        y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
        y = Activation('relu')(y)
            
        y = ReflectionPadding2D(padding =(1,1))(y)
        y = Conv2D(filters, kernel_size=(3, 3), strides=1, padding='valid')(y)
        y = InstanceNormalization(axis = -1, center = False, scale = False)(y)

        return tf.add(shortcut, y)

    def upsample(layer_input,filters):
        y = Conv2DTranspose(filters, kernel_size=(3, 3), strides=2, padding='same')(layer_input)
        y = InstanceNormalization(axis = -1, center = False, scale = False)(y)
        y = Activation('relu')(y)
    
        return y

    img = Input(shape=input_shape)

    y = img
    gen_n_filters = 32
    y = conv7s1(y, gen_n_filters, False)
    y = downsample(y, gen_n_filters * 2)
    y = downsample(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = residual(y, gen_n_filters * 4)
    y = upsample(y, gen_n_filters * 2)
    y = upsample(y, gen_n_filters)
    y = conv7s1(y, 3, True)

    output = y

   
    return Model(img, output)


# Compile model

d_A = build_descriminator()
d_B = build_descriminator()
d_A.compile(loss = 'mse',optimizer = tf.keras.optimizers.Adam(0.0002,0.5),metrics=['acc'])
d_B.compile(loss = 'mse',optimizer = tf.keras.optimizers.Adam(0.0002,0.5),metrics=['acc'])

# generator G_AB, G_BA
# A -> B,  B -> A 
# apple => orange, orange => apple

g_AB = build_generator_resnet()
g_BA = build_generator_resnet()

d_A.trainable = False
d_B.trainable = False

#A = apple, B = orange
img_A = Input(shape=(256,256,3))
img_B = Input(shape=(256,256,3))

fake_A = g_BA(img_B)
fake_B = g_AB(img_A)

valid_A = d_A(fake_A)
valid_B = d_B(fake_B)

reconstruct_A = g_BA(fake_B)
reconstruct_B = g_AB(fake_A)

img_A_id = g_BA(img_A)
img_B_id = g_AB(img_B)

combined = Model(inputs=[img_A,img_B],outputs=[valid_A,valid_B,reconstruct_A,reconstruct_B,img_A_id,img_B_id])

combined.compile(loss=['mse','mse','mae','mae','mae','mae'],
                 loss_weights=[1,1,10,10,5,5],
                 optimizer=tf.keras.optimizers.Adam(0.0002,0.5))
d_A.trainable = True
d_B.trainable = True

def train_discriminators(imgs_A, imgs_B, valid, fake):

        # Translate images to opposite domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)

        # Train the discriminators (original images = real / translated = Fake)
        dA_loss_real = d_A.train_on_batch(imgs_A, valid)
        dA_loss_fake = d_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = d_B.train_on_batch(imgs_B, valid)
        dB_loss_fake = d_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss_total = 0.5 * np.add(dA_loss, dB_loss)

        return (
            d_loss_total[0]
            , dA_loss[0], dA_loss_real[0], dA_loss_fake[0]
            , dB_loss[0], dB_loss_real[0], dB_loss_fake[0]
            , d_loss_total[1]
            , dA_loss[1], dA_loss_real[1], dA_loss_fake[1]
            , dB_loss[1], dB_loss_real[1], dB_loss_fake[1]
        )

def train_generators(imgs_A, imgs_B, valid):

    return combined.train_on_batch([imgs_A, imgs_B],
                                                [valid, valid,
                                                imgs_A, imgs_B,
                                                imgs_A, imgs_B])

def train_normal(batch_size,img_row,epochs,print_every_n_batches = 10):
    
    d_lossess = []
    g_lossess = []
    patch = int(img_row / 2 ** 3)
    disc_patch = (patch,patch,1)
    
    valid = np.ones((batch_size,) + disc_patch)
    fake = np.zeros((batch_size,) + disc_patch)

    epoch = 0

    for epoch in range(epoch,epochs):
        for i in range(1000):
            data_batch_A = data_A[i:(i+1)]
            data_batch_B = data_B[i:(i+1)]
            data_batch_A = data_batch_A.reshape((1,256,256,3))
            data_batch_B = data_batch_B.reshape((1,256,256,3))
            d_loss = train_discriminators(data_batch_A, data_batch_B, valid, fake)
            g_loss = train_generators(data_batch_A, data_batch_B, valid)

            if i % 100 == 0:
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] "\
                        % ( epoch, epochs,
                            i, min(len(data_A),len(data_B)),
                            d_loss[0], 100*d_loss[7],
                            g_loss[0],
                            np.sum(g_loss[1:3]),
                            np.sum(g_loss[3:5]),
                            np.sum(g_loss[5:7]),
                            ))
                sample_images(i)
            d_lossess.append(d_loss)
            g_lossess.append(g_loss)     
        
    return d_lossess,g_lossess

def sample_images(batch_i):
        
    r, c = 2, 4
    arr = ['trainA', 'testB', 'trainB', 'testA']
    for p in range(2):

        if p == 1:
            path_A = img_path + arr[0]
            path_B = img_path + arr[2]
            img_list_A = os.listdir(path_A)
            img_list_B = os.listdir(path_B)
            imgs_A = cv2.imread(path_A + "/" + img_list_A[np.random.randint(0,len(img_list_A))])
            imgs_A = cv2.cvtColor(imgs_A,cv2.COLOR_BGR2RGB)
            imgs_A = cv2.resize(imgs_A,(256,256),interpolation = cv2.INTER_CUBIC)
            imgs_B = cv2.imread(path_B + "/" + img_list_B[np.random.randint(0,len(img_list_B))])
            imgs_B = cv2.cvtColor(imgs_B,cv2.COLOR_BGR2RGB)
            imgs_B = cv2.resize(imgs_B,(256,256),interpolation = cv2.INTER_CUBIC)
        else:
            path_A = img_path + arr[0]
            path_B = img_path + arr[2]
            img_list_A = os.listdir(path_A)
            img_list_B = os.listdir(path_B)
            imgs_A = cv2.imread(path_A + "/" + img_list_A[np.random.randint(0,len(img_list_A))])
            imgs_A = cv2.cvtColor(imgs_A,cv2.COLOR_BGR2RGB)
            imgs_A = cv2.resize(imgs_A,(256,256),interpolation = cv2.INTER_CUBIC)
            imgs_B = cv2.imread(path_B + "/" + img_list_B[np.random.randint(0,len(img_list_B))])
            imgs_B = cv2.cvtColor(imgs_B,cv2.COLOR_BGR2RGB)
            imgs_B = cv2.resize(imgs_B,(256,256),interpolation = cv2.INTER_CUBIC)

        imgs_A = imgs_A.reshape((1,256,256,3))
        imgs_B = imgs_B.reshape((1,256,256,3))

        imgs_A = imgs_A / 255.0
        imgs_B = imgs_B / 255.0
            # Translate images to the other domain
        fake_B = g_AB.predict(imgs_A)
        fake_A = g_BA.predict(imgs_B)
            # Translate back to original domain
        reconstr_A = g_BA.predict(fake_B)
        reconstr_B = g_AB.predict(fake_A)

            # ID the images
        id_A = g_BA.predict(imgs_A)
        id_B = g_AB.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, id_A, imgs_B, fake_A, reconstr_B, id_B])

            # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        gen_imgs = np.clip(gen_imgs, 0, 1)

        titles = ['Original', 'Translated', 'Reconstructed', 'ID']
        fig, axs = plt.subplots(r, c, figsize=(25,12.5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt],cmap='gray_r')
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        plt.show()        
        plt.close()

with tf.device('/GPU:1'):
    d_losses,g_losses = train_normal(1,256,3)
    