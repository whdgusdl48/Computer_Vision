import os
import matplotlib.pyplot as plt
import tensorflow as tf
from make_data import DIV2K
from SRGAN import SRGAN
from PIL import Image
import cv2
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

div2k_train = DIV2K(scale=4, subset='train', downgrade='bicubic')
div2k_valid = DIV2K(scale=4, subset='valid', downgrade='bicubic')
weights_dir = '/home/ubuntu/bjh/Gan/SRGAN/Weights'
weights_file = lambda filename: os.path.join(weights_dir, filename)
# first
train_ds = div2k_train.dataset(batch_size=16, random_transform=True)
valid_ds = div2k_valid.dataset(batch_size=16, random_transform=True, repeat_count=1)

print(train_ds)

SRGAN = SRGAN(24,96)

# SRGAN.train(train_ds)

# SRGAN.G.save_weights(weights_file('gan_generator.h5'))
# SRGAN.D.save_weights(weights_file('gan_discriminator.h5'))

test = SRGAN.G
test.load_weights(weights_file('gan_generator.h5'))
test.summary()
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def resolve_and_plot(lr_image_path,lr_image_path2):
    # lr = load_image(lr_image_path)
    lr2 = load_image(lr_image_path2)
    lr2 = cv2.resize(lr2,(300,350))
    # print(lr.shape)
    # gan_sr = resolve_single(test, lr)

    gan_sr2 = resolve_single(test, lr2)
    tf.keras.preprocessing.image.save_img('/home/ubuntu/bjh/Gan/SRGAN/result/' + 'gan_img5.png',gan_sr2)    
    
    print(gan_sr2.shape)
    # plt.figure(figsize=(20, 20))
    
    # images = [lr ,gan_sr,lr2,gan_sr2]
    # titles = ['LR','SR (GAN)','LR2','SRGAN2']
    # positions = [1,2,3,4]
    
    # for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
    #     plt.subplot(2, 2, pos)
    #     plt.imshow(img)
    #     plt.title(title)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.savefig('/home/ubuntu/bjh/Gan/SRGAN/result/' + 'test2.png')

resolve_and_plot('/home/ubuntu/bjh/Gan/div2k/images/DIV2K_train_LR_bicubic/X4/0024x4.png',
'/home/ubuntu/bjh/Gan/div2k/images/DIV2K_valid_LR_bicubic/X4/0855x4.png')