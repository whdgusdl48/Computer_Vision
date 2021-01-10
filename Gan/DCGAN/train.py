from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import image_loader,image_processing,split_data,load_batch,image_loader_celeba
from model import DCGAN
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

# ---- image load and processing -------

img_path = os.path.join('/home/ubuntu/bjh/Gan/archive/img_align_celeba/img_align_celeba')
img_list = os.listdir(img_path)

IMAGE_SIZE = 64
data = image_loader_celeba(img_path,(IMAGE_SIZE,IMAGE_SIZE))

# x_train, y_train = train_data[0], train_data[1]
# x_test, y_test = test_data[0], test_data[1]

# x_train = image_processing(x_train)
# x_test = image_processing(x_test)
BATCH_SIZE = 64

data = image_processing(data)

data = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(16)
# x_test = tf.data.Dataset.from_tensor_slices(x_test).shuffle(len(x_test)).batch(BATCH_SIZE)

gan = DCGAN(input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
            generator_filters = [512,256,128,128,3],
            discriminator_filters = [64,128,256,512,1024],
            generator_strides =[2,2,2,2,1],
            discriminator_strides = [2,2,2,2,1],
            checkpoint_dir = 'DCGAN/DCGAN',
            batch_size = 16
            )
# print(gan.generator.summary())
with tf.device('/GPU:0'):
  gan.train(data)