from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import load_ffhq, image_processing
from model import SAGAN

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
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

img_path = os.path.join('/home/ubuntu/bjh/Gan/thumbnails128x128-20210121T043548Z-002/thumbnails128x128')
img_list = os.listdir(img_path)

IMAGE_SIZE = 128
data = load_ffhq(img_path,(IMAGE_SIZE,IMAGE_SIZE))
data = image_processing(data)
data = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(4)

gan = SAGAN(generator_layer_num = 6,
                 discriminator_layer_num = 6,
                 image_size =(128,128,3),
                 batch_size = 4,
                 checkpoint_dir = 'SAGAN/SAGAN',
                 epochs=500)
with tf.device('/GPU:0'):
  gan.train(data)