from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import image_loader_celeba, image_processing
from model import PGGAN
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

img_path = os.path.join('/home/ubuntu/bjh/Gan/archive/img_align_celeba/img_align_celeba')
img_list = os.listdir(img_path)

IMAGE_SIZE = 4
data = image_loader_celeba(img_path,(IMAGE_SIZE,IMAGE_SIZE))
data = image_processing(data)
data = tf.data.Dataset.from_tensor_slices(data).shuffle(len(data)).batch(4)

gan = PGGAN(batch_size=4,image_size=(4,4,3),epochs=500,target_size=512,checkpoint_dir = 'PGGAN/PGGAN',)
gan.train(data)