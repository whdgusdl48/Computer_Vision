from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10,mnist
from data_loader import load_data
from model import AlexNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential, Model

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

train_data, test_data = load_data()

model = AlexNet(32,(224,224,3),100,0.001)

model.train_fit(train_data,test_data)