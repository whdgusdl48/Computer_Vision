from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10,mnist
from data_loader import load_data,load_data2
from model import AlexNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, MaxPooling2D,Flatten,Dropout
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

def Net():
        input = Input(shape=(112,112,3))
        x = input
        x = Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')(x)
        x = Conv2D(filters=64,kernel_size=3,strides=1,padding='same',activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu')(x)
        x = Conv2D(filters=128,kernel_size=3,strides=1,padding='same',activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_3_1')(x)
        x = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_3_2')(x)
        x = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_3_3')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_4_1')(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_4_2')(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_4_3')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_5_1')(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_5_2')(x)
        x = Conv2D(filters=512,kernel_size=3,strides=1,padding='same',activation='relu',name='conv_layer_5_3')(x)
        x = MaxPooling2D(2,2)(x)
        x = Flatten()(x)
        x = Dense(1024,activation='relu')(x)
        x = Dense(1024,activation='relu')(x)
        
        x = Dense(6,activation='softmax')(x)

        model = Model(input,x)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics = ['acc'])
        return model

model = Net()
model.fit(train_data,epochs=10)