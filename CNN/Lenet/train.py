from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from data_loader import load_cifar, image_processing
from model import Lenet_5
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

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.reshape(len(y_train),)
y_test = y_test.reshape(len(y_test),)
# y_train = tf.one_hot(y_train,10,axis=-1)
# y_test = tf.one_hot(y_test,10,axis=-1)
train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(256)
test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(256)

gan = Lenet_5(256,100,(32,32,3))

# def Net(shape):
#         input = Input(shape=shape)
#         x = input
#         x = Conv2D(filters=32,strides=1,kernel_size=5,activation='relu')(x)
#         x = MaxPooling2D(2,2)(x)
#         x = Conv2D(filters=64,strides=1,kernel_size=5,activation='relu')(x)
#         x = Flatten()(x)
#         x = Dense(120,activation='relu')(x)
#         x = Dense(84,activation='relu')(x)
#         x = Dense(10,activation='softmax')(x)

#         model = Model(input,x)
       
#         return model

gan.train(train_data,test_data)

# model = Net((32,32,3))

# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['acc'])

# model.fit(x_train,y_train,epochs=15,batch_size=256,validation_data=(x_test,y_test))

