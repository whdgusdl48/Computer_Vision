from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10,mnist
from data_loader import load_data,load_data2
# from model import AlexNet
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, MaxPooling2D,Flatten,Dropout,Add,Activation,BatchNormalization,ZeroPadding2D
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
print(x_train.shape)
data = []
for i in range(10000):
  img = tf.constant(x_train[i,:,:,:])
  img = tf.image.resize(img,(150,150))
  data.append(img)
data = np.array(data)
x_train = data[:9000]

y_train = y_train[:10000]
y_train = y_train.reshape(-1,)
y_train = tf.one_hot(y_train,10)
y_train2 = y_train[:9000]
x_test = data[9000:]
y_test = y_train[9000:]
x_train = x_train / 255.
x_test = x_test / 255.
# train_data = tf.data.Dataset.from_tensor_slices((x_train,y_train2)).shuffle(200).batch(4)
# test_data = tf.data.Dataset.from_tensor_slices((x_test,y_test)).shuffle(len(x_test)).batch(4)
print(data.shape,y_test.shape)
def residual(layers,filters,kernel,strides,last_strides,first=False):
  x = layers

  x = Conv2D(filters=filters[0],kernel_size=kernel[0],strides=strides[0],padding='valid')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  x = Conv2D(filters=filters[1],kernel_size=kernel[1],strides=strides[1],padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  
  if first:
    x = Conv2D(filters=filters[2],kernel_size=kernel[2],strides=strides[2],padding='valid')(x)
    layers = Conv2D(filters=filters[2],strides=last_strides,kernel_size=kernel[2],padding='valid')(layers)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    add_x = Add()([x,layers])
    add_x = Activation('relu')(add_x)
    return add_x
  
  else:
    x = Conv2D(filters=filters[2],kernel_size=kernel[2],strides=strides[2],padding='valid')(x)
    x = BatchNormalization()(x)
    add_x = Add()([x,layers])
    add_x = Activation('relu')(add_x)
    return add_x

  

def net(input_shape):
    input = Input(shape=input_shape)
    x = input
    x = Conv2D(64, (7, 7), strides=(2, 2),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(2,2)(x)
    x = residual(x,[64,64,256],[1,3,1],[1,1,1],1,first=True)
    x = residual(x,[64,64,256],[1,3,1],[1,1,1],1)
    x = residual(x,[64,64,256],[1,3,1],[1,1,1],1)
    x = residual(x,[128,128,512],[1,3,1],[2,1,1],2,first=True)
    x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
    x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
    x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
    x = residual(x,[256,256,1024],[1,3,1],[2,1,1],2,first=True)
    x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
    x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
    x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
    x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
    x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
    x = residual(x,[512,512,2048],[1,3,1],[2,1,1],2,first=True)
    x = residual(x,[512,512,2048],[1,3,1],[1,1,1],2)
    x = residual(x,[512,512,2048],[1,3,1],[1,1,1],2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = Dense(10,activation='softmax')(x)
    return Model(input,output)
model = net((150,150,3))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.0001,beta_1=0.9),metrics=['acc'])
model.fit(x_train,y_train2,batch_size=4,epochs=30,validation_data=(x_test,y_test))