import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Dense, BatchNormalization,Input,ReLU,UpSampling2D,InputSpec, Add,Lambda
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import time

def attention_layer(x,channels):
        
        f = Conv2D(filters=channels//8,kernel_size=1,strides=1,padding='same')(x)
        g = Conv2D(filters=channels//8,kernel_size=1,strides=1,padding='same')(x)
        h = Conv2D(filters=channels,kernel_size=1,strides=1,padding='same')(x)

        f_shape = K.int_shape(f)[1:]
        g_shape = K.int_shape(g)[1:]
        h_shape = K.int_shape(h)[1:]

        f = Reshape((f_shape[0] * f_shape[1], f_shape[2]))(f)
        g = Reshape((g_shape[0] * g_shape[1], g_shape[2]))(g)
        h = Reshape((h_shape[0] * h_shape[1], h_shape[2]))(h)
        print(f.shape,g.shape,h.shape)
        s = tf.matmul(f,g,transpose_b = True)
        print(s.shape)
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta,h)
        gamma = Input(shape=(1,))
        print(x.shape)
        o = Reshape(x.shape[1:])(o)

        o = Conv2D(channels,kernel_size=1,strides=1)(o)

        l = gamma * o 
        print(l.shape)
        x = Add()([l,x])
        return x

a = Input(shape=(128,128,128))

print(attention_layer(a,128))