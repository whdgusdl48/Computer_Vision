import tensorflow as tf
import numpy as np

class MinibatchstdDev(tf.keras.layers.Layer):
    
    def __init__(self,group_size=4):
        super(MinibatchstdDev,self).__init__()
        self.group_size = group_size


    def call(self,inputs):
        group_size = tf.minimum(self.group_size,tf.shape(inputs)[0])

        shape = inputs.shape
        y = tf.reshape(inputs,[group_size,-1,shape[1],shape[2],shape[3]])
        y = tf.cast(y,tf.float32)
        y -= tf.reduce_mean(y,axis=0,keepdims=True)
        y = tf.reduce_mean(tf.square(y),axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y,axis=[1,2,3],keepdims=True)
        y = tf.cast(y,inputs.dtype)
        y = tf.tile(y,[group_size,shape[1],shape[2],1])
        return tf.concat([inputs,y],axis=-1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[1],input_shape[2],input_shape[3] + 1)

        