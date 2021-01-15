import tensorflow as tf
import numpy as np

class PixelNoramlization(tf.keras.layers.Layer):

    def __init__(self,epsilon=1e-8):
        super(PixelNoramlization,self).__init__()
        self.epsilon = epsilon

    def call(self,inputs):
        return inputs/ tf.sqrt(tf.reduce_mean(tf.square(inputs),axis=-1,keepdims=True) + self.epsilon)   

    def compute_output_shape(self,input_shape):
        return input_shape     