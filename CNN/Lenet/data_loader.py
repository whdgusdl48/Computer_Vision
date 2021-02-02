import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np
import cv2

def load_cifar(image_size):
    data = []
    test_data = []
    (x_train, y_train), (x_test,y_test) = cifar10.load_data()
    y_train = y_train.reshape(len(y_train),)
    y_test = y_test.reshape(len(y_test),)
    y_train = tf.one_hot(y_train,10,axis=-1)
    y_test = tf.one_hot(y_test,10,axis=-1)
    for i in range(len(x_train)):
        img = cv2.resize(x_train[i,:,:,:],(image_size,image_size),interpolation=cv2.INTER_CUBIC)
        data.append(img)
    for i in range(len(x_test)):
        img = cv2.resize(x_test[i,:,:,:],(image_size,image_size),interpolation=cv2.INTER_CUBIC)
        test_data.append(img)    
    data = np.array(data)
    test_data = np.array(test_data)
    data = tf.cast(data,tf.float32)
    data = (data - 127.5) / 127.5
    test_data = tf.cast(test_data,tf.float32)
    test_data = (test_data - 127.5) / 127.5
    print(data.shape,test_data.shape,y_train.shape)

    return (data,y_train), (test_data,y_test)

def image_processing(dataset):
    data = tf.cast(dataset,tf.float32)
    data = (data - 127.5) / 127.5
    return data

