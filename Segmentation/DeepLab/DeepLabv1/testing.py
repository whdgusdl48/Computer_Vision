from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf
from tensorflow.keras.applications import vgg16

def deeplabv1(input_shape=(256,256,3),filters=64):
    input = Input(shape=input_shape)
    x = input

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=x,
        pooling=None,
        classes=1000)
    # model.summary()
    a = model.get_layer('block4_conv2').output
    a = MaxPool2D(1,1)(a)
    a = Conv2D(1024,kernel_size=3,strides=1,dilation_rate=2,padding='same',activation='relu')(a)
    a = Conv2D(1024,kernel_size=3,strides=1,dilation_rate=2,padding='same',activation='relu')(a)
    a = Conv2D(1024,kernel_size=3,strides=1,dilation_rate=2,padding='same',activation='relu')(a)
    a = MaxPool2D(1,1)(a)
    a = Conv2D(4096,kernel_size=7, strides=1, padding='same',dilation_rate=4,activation='relu')(a)
    a = Conv2D(4096,kernel_size=3, strides=1, padding='same',activation='relu')(a)
    a = Conv2D(23,kernel_size=3, strides=1, padding='same',activation='softmax')(a)
    a = UpSampling2D((8,8),interpolation='bilinear')(a)
    test = Model(inputs = input, outputs=a)
    return test
