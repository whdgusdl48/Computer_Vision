# tomorrow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

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

def fcn32(input_shape=(256,256,3),filter=64):
    input = Input(shape=input_shape)
    x = input
    conv1_1 = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    conv1_2 = Conv2D(filters=filter, kernel_size=3, strides=1, padding='same', activation='relu')(conv1_1)
    max_pool1 = MaxPool2D(2,2)(conv1_2)

    conv2_1 = Conv2D(filters=filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool1)
    conv2_2 = Conv2D(filters=filter * 2, kernel_size=3, strides=1, padding='same', activation='relu')(conv2_1)
    max_pool2 = MaxPool2D(2,2)(conv2_2)

    conv3_1 = Conv2D(filters=filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool2)
    conv3_2 = Conv2D(filters=filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(conv3_1)
    conv3_3 = Conv2D(filters=filter * 4, kernel_size=3, strides=1, padding='same', activation='relu')(conv3_2)
    max_pool3 = MaxPool2D(2,2)(conv3_3)

    conv4_1 = Conv2D(filters=filter * 8, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool3)
    conv4_2 = Conv2D(filters=filter * 8, kernel_size=3, strides=1, padding='same', activation='relu')(conv4_1)
    conv4_3 = Conv2D(filters=filter * 8, kernel_size=3, strides=1, padding='same', activation='relu')(conv4_2)
    max_pool4 = MaxPool2D(2,2)(conv4_3)

    conv5_1 = Conv2D(filters=filter * 16, kernel_size=3, strides=1, padding='same', activation='relu')(max_pool4)
    conv5_2 = Conv2D(filters=filter * 16, kernel_size=3, strides=1, padding='same', activation='relu')(conv5_1)
    conv5_3 = Conv2D(filters=filter * 16, kernel_size=3, strides=1, padding='same', activation='relu')(conv5_2)
    max_pool5 = MaxPool2D(2,2)(conv5_3)

    conv6_1 = Conv2D(filters= filter*64, kernel_size=7, strides=1, padding='same', activation='relu')(max_pool5)
    conv6_2 = Conv2D(filters= filter*64, kernel_size=1, strides=1, padding='same', activation='relu')(conv6_1)
    conv6_3 = Conv2D(filters= 23, kernel_size=1, strides=1, padding='same', activation='softmax')(conv6_2)

    output = UpSampling2D(size=(32,32))(conv6_3)

    model = Model(input,output)

    return model

a = fcn32()
a.summary()