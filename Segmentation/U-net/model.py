from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2

import tensorflow as tf

def get_unet(weight = None,input_shape = (256,256,3),initial_filter = 64):
    inputs = Input(input_shape)
    conv1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPool2D(pool_size = (2,2))(conv1_1)

    conv2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPool2D(pool_size = (2,2))(conv2_1)

    conv3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPool2D(pool_size = (2,2))(conv3_1)

    conv4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4_1 = Dropout(0.5)(conv4_1)
    pool4 = MaxPool2D(pool_size = (2,2))(conv4_1)

    ####################################downsampling#############################################

    conv5 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5_1 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5_1 = Dropout(0.5)(conv5_1)
    up5 = UpSampling2D(size = (2,2))(conv5_1)

    ####################################upsampling#############################################

    merge_4 = concatenate([conv4_1,up5], axis = 3)
    conv_4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_4)
    conv_4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_4)
    up4 = UpSampling2D(size = (2,2))(conv_4_1)

    merge_3 = concatenate([conv3_1,up4], axis = 3)
    conv_3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_3)
    conv_3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_3)
    up3 = UpSampling2D(size = (2,2))(conv_3_1)

    merge_2 = concatenate([conv2_1,up3], axis = 3)
    conv_2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_2)
    conv_2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2)
    up2 = UpSampling2D(size = (2,2))(conv_2_1)

    merge_1 = concatenate([conv1_1,up2], axis = 3)
    conv_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1)
    conv_1_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)

    out = Conv2D(23, (1,1), activation = 'softmax')(conv_1_1)

    model = Model(inputs,out)
    #model = multi_gpu_model(model,gpus =2)
    model.compile(optimizer = Adam(lr = 3e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])


    return model