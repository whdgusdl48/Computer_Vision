# tomorrow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

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

    conv6 = Conv2D(filters= filter*32, kernel_size=7, strides=1, padding='same', activation='relu')(max_pool5)
    drop_out_1 = Dropout(0.05)(conv6)

    conv7 = Conv2D(filters= filter*32, kernel_size=1, strides=1, padding='same', activation='relu')(drop_out_1)
    drop_out_2 = Dropout(0.05)(conv7)

    classes = Conv2D(filters=23, kernel_size=1, strides=1, padding='same',activation='softmax')(drop_out_2)
    output = UpSampling2D((32,32))(classes)


    model = Model(input,output)

    return model

def fcn16(input_shape=(256,256,3),filter=64):
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

    conv6 = Conv2D(filters= filter*32, kernel_size=7, strides=1, padding='same', activation='relu')(max_pool5)
    drop_out_1 = Dropout(0.05)(conv6)

    conv7 = Conv2D(filters= filter*32, kernel_size=1, strides=1, padding='same', activation='relu')(drop_out_1)
    drop_out_2 = Dropout(0.05)(conv7)


    first = Conv2D(23,kernel_size=1,strides=1,padding='same',activation='relu')(max_pool4)

    upsample = Conv2DTranspose(23,kernel_size=4,strides=2,padding='same',activation='relu')(drop_out_2)

    add = Add()([first,upsample])

    output = Conv2DTranspose(23,kernel_size=3, strides=16,padding='same',activation='relu')(add)

    model = Model(input,output)

    return model

def fcn8(input_shape=(256,256,3),filter=64):
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

    conv6 = Conv2D(filters= filter*32, kernel_size=7, strides=1, padding='same', activation='relu')(max_pool5)
    drop_out_1 = Dropout(0.5)(conv6)

    conv7 = Conv2D(filters= filter*32, kernel_size=1, strides=1, padding='same', activation='relu')(drop_out_1)
    drop_out_2 = Dropout(0.5)(conv7)

    last = Conv2D(23,kernel_size=1,strides=1,padding='same',activation='relu')(drop_out_2)

    last = Conv2DTranspose(23,kernel_size=2,strides=2, padding='valid')(last)

    skip_con1 = Conv2D(23, kernel_size=1, padding='same')(max_pool4)

    sum = Add()([last,skip_con1])

    up_1 = Conv2DTranspose(23,kernel_size=2,strides=2,padding='valid')(sum)

    skip_con2 = Conv2D(23, kernel_size=1, padding='same')(max_pool3)

    sum2 = Add()([skip_con2,sum])

    Up = Conv2DTranspose(23,kernel_size=8,strides=8,padding='valid',activation='softmax')(sum2)

    model = Model(input,Up)
    return model
