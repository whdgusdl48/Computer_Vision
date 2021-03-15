from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

def DeepLabv2(input_shape=(256,256,3),upsampling=2,softmax = True,filters=64):

    input = Input(shape=input_shape)

    x = input
    # block 1
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
    # block 2
    x = Conv2D(filters=filters * 2, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 2, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
    # block 3
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 4, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
    # block 4
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
    # block 5
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = Conv2D(filters=filters * 8, kernel_size=3, strides=1, padding='same',activation='relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    p5 = MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)

    b1 = ZeroPadding2D(padding=(6, 6))(p5)
    b1 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(6, 6), activation='relu', name='fc6_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_1')(b1)
    b1 = Dropout(0.5)(b1)
    b1 = Conv2D(filters=23, kernel_size=(1, 1), activation='relu', name='fc8_voc12_1')(b1)

    b2 = ZeroPadding2D(padding=(12, 12))(p5)
    b2 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(12, 12), activation='relu', name='fc6_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_2')(b2)
    b2 = Dropout(0.5)(b2)
    b2 = Conv2D(filters=23, kernel_size=(1, 1), activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = ZeroPadding2D(padding=(18, 18))(p5)
    b3 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(18, 18), activation='relu', name='fc6_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_3')(b3)
    b3 = Dropout(0.5)(b3)
    b3 = Conv2D(filters=23, kernel_size=(1, 1), activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = ZeroPadding2D(padding=(24, 24))(p5)
    b4 = Conv2D(filters=1024, kernel_size=(3, 3), dilation_rate=(24, 24), activation='relu', name='fc6_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=1024, kernel_size=(1, 1), activation='relu', name='fc7_4')(b4)
    b4 = Dropout(0.5)(b4)
    b4 = Conv2D(filters=23, kernel_size=(1, 1), activation='relu', name='fc8_voc12_4')(b4)

    s = Add()([b1, b2, b3, b4])
    logits = UpSampling2D(size=upsampling, interpolation='bilinear')(s)
    out = Activation('softmax')(logits)

    model = Model(input, out, name='deeplabV2')

    return model