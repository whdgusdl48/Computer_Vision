# tomorrow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import vgg16
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


def FCN8_helper(nClasses, input_height, input_width):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    img_input = Input(shape=(input_height, input_width, 3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input,
        pooling=None,
        classes=1000)
    assert isinstance(model, Model)

    o = Conv2D(
        filters=4096,
        kernel_size=(
            7,
            7),
        padding="same",
        activation="relu",
        name="fc6")(
        model.output)
    o = Dropout(rate=0.5)(o)
    o = Conv2D(
        filters=4096,
        kernel_size=(
            1,
            1),
        padding="same",
        activation="relu",
        name="fc7")(o)
    o = Dropout(rate=0.5)(o)

    o = Conv2D(filters=nClasses, kernel_size=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal",
               name="score_fr")(o)

    o = Conv2DTranspose(filters=nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score2")(o)

    fcn8 = Model(inputs=img_input, outputs=o)
    # mymodel.summary()
    return fcn8

def fcn8(nClasses, input_height, input_width):

    fcn8 = FCN8_helper(nClasses, input_height, input_width)

    # Conv to be applied on Pool4
    skip_con1 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool4")(fcn8.get_layer("block4_pool").output)
    Summed = add(inputs=[skip_con1, fcn8.output])

    x = Conv2DTranspose(nClasses, kernel_size=(2, 2), strides=(2, 2), padding="valid", activation=None,
                        name="score4")(Summed)

    ###
    skip_con2 = Conv2D(nClasses, kernel_size=(1, 1), padding="same", activation=None, kernel_initializer="he_normal",
                       name="score_pool3")(fcn8.get_layer("block3_pool").output)
    Summed2 = add(inputs=[skip_con2, x])

    #####
    Up = Conv2DTranspose(nClasses, kernel_size=(8, 8), strides=(8, 8),
                         padding="valid", activation=None, name="upsample")(Summed2)

    Up = Activation("softmax")(Up)

    mymodel = Model(inputs=fcn8.input, outputs=Up)

    return mymodel
