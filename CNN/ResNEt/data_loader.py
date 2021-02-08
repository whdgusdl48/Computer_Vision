import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

def load_data():

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )

    train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/bjh/Gan/archive (2)/images/images',
        target_size=(112,112),
        batch_size=8,
        shuffle=False,
        class_mode='categorical'
        )

    

    return train_generator


def load_data2():
    train_path = '/home/ubuntu/bjh/Gan/archive (2)/images/images'

    x_train,y_train,x_test,y_test = [],[],[],[]

    train_list = os.listdir(train_path)
    
    print(train_list)
    for i in range(len(train_list)):
        data_path = os.listdir(train_path + '/' + train_list[i])
        for j in range(len(data_path)):
            img = cv2.imread(train_path + '/' + train_list[i] +'/'+ data_path[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(112,112))
            img = np.array(img)
            x_train.append(img)
            y_train.append(i)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape,y_train.shape)

    x_train = x_train/ 255.
    
    y_train = tf.keras.utils.to_categorical(y_train)

    y_train = y_train.astype(np.float32)
    
    print(x_train.shape,y_train.shape)

    return x_train,y_train