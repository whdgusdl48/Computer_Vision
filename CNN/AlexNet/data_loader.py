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
        '/home/ubuntu/bjh/Gan/archive2/seg_train/seg_train',
        target_size=(112,112),
        batch_size=32,
        shuffle=True,
        class_mode='categorical'
        )

    validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/bjh/Gan/archive2/seg_test/seg_test',
        target_size=(112,112),
        batch_size=32,
        
        )

    return train_generator, validation_generator

def load_data2():
    train_path = '/home/ubuntu/bjh/Gan/archive2/seg_train/seg_train'
    test_path = '/home/ubuntu/bjh/Gan/archive2/seg_test/seg_test'
    x_train,y_train,x_test,y_test = [],[],[],[]

    train_list = os.listdir(train_path)
    test_list = os.listdir(test_path)
    print(train_list)
    for i in range(len(train_list)):
        data_path = os.listdir(train_path + '/' + train_list[i])
        for j in range(len(data_path)):
            img = cv2.imread(train_path + '/' + train_list[i] +'/'+ data_path[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(224,224))
            x_train.append(img)
            y_train.append(i)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape,y_train.shape)

    for i in range(len(test_list)):
        data_path = os.listdir(test_path + '/' + test_list[i])
        for j in range(len(data_path)):
            img = cv2.imread(test_path + '/' + test_list[i] +'/'+ data_path[i])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(224,224))
            x_test.append(img)
            y_test.append(i)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_train = x_train/ 255.
    x_test= x_test/ 255.

    
    print(x_train.shape,y_train.shape)

    return (x_train,y_train) , (x_test, y_test)