import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os,cv2

def load_ffhq(path,image_size):
    img_path = os.path.join(path)
    img_list = os.listdir(img_path)
    data = []
    for i in range(len(img_list)):
        path_folder = img_path + "/" + img_list[i]
        lising = os.listdir(path_folder)
        for number in range(len(lising)):
            data_img = cv2.imread(path_folder + "/" + lising[number])
            data_img = cv2.cvtColor(data_img,cv2.COLOR_BGR2RGB)
            data_img = np.array(data_img)
            data_img = cv2.resize(data_img,image_size,interpolation = cv2.INTER_CUBIC)
            data.append(data_img)
    data = np.array(data[:3000])
    print(data.shape)
    return data    

def image_processing(dataset):
    data = tf.cast(dataset,tf.float32)
    data = (data - 127.5) / 127.5
    return data

