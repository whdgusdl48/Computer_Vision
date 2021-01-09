import os
import tensorflow as tf
import numpy as np
import cv2

def image_loader(path,image_size):
    img_path = os.path.join(path)
    directory_list = os.listdir(img_path)
    print(directory_list)
    for a in range(4):
        if a == 3 or a == 1:
            print('test_data')
            x_test = []
            y_test = []
            img_list = os.listdir(img_path + "/" + directory_list[a])
            print(directory_list[a])
            for i in range(len(img_list)):
                img = cv2.imread(img_path + "/" + directory_list[a] + "/" + img_list[i])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,image_size,interpolation=cv2.INTER_CUBIC)
                x_test.append(img)
                if a == 0:
                    y_test.append(0)
                else:
                    y_test.append(1)
        else:
            print('train_data')
            x_train = []
            y_train = []
            img_list = os.listdir(img_path + "/" + directory_list[a])
            print(directory_list[a])
            for i in range(len(img_list)):
                img = cv2.imread(img_path + "/" + directory_list[a] + "/" + img_list[i])
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,image_size,interpolation=cv2.INTER_CUBIC)
                x_train.append(img)
                if a == 0:
                    y_train.append(0)
                else:
                    y_train.append(1)

    train_data , test_data = (np.array(x_train),np.array(y_train)) , (np.array(x_test), np.array(y_test))                               
    return train_data, test_data

def image_loader_celeba(path,image_size):
    data = []
    img_path = os.path.join(path)
    img_list= os.listdir(img_path)
    print(len(img_list))
    for i in range(2000):
        number = np.random.randint(0,len(img_list))
        data_img = cv2.imread(img_path + "/" + img_list[number])
        data_img = cv2.cvtColor(data_img,cv2.COLOR_BGR2RGB)
        data_img = np.array(data_img)
        data_img = cv2.resize(data_img,(128,128),interpolation = cv2.INTER_CUBIC)
        data.append(data_img)
    
    data = np.array(data)
    print(data.shape)
    return data

def image_processing(dataset):
    data = np.array(dataset).astype(np.float32)
    data = (data - 127.5) / 127.5
    return data

def split_data(dataset,test_size):
    length = len(dataset)
    train_data = dataset[:length * test_size]
    train_data = np.random.shuffle(train_data)
    test_data = dataset[length * test_size:]
    test_data = np.random.shuffle(test_data)
    return train_data, test_data

def load_batch(dataset,batch_size):
    data = []
    length = int(len(dataset) / batch_size)
    for i in range(length):
        data.append(dataset[i * batch_size : (i+1)*batch_size])

    data = np.array(data)
    return data    