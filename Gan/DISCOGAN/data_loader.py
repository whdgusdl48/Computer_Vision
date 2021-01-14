import scipy
from glob import glob
import numpy as np
import cv2
class DataLoader():
    
    def __init__(self,dataset_name,img_resize=(128,128)):
        self.dataset_name = dataset_name
        self.img_resize = img_resize
    
    def load_data(self,batch_size=1,is_testing=False):
        data_type = 'train' if not is_testing else 'val'
        path = glob('/home/ubuntu/bjh/Gan/DISCOGAN/datasets/%s/%s/*' % (self.dataset_name, data_type))
        batch = np.random.choice(path, size=batch_size)

        imgs_A, imgs_B = [], []
        for img in batch:
            img = self.imread(img)
            h, w, _ = img.shape
            half_w = int(w/2)
            img_A = img[:, :half_w, :]
            img_B = img[:, half_w:, :]

            img_A = cv2.resize(img_A,self.img_resize)
            img_B = cv2.resize(img_B,self.img_resize)

            if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = 'train' if not is_testing else 'val'
        path = glob('/home/ubuntu/bjh/Gan/DISCOGAN/datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape
                half_w = int(w/2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = cv2.resize(img_A,self.img_resize)
                img_B = cv2.resize(img_B,self.img_resize)

                if not is_testing and np.random.random() > 0.5:
                        img_A = np.fliplr(img_A)
                        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = cv2.resize(img, self.img_resize)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return cv2.imread(path, cv2.COLOR_BGR2RGB).astype(np.float)    