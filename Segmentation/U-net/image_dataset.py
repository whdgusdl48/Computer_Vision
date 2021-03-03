from config import config
from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import io
from tensorflow.keras.callbacks import ModelCheckpoint
import xml.etree.ElementTree as ET
import cv2
import os
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

print("[INFO] 데이터셋 로드중입니다.")
seg_list = os.listdir(config.SEGMENT_PATH)
real_list = os.listdir(config.IMAGE_PATH)
annot_list = os.listdir(config.ANNOT_PATH)
data = []
target = []
label = []
real_imagePaths = []
target_imagePaths = []

# sparse image 생성
# label2color 실행
# Annotation

class_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'void']

def _color_map(n_classes=256, normalized=False):
    """
    Builds the PASCAL VOC color map for the specified number of classes.
    :param n_classes: the number of classes in the colormap
    :param normalized: normalize pixel intensities, default is False
    :return: a list of RGB colors
    """
    def _bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((n_classes, 3), dtype=dtype)
    for i in range(n_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (_bitget(c, 0) << 7-j)
            g = g | (_bitget(c, 1) << 7-j)
            b = b | (_bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def color_map_viz(class_labels):
    """
    Plots the PASCAL VOC color map using the specified class labels.
    The number of classes is inferred from the length of the `class_labels` parameter.
    :param class_labels: the list of class labels
    :return: None
    """
    n_classes = len(class_labels) - 1
    row_size = 50
    col_size = 500
    cmap = _color_map()
    array = np.empty((row_size*(n_classes+1), col_size, cmap.shape[1]), dtype=cmap.dtype)
    for i in range(n_classes):
        array[i*row_size:i*row_size+row_size, :] = cmap[i]
    array[n_classes*row_size:n_classes*row_size+row_size, :] = cmap[-1]

    plt.imshow(array)
    plt.yticks([row_size*i+row_size/2 for i in range(n_classes+1)], class_labels)
    plt.xticks([])
    plt.show()

# color_map_viz(class_labels)
def color_map():
    """
    Builds the standard 21 class PASCAL VOC color map, plus one additional
    void/ignore label appended to the end of the list.
    :return: A list of RGB values.
    """
    cmap = _color_map()
    cmap = np.vstack([cmap[:len(class_labels)], cmap[-1].reshape(1, 3)])
    return cmap

def colors2labels(im, cmap, one_hot=False):
    """
    Converts a RGB ground truth segmentation image into a labels matrix with optional one-hot encoding.
    """
    if one_hot:
        labels = np.zeros((*im.shape[:-1], len(cmap)), dtype='uint8')
        for i, color in enumerate(cmap):
            labels[:, :, i] = np.all(im == color, axis=2).astype('uint8')
    else:
        labels = np.zeros(im.shape[:-1], dtype='uint8')
        for i, color in enumerate(cmap):
            labels += i * np.all(im == color, axis=2).astype(dtype='uint8')
    return labels

for i,seg in enumerate(seg_list):
    real_image_path = config.IMAGE_PATH + '/' + seg[:-4] + '.jpg'
    seg_image_path = config.SEGMENT_PATH + '/' + seg
    annot_path = config.ANNOT_PATH + '/' + seg[:-4] + '.xml'
    real_image = cv2.imread(real_image_path)
    target_image = cv2.imread(seg_image_path)
    # print(annot_path)
    # load Annotation xml data

    real_image = load_img(real_image_path, target_size=(128,128))
    target_image = load_img(seg_image_path, target_size=(128,128))

    real_image = img_to_array(real_image) / 255.0
    target_image = img_to_array(target_image)
    target_image = colors2labels(target_image,color_map())

    # cv2.imwrite('sparse/' + seg[:-4] + '.jpg',target_image)
    x = np.zeros([128, 128, 23])
    for i in range(128):
        for j in range(128):
            x[i, j, int(target_image[i][j])] = 1
    data.append(real_image)
    target.append(x)
    real_imagePaths.append(real_image_path)

data = np.array(data,dtype=np.float32)
target = np.array(target)
print(data.shape,target.shape)
split = train_test_split(data,target,test_size=0.2,random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]

model = model.get_unet(input_shape=(128,128,3))
model_checkpoint = ModelCheckpoint('unet_camvid.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit(trainImages,trainLabels,
          epochs=100,
          batch_size=8,
          validation_data=(testImages,testLabels),
          callbacks=[model_checkpoint])

val_preds = model.predict(testImages)

real = np.zeros((128,128,3))
a = color_map()
for i in range(128):
    for j in range(128):
        num = np.argmax(val_preds[0][i][j])
        real[i,j] = a[num]

real = cv2.resize(real,(512,512))
cv2.imshow('real',testImages[0])
cv2.imshow('pred',real)
cv2.waitKey()
cv2.destroyAllWindows()
