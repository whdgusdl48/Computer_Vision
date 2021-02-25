from utils import config
from tensorflow.keras.preprocessing.image import img_to_array
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import io
from model import unet
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
seg_list = os.listdir(config.SEMANTIC_PATH)
real_list = os.listdir(config.IMAGE_PATH)
annot_list = os.listdir(config.ANNOT_PATH)
data = []
target = []
label = []
real_imagePaths = []
target_imagePaths = []

for i,seg in enumerate(seg_list):
    real_image_path = config.IMAGE_PATH + '/' + seg[:-4] + '.jpg'
    seg_image_path = config.SEMANTIC_PATH + '/' + seg
    annot_path = config.ANNOT_PATH + '/' + seg[:-4] + '.xml'
    real_image = cv2.imread(real_image_path)
    target_image = cv2.imread(seg_image_path)
    # print(annot_path)
    # load Annotation xml data

    doc = ET.parse(annot_path)
    root = doc.getroot()

    objects = root.findall("object")
    img_label = []
    for _object in objects:
        name = _object.find("name").text
        img_label.append(name)
    label.append(img_label)

    real_image = load_img(real_image_path, target_size=(256,256))
    target_image = load_img(seg_image_path, target_size=(256,256))

    real_image = img_to_array(real_image)
    target_image = img_to_array(target_image)

    data.append(real_image)
    target.append(target_image)
    real_imagePaths.append(real_image_path)


data = np.array(data,dtype=np.float32) / 255.0
target = np.array(target, dtype=np.float32) / 255.0
lb = LabelBinarizer()
labels = lb.fit_transform(label)
print(lb.classes)
# print(label.shape)

print(target.shape)
real_image_path = np.array(real_imagePaths)

split = train_test_split(data,target,real_image_path,test_size=0.2,random_state=42)
#
# (trainImages, testImages) = split[:2]
# (trainLabels, testLabels) = split[2:4]
# (trainPaths, testPaths) = split[4:6]
# print("[INFO] 테스트를 실행할 이미지 경로를 저장합니다....")
# f = open(config.TEST_PATH, "w")
# f.write("\n".join(testPaths))
# f.close()
#
# model = unet.get_model((256,256),3)
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.Adam(learning_rate=config.INIT_LR),)
# model.fit(trainImages,trainLabels,
#           epochs=25,
#           batch_size=4,
#           validation_data=(testImages,testLabels))
#
# val_preds = model.predict(testImages)
#
# cv2.imshow('real',testLabels[0])
# cv2.imshow('testing',val_preds[0])
# cv2.waitKey()
# cv2.destroyAllWindows()

