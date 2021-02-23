from pyimageserach import config
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import io
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
data = []
labels = []
bboxes = []
imagePaths = []
if not os.path.isdir('/home/ubuntu/bjh/objectDetection/multi-class-object-detection/output/plots'):
    os.mkdir('/home/ubuntu/bjh/objectDetection/multi-class-object-detection/output/plots')
    print('그래프 폴더 생성')
else:
    print('이미 폴더가 존재합니다.')
# mat 파일을 변환해야함.
annots_list = os.listdir(config.ANNOTS_PATH)
print(annots_list)

for i,annot in enumerate(annots_list):
    # 각 어노테이션 폴더를 읽어온다.
    annots = os.listdir(config.ANNOTS_PATH + '/' + annot)
    annot_path = config.ANNOTS_PATH + '/' + annot
    for j in range(len(annots)):
        mats = io.loadmat(annot_path + '/' + annots[j])
        (filename,startY, endY, startX, endX,label) = 'image_'+annots[j][11:15]+'.jpg',\
                                                mats['box_coord'][0][0],\
                                                mats['box_coord'][0][1],\
                                                mats['box_coord'][0][2],\
                                                mats['box_coord'][0][3],\
                                                annot
        imagePath = os.path.sep.join([config.IMAGES_PATH,annot,filename])
        # print(imagePath)
        image = cv2.imread(imagePath)
        # 이미지 바운딩박스 일치한지 확인하는 코드
        # if i == 2:
        #     cv2.rectangle(image, (startX, startY), (endX, endY),
        #               (0, 255, 0), 2)
        #     cv2.imshow('test',image)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        (h, w) = image.shape[:2]
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        image = load_img(imagePath, target_size=(224, 224))

        image = img_to_array(image)
        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)

data = np.array(data,dtype=np.float32) / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

if len(lb.classes_) == 2:
    labels = to_categorical(labels)

print(labels.shape)
split = train_test_split(data, labels, bboxes, imagePaths,
                         test_size=0.20, random_state=42)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

print("[INFO] 테스트를 실행할 이미지 경로를 저장합니다....")
f = open(config.TEST_PATH, "w")
f.write("\n".join(testPaths))
f.close()

vgg = VGG19(weights='imagenet', include_top= False,
            input_tensor= Input(shape=(224,224,3)))
vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

# 추출해야할 값 => 박스 와 클래스 값 2개값을 도출

boxhead = Dense(128, activation='relu')(flatten)
boxhead = Dense(64, activation='relu')(boxhead)
boxhead = Dense(32, activation='relu')(boxhead)
boxhead = Dense(4, activation='sigmoid', name='bounding_box')(boxhead)

classification = Dense(512, activation='relu')(flatten)
classification = Dropout(0.5)(classification)
classification = Dense(512, activation='relu')(classification)
classification = Dropout(0.5)(classification)
classification = Dense(len(lb.classes_), activation='softmax', name='class_label')(classification)

model = Model(inputs= vgg.input, outputs=[boxhead, classification])

losses = {
    "class_label" : 'categorical_crossentropy',
    "bounding_box" : 'mean_squared_error'
}

lossWeights = {
    "class_label" : 1.0,
    "bounding_box" : 1.0
}

opt = Adam(lr=config.INIT_LR)
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)
print(model.summary())

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}
# construct a second dictionary, this one for our target testing
# outputs
testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

print("[INFO] training model...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1)
# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# serialize the label binarizer to disk
print("[INFO] saving label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config.NUM_EPOCHS)
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
    # plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()
# save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plotPath = os.path.sep.join([config.PLOT_PATH, "losses.png"])
plt.savefig(plotPath)
plt.close()

plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
         label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
         label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
# save the accuracies plot
plotPath = os.path.sep.join([config.PLOT_PATH, "accs.png"])
plt.savefig(plotPath)