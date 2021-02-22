from pyimagesearch import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy import io
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
print('데이터 셋을 로드합니다.')
rows = os.listdir(config.ANNOTS_PATH)
img = os.listdir(config.IMAGES_PATH)
print(len(img))
data = []
targets = []
filenames = []

for i,row in enumerate(rows):
    a = row

    b = io.loadmat(config.ANNOTS_PATH + '/' + row)

    (filename,startY, endY, startX, endX) = 'image_'+a[11:15]+'.jpg',b['box_coord'][0][0],b['box_coord'][0][1],b['box_coord'][0][2],b['box_coord'][0][3],
    print(filename,startX,startY,endX,endY)
    imagePath = os.path.sep.join([config.IMAGES_PATH,filename])
    image = cv2.imread(imagePath)
    # cv2.rectangle(image, (startX, startY), (endX, endY),
    #               (0, 255, 0), 2)
    # cv2.imshow('test',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    (h, w) = image.shape[:2]
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h

    image = load_img(imagePath, target_size=(224, 224))

    image = img_to_array(image)

    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

split = train_test_split(data, targets, filenames, test_size=0.10,
                         random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print("[INFO] saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

vgg = VGG16(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))

vgg.trainable = False

flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation='sigmoid')(bboxHead)

model = Model(inputs=vgg.input, outputs=bboxHead)

opt = Adam(lr=config.INIT_LR)
model.compile(loss="mse", optimizer=opt,metrics=['acc'])
print(model.summary())
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose=1)

print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")
# plot the model training history
N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)