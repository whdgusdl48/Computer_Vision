from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
from pyimageserach import sliding_window, image_pyramid
import numpy as np
import argparse
import imutils
import time
import cv2

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

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to the input image")
ap.add_argument("-s", "--size", type=str, default="(350, 500)",
                help="ROI size (in pixels)")
ap.add_argument("-c", "--min-conf", type=float, default=0.9,
                help="minimum probability to filter weak detections")
ap.add_argument("-v", "--visualize", type=int, default=-1,
                help="whether or not to show extra visualizations for debugging")
args = vars(ap.parse_args())

WIDTH = 600
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = eval(args["size"])
INPUT_SIZE = (224,224)

print("[INFO] ResNet50 네트워크를 호출합니다.")
model = ResNet50(weights="imagenet", include_top=True)

orig = cv2.imread(args["image"])
orig = cv2.resize(orig,(500,500))
orig = imutils.resize(orig, width=WIDTH)
(H, W) = orig.shape[:2]

pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)

rois = []
locs = []

start = time.time()

for image in pyramid:

    scale = W / float(image.shape[1])

    for (x,y,roiOrig) in sliding_window(image,WIN_STEP,ROI_SIZE):

        x = int(x * scale)
        y = int(y * scale)
        w = int(ROI_SIZE[0] * scale)
        h = int(ROI_SIZE[1] * scale)

        roi = cv2.resize(roiOrig,INPUT_SIZE)
        roi = img_to_array(roi)
        roi = preprocess_input(roi)

        rois.append(roi)
        locs.append((x, y, x + w, y + h))

        if args["visualize"] > 0:
            clone = orig.copy()
            cv2.rectangle(clone, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            cv2.imshow("Visualization", clone)
            cv2.imshow("ROI", roiOrig)
            cv2.waitKey(0)

end = time.time()

print("[INFO] 윈도우 크기 실행시간 {:.5f} ".format(
    end - start))

rois = np.array(rois, dtype='float32')
print("RoI를 분류합니다.")

start = time.time()
preds = model.predict(rois)
end = time.time()
print("[INFO] RoI 분류시간 {:.5f} seconds".format(
    end - start))

preds = imagenet_utils.decode_predictions(preds, top=1)
labels = {}

for (i, p) in enumerate(preds):

    (imagenetID, label, prob) = p[0]

    if prob >= args["min_conf"]:
        box = locs[i]
        L = labels.get(label, [])
        L.append((box, prob))
        labels[label] = L
print(labels)
for label in labels.keys():

    print('결과 출력중...',label)
    clone = orig.copy()

    for (box, prob) in labels[label]:

        (startX,startY,endX,endY) = box

        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)

    cv2.imshow('before',clone)
    clone = orig.copy()

    boxes = np.array([p[0] for p in labels[label]])
    probs = np.array([p[1] for p in labels[label]])
    boxes = non_max_suppression(boxes,probs)

    for (startX,startY,endX,endY) in boxes:

        cv2.rectangle(clone,(startX,startY),(endX,endY),(0,255,0),2)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(clone, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    cv2.imshow('after',clone)
    cv2.waitKey(0)

