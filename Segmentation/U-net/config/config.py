import os

BASE_PATH = '/tmp/pascal_voc_data/VOCdevkit/VOC2012'

IMAGE_PATH = os.path.sep.join([BASE_PATH,'JPEGImages'])
SEGMENT_PATH = os.path.sep.join([BASE_PATH,'SegmentationClass'])
ANNOT_PATH = os.path.sep.join([BASE_PATH,'Annotations'])

BASE_OUTPUT = 'output'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'detector.h5'])
LB_PATH = os.path.sep.join([BASE_OUTPUT, 'lb.pickle'])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, 'plots'])
TEST_PATH = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32