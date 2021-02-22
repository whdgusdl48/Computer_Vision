import os

BASE_PATH = 'dataset'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'Annotations/Airplanes_Side_2'])

BASE_OUTPUT = 'output'
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INIT_LR = 1e-4
NUM_EPOCHS = 25
BATCH_SIZE = 32
