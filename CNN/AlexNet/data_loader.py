import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )

    test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        )

    train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/bjh/Gan/archive2/seg_train/seg_train',
        target_size=(224, 224),
        batch_size=32,
        class_mode = 'categorical'
        )

    validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/bjh/Gan/archive2/seg_test/seg_test',
        target_size=(224, 224),
        batch_size=32,
        class_mode = 'categorical'
        )

    return train_generator, validation_generator

