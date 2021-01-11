from data_loader import load_img
from models import NeuralTransfer
import matplotlib.pyplot as plt
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

content_path = '/home/ubuntu/bjh/Gan/neural_transfer/tubingen.jpg'
style_path = '/home/ubuntu/bjh/Gan/neural_transfer/starry-night.jpg'
content_image = load_img(content_path)
style_image = load_img(style_path)

neural = NeuralTransfer(input_shape=(600,800),
                        content_image=content_image,
                        style_image=style_image,
                        epochs = 5,
                        step_per_epoch = 100)

image = tf.Variable(content_image)                        
neural.train(image)

# print(content_image)

# content_layers, style_layers = layers_name(content_image)
