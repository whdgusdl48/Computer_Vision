import tensorflow as tf
import numpy as np
import time
import functools
import matplotlib.pyplot as plt
from data_loader import tensor_to_image
class NeuralTransfer():

    def __init__(self,
                 input_shape,
                 content_image,
                 style_image,
                 epochs,
                 step_per_epoch):
      self.input_shape = input_shape
      self.content_image = content_image
      self.style_image = style_image
      self.content_layers = ['block5_conv2']
      self.style_layers = ['block1_conv1',
                           'block2_conv1',
                           'block3_conv1', 
                           'block4_conv1', 
                           'block5_conv1']
      self.num_content_layers = len(self.content_layers)
      self.num_style_layers = len(self.style_layers)
      self.vgg = self.vgg_layers(self.content_layers + self.style_layers)
      self.optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
      self.style_weight = 1e-2
      self.content_weight = 1e4
      self.epochs = epochs
      self.step_per_epoch = step_per_epoch

    def load_VGG(self):
      x = tf.keras.applications.vgg19.preprocess_input(self.content_image*255)
      x = tf.image.resize(x,(224,224))
      vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
      prediction_probablities = vgg(x)
      print(prediction_probablities.shape)
      return vgg

    def vgg_layers(self,layer_names):
      vgg = tf.keras.applications.VGG19(include_top=False,weights='imagenet')
      vgg.trainable = False

      outputs = [vgg.get_layer(name).output for name in layer_names]

      model = tf.keras.Model([vgg.input], outputs)
      return model

    def gram_matrix(self,input_tensor):
      result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
      input_shape = tf.shape(input_tensor)
      num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
      return result/(num_locations)

    def styleContentModel(self,inputs):
      inputs = inputs*255.0
      preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
      outputs = self.vgg(preprocessed_input)
      style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

      style_outputs = [self.gram_matrix(style_output)
                     for style_output in style_outputs]

      content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

      style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
      
      return {'content':content_dict, 'style':style_dict}

    def style_content_loss(self,outputs):
      style_targets = self.styleContentModel(self.style_image)['style']
      content_targets = self.styleContentModel(self.content_image)['content']
      style_outputs = outputs['style']
      content_outputs = outputs['content']
      style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
      style_loss *= self.style_weight / self.num_style_layers

      content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
      content_loss *= self.content_weight / self.num_content_layers
      loss = style_loss + content_loss
      return loss

    def clip_0_1(self,image):
      return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def high_pass_x_y(self,image):
      x_var = image[:,:,1:,:] - image[:,:,:-1,:]
      y_var = image[:,1:,:,:] - image[:,:-1,:,:]

      return x_var, y_var

    def total_variation_loss(self,image):
      x_deltas, y_deltas = self.high_pass_x_y(image)
      return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

    def train_step(self,image):
      with tf.GradientTape() as t:
        outputs = self.styleContentModel(image)
        loss = self.style_content_loss(outputs)
        loss += 30 * self.total_variation_loss(image)
      grad = t.gradient(loss,image)
      self.optimizer.apply_gradients([(grad,image)])
      image.assign(self.clip_0_1(image))


    def train(self,image):
      step = 0
      start = time.time()

      for i in range(self.epochs):
          for m in range(self.step_per_epoch):
            step += 1
            self.train_step(image)
            print('.',end='')

          images = tensor_to_image(image)
          plt.imshow(images,cmap='gray_r') 
          plt.savefig('/home/ubuntu/bjh/Gan/neural_transfer/image/' + "%d_%d.png" % (i, m))  
          print("훈련 스텝: {}".format(step))
      end = time.time()
      print('time : {:.1f}'.format(end - start))

