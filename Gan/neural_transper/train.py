from data_loader import load_img, imshow
from models import load_model,print_layers, layers_name,vgg_layers, StyleContentModel
import matplotlib.pyplot as plt
import tensorflow as tf

content_path = '/home/ubuntu/bjh/Gan/neural_transfer/tubingen.jpg'
style_path = '/home/ubuntu/bjh/Gan/neural_transfer/starry-night.jpg'
content_image = load_img(content_path)
style_image = load_img(style_path)
print(content_image)

content_layers, style_layers = layers_name(content_image)
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('스타일:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    크기: ", output.numpy().shape)
  print("    최솟값: ", output.numpy().min())
  print("    최댓값: ", output.numpy().max())
  print("    평균: ", output.numpy().mean())
  print()

print("콘텐츠:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    크기: ", output.numpy().shape)
  print("    최솟값: ", output.numpy().min())
  print("    최댓값: ", output.numpy().max())
  print("    평균: ", output.numpy().mean())

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def style_content_loss(outputs,style_weight,content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / 1

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / 5
    loss = style_loss + content_loss
    return loss

def train_step(img,i):
    with tf.GradientTape() as tape:
        outputs = extractor(img)
        loss = style_content_loss(outputs,1e-2,1e-4)

    grad = tape.gradient(loss, img)
    opt.apply_gradients([(grad, img)])
    image.assign(clip_0_1(img)) 
    plt.savefig('/home/ubuntu/bjh/Gan/code/' + "%d.png" % (i))

import time
def train(image):
    
    start = time.time()

    epochs = 10
    steps_per_epoch = 100

    step = 0
    for n in range(epochs):
        for m in range(steps_per_epoch):
            step += 1
            train_step(image,step)
            print(".", end='')
        display.clear_output(wait=True)
        display.display(tensor_to_image(image))
        print("훈련 스텝: {}".format(step))

    end = time.time()
    print("전체 소요 시간: {:.1f}".format(end-start))

train(image)