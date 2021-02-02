import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, MaxPooling2D,Flatten
from tensorflow.keras.models import Sequential, Model
import numpy as np
import time

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

class Lenet_5():

    def __init__(self,batch_size,epoch,image_size):
        self.batch_size = batch_size
        self.epoch = epoch
        self.image_size = image_size
        self.models = self.Net(self.image_size)
        self.models.summary()
        self.models.optimizer = tf.keras.optimizers.Adam(lr=0.00)

    def loss(self,logits,labels):
        loss = tf.keras.losses.categorical_crossentropy(logits,labels)
        return loss

    def train_step(self,images,labels):
        with tf.GradientTape() as t:
            logits = self.models(images,training=True)
            loss = self.loss(logits,labels)
        grad = t.gradient(loss,self.models.trainable_variables)
        self.models.optimizer.apply_gradients(zip(grad,self.models.trainable_variables))
        return loss,logits

    def train(self,dataset,testdataset):
        train_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.Accuracy()
        test_loss = tf.keras.metrics.Mean()
        test_acc = tf.keras.metrics.Accuracy()

        for epoch in range(self.epoch):
            start = time.time()
            print('start')
            for image_batch,label_batch in dataset:
                loss,logit = self.train_step(image_batch,label_batch)
                train_loss.update_state(loss)
                train_acc.update_state(tf.argmax(label_batch),tf.argmax(logit))
            
            for image_batch,label_batch in testdataset:
                predictions = self.models(image_batch)
                loss_value = self.loss(label_batch, predictions)
                test_loss.update_state(loss_value)
                test_acc.update_state(tf.argmax(label_batch), tf.argmax(predictions))
            print('epoch: {}/{}, train loss: {:.4f}, train accuracy: {:.4f}, test loss: {:.4f}, test accuracy: {:.4f}'.format(
                epoch + 1, self.epoch, train_loss.result().numpy(), train_acc.result().numpy(), test_loss.result().numpy(), test_acc.result().numpy()))
            
    def Net(self,shape):
        input = Input(shape=shape)
        x = input
        x = Conv2D(filters=32,strides=1,kernel_size=5,activation='relu')(x)
        x = MaxPooling2D(2,2)(x)
        x = Conv2D(filters=64,strides=1,kernel_size=5,activation='relu')(x)
        x = Flatten()(x)
        x = Dense(120,activation='relu')(x)
        x = Dense(84,activation='relu')(x)
        x = Dense(10,activation='softmax')(x)

        model = Model(input,x)
       
        return model
