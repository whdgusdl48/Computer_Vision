import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, Lambda, MaxPooling2D,Flatten,BatchNormalization,Activation
from tensorflow.keras.models import Sequential, Model
import numpy as np
import time

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.95):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True

class AlexNet():

    def __init__(self,
                 batch_size,
                 image_size,
                 epoch,
                 lr
                 ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.epochs = epoch
        self.lr = lr
        self.model = self.Net()
        self.model.optimizer = tf.keras.optimizers.Adam()
        self.model.summary()

    def train_fit(self,dataset,testdataset):
        
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])

        history = self.model.fit_generator(dataset,epochs = self.epochs,validation_data = testdataset,validation_steps=10)
    
    def loss(self,logits,labels):
        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(logits,labels))
        return loss

    def train_step(self,images,labels):
        with tf.GradientTape() as t:
            logits = self.model(images)
            loss = self.loss(logits,labels)
        grad = t.gradient(loss,self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grad,self.model.trainable_variables))
        return loss,logits

    def train(self,dataset,testdataset):
        train_loss = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.Accuracy()
        test_loss = tf.keras.metrics.Mean()
        test_acc = tf.keras.metrics.Accuracy()

        for epoch in range(self.epochs):
            start = time.time()
            print('start')
            for image_batch,label_batch in dataset:
            
                loss,logit = self.train_step(image_batch,label_batch)
                print(loss)
                train_loss.update_state(loss)
                train_acc.update_state(tf.argmax(label_batch),tf.argmax(logit))
            
            for image_batch,label_batch in testdataset:
                predictions = self.model(image_batch,training=True)
                loss_value = self.loss(label_batch, predictions)
                test_loss.update_state(loss_value)
                test_acc.update_state(tf.argmax(label_batch), tf.argmax(predictions))
            print('epoch: {}/{}, train loss: {:.4f}, train accuracy: {:.4f}, test loss: {:.4f}, test accuracy: {:.4f}'.format(
                epoch + 1, self.epochs, train_loss.result().numpy(), train_acc.result().numpy(), test_loss.result().numpy(), test_acc.result().numpy()))
        train_acc.reset_state()
        train_loss.reset_state()
        test_loss.reset_state()
        test_acc.reset_state() 

    def Net(self):
        input = Input(shape=self.image_size)
        x = input
        x = Conv2D(filters=96,kernel_size=11,strides=4,name='conv_layer_1')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=256,kernel_size=5,strides=1,name='conv_layer_2')(x)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Conv2D(filters=384,kernel_size=3,strides=1,name='conv_layer_3')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Conv2D(filters=384,kernel_size=3,strides=1,padding='same',name='conv_layer_4')(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=256,kernel_size=3,strides=1,padding='same',name='conv_layer_5')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2,2),strides=2)(x)
        x = Flatten()(x)
        x = Dense(4096,activation='relu')(x)
        x = Dense(4096,activation='relu')(x)
        
        x = Dense(6,activation='softmax')(x)

        model = Model(input,x)
        return model
    