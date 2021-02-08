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

class ResNet():

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
        
    

        history = self.model.fit(dataset,epochs = self.epochs,validation_data = testdataset)
        
        self.model.save('mymodel.h5')

    def loss(self,logits,labels):
        loss = tf.keras.losses.categorical_crossentropy(logits,labels)
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
    def residual(layers,filters,kernel,strides,last_strides,first=False):
        x = layers

        x = Conv2D(filters=filters[0],kernel_size=kernel[0],strides=strides[0],padding='valid')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
  
        x = Conv2D(filters=filters[1],kernel_size=kernel[1],strides=strides[1],padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
  
        if first:
            x = Conv2D(filters=filters[2],kernel_size=kernel[2],strides=strides[2],padding='valid')(x)
            layers = Conv2D(filters=filters[2],strides=last_strides,kernel_size=kernel[2],padding='valid')(layers)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            add_x = Add()([x,layers])
            add_x = Activation('relu')(add_x)
            return add_x
  
        else:
            x = Conv2D(filters=filters[2],kernel_size=kernel[2],strides=strides[2],padding='valid')(x)
            x = BatchNormalization()(x)
            add_x = Add()([x,layers])
            add_x = Activation('relu')(add_x)
            return add_x
    def net(input_shape):
        input = Input(shape=input_shape)
        x = input
        x = Conv2D(64, (7, 7), strides=(2, 2),padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = MaxPooling2D(2,2)(x)
        x = residual(x,[64,64,256],[1,3,1],[1,1,1],1,first=True)
        x = residual(x,[64,64,256],[1,3,1],[1,1,1],1)
        x = residual(x,[64,64,256],[1,3,1],[1,1,1],1)
        x = residual(x,[128,128,512],[1,3,1],[2,1,1],2,first=True)
        x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
        x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
        x = residual(x,[128,128,512],[1,3,1],[1,1,1],2)
        x = residual(x,[256,256,1024],[1,3,1],[2,1,1],2,first=True)
        x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
        x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
        x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
        x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
        x = residual(x,[256,256,1024],[1,3,1],[1,1,1],2)
        x = residual(x,[512,512,2048],[1,3,1],[2,1,1],2,first=True)
        x = residual(x,[512,512,2048],[1,3,1],[1,1,1],2)
        x = residual(x,[512,512,2048],[1,3,1],[1,1,1],2)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        output = Dense(4,activation='softmax')(x)
        return Model(input,output)
    