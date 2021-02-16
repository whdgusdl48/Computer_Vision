import tensorflow as tf
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import Mean
from tensorflow.keras.applications.vgg19 import preprocess_input

class SRGAN():

    def __init__(self,lr_size,hr_size):
        self.lr_size = lr_size
        self.hr_size = hr_size
        self.num_filters = 64
        self.num_res_block = 16
        self.content_loss = 'VGG54'
        self.vgg = self.vgg_54()
        self.lr = PiecewiseConstantDecay(boundaries=[100000],values=[1e-4,1e-5])
        self.D = self.build_discriminator()
        self.G = self.build_generator()
        self.D_optimizer = Adam(learning_rate = self.lr)
        self.G_optimizer = Adam(learning_rate = self.lr)
        self.binary_cross_entropy = BinaryCrossentropy(from_logits=False)
        self.mean_squared_error = MeanSquaredError()

    @tf.function
    def _content_loss(self, hr, sr):
        sr = preprocess_input(sr)
        hr = preprocess_input(hr)
        sr_features = self.vgg(sr) / 12.75
        hr_features = self.vgg(hr) / 12.75
        return self.mean_squared_error(hr_features, sr_features)

    def _generator_loss(self, sr_out):
        return self.binary_cross_entropy(tf.ones_like(sr_out), sr_out)

    def _discriminator_loss(self, hr_out, sr_out):
        hr_loss = self.binary_cross_entropy(tf.ones_like(hr_out), hr_out)
        sr_loss = self.binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
        return hr_loss + sr_loss

    def train_step(self,lr,hr):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as d_tape:
            lr = tf.cast(lr,tf.float32)
            hr = tf.cast(hr,tf.float32)

            sr = self.G(lr,training=True)

            hr_output = self.D(hr,training=True)
            sr_output = self.D(sr,training=True)

            con_loss = self._content_loss(hr,sr)
            gen_loss = self._generator_loss(sr_output)
            perc_loss = con_loss + 0.001 * gen_loss
            disc_loss = self._discriminator_loss(hr_output,sr_output)

        grad1 = gen_tape.gradient(perc_loss,self.G.trainable_variables)
        grad2 = d_tape.gradient(disc_loss,self.D.trainable_variables)

        self.D_optimizer.apply_gradients(zip(grad2,self.D.trainable_variables))
        self.G_optimizer.apply_gradients(zip(grad1,self.G.trainable_variables))

        return perc_loss,disc_loss
    
    def train(self,dataset,steps=200000):
        pls_metric = Mean()
        dls_metric = Mean()

        step = 0

        for lr, hr in dataset.take(steps):
            step += 1

            pl, dl = self.train_step(lr, hr)
            pls_metric(pl)
            dls_metric(dl)

            if step % 50 == 0:
                print(f'{step}/{steps}, perceptual loss = {pls_metric.result():.4f}, discriminator loss = {dls_metric.result():.4f}')
                pls_metric.reset_states()
                dls_metric.reset_states()

    def upsampling(self,layer,num_filters):
        x = Conv2D(filters=num_filters,kernel_size=3,padding='same')(layer)
        x = Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
        x = PReLU(shared_axes=[1,2])(x)

        return x
    
    def bottle_neck(self,layer,num_filters,momentum=0.8):
        x = Conv2D(filters=num_filters,kernel_size=3,strides=1,padding='same')(layer)
        x = BatchNormalization(momentum=momentum)(x)
        x = PReLU(shared_axes=[1,2])(x)
        x = Conv2D(filters=num_filters,kernel_size=3,strides=1,padding='same')(x)
        x = BatchNormalization(momentum=momentum)(x)
        x = Add()([layer,x])

        return x
    
    def discriminator_block(self,layer,num_filters,strides=1,batchnorm=True,momentum=0.8):
        x = Conv2D(filters=num_filters,kernel_size=3,strides=strides,padding='same')(layer)
        if batchnorm:
            x = BatchNormalization(momentum=momentum)(x)
        return LeakyReLU(0.2)(x)

    def build_generator(self):

        input = Input(shape=(self.lr_size,self.lr_size,3))
        x = Lambda(lambda x: x /255.)(input)

        x = Conv2D(filters=self.num_filters,kernel_size=9,strides=1,padding='same')(x)
        x = PReLU(shared_axes=[1,2])(x)
        x_1 = PReLU(shared_axes=[1,2])(x)

        for _ in range(self.num_res_block):
            x = self.bottle_neck(x,self.num_filters)
        
        x = Conv2D(filters=self.num_filters, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Add()([x_1, x])

        x = self.upsampling(x, self.num_filters * 4)
        x = self.upsampling(x, self.num_filters * 4)

        x = Conv2D(filters=3, kernel_size=9, padding='same', activation='tanh')(x)
        x = Lambda(lambda x: (x+1) * 127.5)(x)

        return Model(input, x)
    
    def build_discriminator(self):
        input = Input(shape=(self.hr_size,self.hr_size,3))
        x = Lambda(lambda x: x / 127.5 - 1)(input)

        x = self.discriminator_block(x,self.num_filters,batchnorm=False)
        x = self.discriminator_block(x,self.num_filters,strides=2)

        x = self.discriminator_block(x,self.num_filters * 2)
        x = self.discriminator_block(x,self.num_filters * 2,strides=2)

        x = self.discriminator_block(x,self.num_filters * 4)
        x = self.discriminator_block(x,self.num_filters * 4,strides=2)

        x = self.discriminator_block(x,self.num_filters * 8)
        x = self.discriminator_block(x,self.num_filters * 8,strides=2)

        x = Flatten()(x)

        x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1,activation='sigmoid')(x)

        return Model(input,x)

    def vgg_22(self):
        return self._vgg(5)
    
    def vgg_54(self):
        return self._vgg(20)
    
    def _vgg(self,output_layer):
        vgg = VGG19(input_shape=(None,None,3), include_top=False)
        return Model(vgg.input, vgg.layers[output_layer].output)
