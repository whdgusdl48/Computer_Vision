from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from data import *
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

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(weight = None,input_shape = (512,512,3),initial_filter = 64):
        inputs = Input(input_shape)
        conv1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1_1 = Conv2D(initial_filter , (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPool2D(pool_size = (2,2))(conv1_1)

        conv2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPool2D(pool_size = (2,2))(conv2_1)

        conv3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPool2D(pool_size = (2,2))(conv3_1)

        conv4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        pool4 = MaxPool2D(pool_size = (2,2))(conv4_1)

    ####################################downsampling#############################################

        conv5 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5_1 = Conv2D(initial_filter * 16, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        up5 = UpSampling2D(size = (2,2))(conv5_1)

    ####################################upsampling#############################################

        merge_4 = concatenate([conv4_1,up5], axis = 3)
        conv_4 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_4)
        conv_4_1 = Conv2D(initial_filter * 8, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_4)
        up4 = UpSampling2D(size = (2,2))(conv_4_1)

        merge_3 = concatenate([conv3_1,up4], axis = 3)
        conv_3 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_3)
        conv_3_1 = Conv2D(initial_filter * 4, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_3)
        up3 = UpSampling2D(size = (2,2))(conv_3_1)

        merge_2 = concatenate([conv2_1,up3], axis = 3)
        conv_2 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_2)
        conv_2_1 = Conv2D(initial_filter * 2, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_2)
        up2 = UpSampling2D(size = (2,2))(conv_2_1)

        merge_1 = concatenate([conv1_1,up2], axis = 3)
        conv_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_1)
        conv_1_1 = Conv2D(initial_filter, (3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv_1)

        out = Conv2D(12, (1,1), activation = 'sigmoid')(conv_1_1)

        model = Model(inputs,out)
    #model = multi_gpu_model(model,gpus =2)
        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])


        return model

    def train(self):
        print("loading data")
        imgs_train, imgs_mask_train, imgs_test = self.load_data()
        print("loading data done")
        model = self.get_unet()
        print("got unet")
        model_checkpoint = ModelCheckpoint('unet_camvid.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1,
                  validation_split=0.1, shuffle=True, callbacks=[model_checkpoint])

        print('predict test data')
        imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
        np.save('./results/camvid_mask_test.npy', imgs_mask_test)

    def save_img(self):
        print("array to image")
        imgs = np.load('./results/camvid_mask_test.npy')
        piclist = []
        for line in open("./results/camvid.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = np.zeros((imgs.shape[1], imgs.shape[2], 3), dtype=np.uint8)
            for k in range(len(img)):
                for j in range(len(img[k])):  # cv2.imwrite也是BGR顺序
                    num = np.argmax(imgs[i][k][j])
                    if num == 0:
                        img[k][j] = [128, 128, 128]
                    elif num == 1:
                        img[k][j] = [128, 0, 0]
                    elif num == 2:
                        img[k][j] = [192, 192, 128]
                    elif num == 3:
                        img[k][j] = [255, 69, 0]
                    elif num == 4:
                        img[k][j] = [128, 64, 128]
                    elif num == 5:
                        img[k][j] = [60, 40, 222]
                    elif num == 6:
                        img[k][j] = [128, 128, 0]
                    elif num == 7:
                        img[k][j] = [192, 128, 128]
                    elif num == 8:
                        img[k][j] = [64, 64, 128]
                    elif num == 9:
                        img[k][j] = [64, 0, 128]
                    elif num == 10:
                        img[k][j] = [64, 64, 0]
                    elif num == 11:
                        img[k][j] = [0, 128, 192]
            img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path, img)


if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    # model.summary()
    # plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()