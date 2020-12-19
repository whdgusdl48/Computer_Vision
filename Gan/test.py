import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ----- DataSet Import ------
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

NUM_CLASSES = 10
print(x_train.shape)

# ----- Feature PreProcessing ----

x_train = x_train[:3000].astype('float32') / 255.0
x_test = x_test[:1000].astype('float32') / 255.0

# ----- One hot encoding -------
y_train = to_categorical(y_train[:3000], NUM_CLASSES)
y_test = to_categorical(y_test[:1000], NUM_CLASSES)

print(x_train[54,12,12,1])

# Using Sequence
model = Sequential([
    Dense(200, activation = 'relu', input_shape=(32,32,3)),
    Flatten(),
    Dense(150, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# Funtional API
input_layer = Input(shape = (32,32,3))

x = Flatten()(input_layer)
x = Dense(200, activation = 'relu')(x)
x = Dense(150, activation = 'relu')(x)
output_layer = Dense(10, activation = 'softmax')(x)

api_model = Model(input_layer,output_layer)

# print(model.summary())
# print(api_model.summary())

optimizer = Adam(lr = 0.0005)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=10, shuffle=True)

model.evaluate(x_test,y_test)

classes = np.array(['airplain','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
preds = model.predict(x_test)
preds_single = classes[np.argmax(preds,axis=-1)]
actual_single = classes[np.argmax(y_test, axis=-1)]

import matplotlib.pyplot as plt

n_to_show = 10
indices = np.random.choice(range(len(x_test)),n_to_show)

fig = plt.figure(figsize=(15,3))
fig.subplots_adjust(hspace=0.4,wspace=0.4)

for i,idx in enumerate(indices):
    img = x_test[idx]
    ax = fig.add_subplot(1,n_to_show,i + 1)
    ax.axis('off')
    ax.imshow(img)
