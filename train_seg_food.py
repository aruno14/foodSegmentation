import os
from glob import glob
from datetime import datetime
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image

#IMG_HEIGHT=600
#IMG_WIDTH=800
IMG_HEIGHT=416
IMG_WIDTH=256

#x_files = glob('myData/images/*')
#y_files = glob('myData/masks/*')
x_files = glob('TrayDataset/TrayDataset/XTrain/*')
y_files = glob('TrayDataset/TrayDataset/yTrain/*')

files_ds = tf.data.Dataset.from_tensor_slices((x_files, y_files))

def process_img(file_path:str, channels=3, diviser=1):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size=(IMG_HEIGHT//diviser, IMG_WIDTH//diviser))
    return img

files_ds = files_ds.map(lambda x, y: (process_img(x), process_img(y, channels=3, diviser=2))).batch(1)


def get_model(IMG_HEIGHT, IMG_WIDTH):
    in1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    preprop = tf.keras.layers.experimental.preprocessing.Resizing(IMG_HEIGHT//2, IMG_WIDTH//2)(in1)
    preprop = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(preprop)
    preprop = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")(preprop)
    preprop = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)(preprop)

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(preprop)#(in1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

    up1 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up2 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    segmentation = Conv2D(3, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])

    losses = {'seg': 'binary_crossentropy'}
    #losses = {'seg': 'mean_squared_error'}

    metrics = {'seg': ['acc']}
    model.compile(optimizer="adam", loss = losses, metrics=metrics)

    return model

model_name = "traySegModel"
if os.path.exists(model_name):
    model = tf.keras.models.load_model(model_name)
else:
    model = get_model(IMG_HEIGHT, IMG_WIDTH)

model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)

def showPrediction(imagePath:str, count=0):
    test_image = process_img(imagePath)
    predictions = model.predict(np.asarray([test_image]))
    predictions *= 255
    predicted_image = predictions[0]

    #im = Image.fromarray(np.squeeze(predicted_image), mode="L")
    print(predicted_image.shape)
    im = Image.fromarray(predicted_image, mode="RGB")
    im.save("predict-"+str(count)+".png")

    return test_image, predictions

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    #test_image, predictions = showPrediction("myData/images/assiette-01.jpg", epoch)
    test_image, predictions = showPrediction("TrayDataset/TrayDataset/XTrain/image-1001a01.jpg", epoch)

    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
      tf.summary.image("Training data", predictions, step=epoch)
      tf.summary.image("Input data", [test_image], step=epoch)

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 5
print(files_ds)
#tensorboard --logdir logs/
history = model.fit(files_ds, epochs=epochs, steps_per_epoch=100, callbacks=[tensorboard_callback, DisplayCallback()])#, steps_per_epoch=10
model.save(model_name)

metrics = history.history

plt.plot(history.epoch, metrics['loss'], metrics['acc'])
plt.legend(['loss', 'acc'])
plt.savefig("fit-history.png")
plt.show()
plt.close()

#for image_path in ["myData/images/assiette-01.jpg", "myData/images/choucroute-01.jpg"]:
#    showPrediction(image_path)
