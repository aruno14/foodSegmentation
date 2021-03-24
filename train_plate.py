import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import os
import glob
import numpy as np

image_size = (128, 128)
batch_size = 16
batch_size_val = 8
epochs = 15
model_name = "plate_model"

datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.3)

train_generator = datagen.flow_from_directory(
    "data/plate",
    target_size=image_size,
    shuffle=True,
    seed=126,
    subset='training',
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    "data/plate",
    target_size=image_size,
    shuffle=True,
    seed=126,
    subset='validation',
    batch_size=batch_size_val,
    class_mode='categorical')

if os.path.exists(model_name):
    print("Load: " + model_name)
    classifier = load_model(model_name)
else:
    in1 = Input(shape=(128, 128)+ (3,))
    x = Conv2D(64, (3, 3), padding='same', activation='relu', input_shape= image_size+(3,))(in1)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.4)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #x = Conv2D(128, (3, 3), activation='relu')(x)
    #x = Dropout(0.4)(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x)

    features = Flatten()(x)
    x = Dense(64, activation='relu')(features)
    x = Dropout(0.2)(x)
    x = Dense(2, activation='sigmoid')(x)
    classifier = Model(inputs=[in1], outputs=[x])

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    classifier.summary()

classifier.fit(train_generator, steps_per_epoch=train_generator.samples//batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=validation_generator.samples//batch_size_val)
classifier.save(model_name)

def load_image_into_numpy_array(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size)
    return img

for image in (glob.glob("data/plate/empty/*.jpg") + glob.glob("data/plate/full/*.jpg")):
    image_np = load_image_into_numpy_array(image)
    prediction = classifier.predict(np.array([image_np]))[0]
    print("Prediction", image, prediction)
