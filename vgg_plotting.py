import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L2
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage

base_model = tf.keras.applications.VGG16(input_shape=(128, 128, 3), include_top=False, weights="imagenet")
# Freezing Layers

for layer in base_model.layers[:-4]:
    layer.trainable = False
model = Sequential()
model.add(base_model)
model.add(Dropout(0.3))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32, kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(5, activation='softmax'))
TARGET_SIZE = (128, 128)
SEED = 42
test_datagen = ImageDataGenerator(rescale=1. / 255)
d = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}


def webcam():
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        cv2.imwrite("D:/DYPIU/SEM 7/PROJECTS/tc_06_Proj/Tc06_project/video/1.png", img)
        test_batches = test_datagen.flow_from_directory(
            directory="D:/DYPIU/SEM 7/PROJECTS/tc_06_Proj/Tc06_project",
            target_size=TARGET_SIZE,
            color_mode='rgb',
            batch_size=1,
            classes=['video'],
            seed=SEED,
            shuffle=False)
        res = model.predict(test_batches)
        cv2.putText(img=img, text=d[np.argmax(res[0])], org=(0 + int(200 / 10), 0 + int(200 / 1.5)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=4, color=(255, 0, 0), thickness=7)
        cv2.imshow("image", img)
        print(np.argmax(res[0]))
        cv2.waitKey(1)


webcam()
