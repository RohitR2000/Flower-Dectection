import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import L2
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage
import pyautogui

TARGET_SIZE = (299, 299)
SEED = 42
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(299, 299, 3), kernel_regularizer=L2(1e-3)),
    MaxPooling2D((3, 3)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=L2(1e-3)),
    MaxPooling2D((3, 3)),
    Dropout(0.3),
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=L2(1e-3)),
    MaxPooling2D((3, 3)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=L2(1e-3)),
    Dropout(0.3),
    Dense(5, activation='softmax')
])
model.load_weights('first_model.h5')
test_generator = ImageDataGenerator(rescale=1 / 255.)
d = {0: 'daisy', 1: 'dandelion', 2: 'rose', 3: 'sunflower', 4: 'tulip'}


def webcam():
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        cv2.imwrite("D:\\DYPIU\\SEM 7\\PROJECTS\\tc_06_Proj\\Tc06_project\\video\\1.png", img)
        test_batches = test_generator.flow_from_directory(
            directory="D:\\DYPIU\\SEM 7\\PROJECTS\\tc_06_Proj\\Tc06_project",
            target_size=TARGET_SIZE,
            color_mode='rgb',
            batch_size=1,
            classes=['video'],
            seed=SEED,
            shuffle=False)
        res = model.predict(test_batches)
        cv2.putText(img=img, text=d[np.argmax(res)], org=(0 + int(200 / 10), 0 + int(200 / 1.5)),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=4, color=(255, 0, 0), thickness=7)
        cv2.imshow("image", img)
        print(d[np.argmax(res)])
        cv2.waitKey(1)


webcam()
# def screen():
#     while True:
#         img=pyautogui.screenshot()
#         img.save("D:\\DYPIU\\SEM 7\\PROJECTS\\tc_06_Proj\\Tc06_project\\video\\1.png")
#         test_batches = test_generator.flow_from_directory(
#             directory="D:\\DYPIU\\SEM 7\\PROJECTS\\tc_06_Proj\\Tc06_project",
#             target_size=TARGET_SIZE,
#             color_mode='rgb',
#             batch_size=1,
#             classes=['video'],
#             seed=SEED,
#             shuffle=False)
#         res = model.predict(test_batches)
#         img=cv2.imread("D:\\DYPIU\\SEM 7\\PROJECTS\\tc_06_Proj\\Tc06_project\\video\\1.png")
#         cv2.putText(img=img, text=d[np.argmax(res)], org=(0 + int(200 / 10), 0 + int(200 / 1.5)),
#                     fontFace=cv2.FONT_HERSHEY_DUPLEX,
#                     fontScale=4, color=(255, 0, 0), thickness=7)
#         cv2.imshow("image", img)
#         print(d[np.argmax(res)])
#         cv2.waitKey(1)
# screen()
# cv2.imshow("img",img)
# cv2.waitKey(1)
# train_generator = ImageDataGenerator(
#     rescale=1 / 255.,
#     validation_split=0.15)
#
# train_batches = train_generator.flow_from_directory(
#     directory="train",
#     target_size=TARGET_SIZE,
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     seed=SEED,
#     subset='training')
#
# valid_batches = train_generator.flow_from_directory(
#     directory="train",
#     target_size=TARGET_SIZE,
#     color_mode='rgb',
#     class_mode='categorical',
#     batch_size=32,
#     seed=SEED,
#     subset='validation')
# Build convolutional neural network using Keras Sequential API


# Print the model summary
#
# model.summary()
# Define the model optimizer, loss function and metrics
#
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )
# Fit the model
#
# history = model.fit(
#     train_batches,
#     validation_data=valid_batches,
#     batch_size=32,
#     epochs=15
# )
# Load the history into a pandas Dataframe
# model.save("basic.h5")
#
# df = pd.DataFrame(history.history)
# print(df.head(10))
# # Make a plot for the loss
# pd.DataFrame(history.history)[['loss','val_loss']].plot()
# plt.title("Loss")
# plt.show()
# # Make a plot for the accuracy
# pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot()
# plt.title("Accuracy")
# plt.show()
