# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:42:59 2021

@author: neham
"""


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pathlib import Path

data_dir = Path(r"/home/ADF/axs1603/test_runs/cloud_detector_simstest/pics")
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


#generating a cloudy pic
cloudy = list(data_dir.glob('cloudy/*'))
#PIL.Image.open(str(cloudy[13]))

#generating a clear sky pic
clear = list(data_dir.glob('clear/*'))
#PIL.Image.open(str(clear[13]))


#defining some parameters

batch_size = 64
img_height = 180
img_width = 180

#make a training set with 80-20 split
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 17,
    image_size = (img_height,img_width),
    batch_size = batch_size)


#make a validation set 
vals_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed = 17,
    image_size=(img_height,img_width),
    batch_size=batch_size)


#verifying by printing class names
class_names = train_ds.class_names
print(class_names)

#visualising the data
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
       # plt.imshow(images[i].numpy().astype("uint8"))
        #plt.title(class_names[labels[i]])
        #plt.axis("off")
        
        
#load data using dataset.cache() to prevent blocking I/O
#use Dataset.prefetch() to overlap data preprocessing and model execution

AUTOTUNE = tf.data.AUTOTUNE

train_df = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
vals_ds = vals_ds.cache().prefetch(buffer_size=AUTOTUNE)

#standardise/normalise the data

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)



#Applying data augmentation to improve predictions 

data_augmentation = keras.Sequential(
[layers.experimental.preprocessing.RandomFlip("horizontal",
                                             input_shape = (img_height, 
                                                           img_width,
                                                           3)),
layers.experimental.preprocessing.RandomRotation(0.4),
layers.experimental.preprocessing.RandomZoom(0.3)])

#visualise augmented pics
plt.figure(figsize=(10,10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        #ax = plt.subplot(3,3,i+1)
        #plt.imshow(augmented_images[0].numpy().astype("uint8"))
        #plt.axis("off")
        
        
#Applying dropout to improve regularisation, and creating the model 

num_classes = 2

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(16, 3, padding = 'same', activation = 'softmax'), #first layer has 16 filters
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding = 'same', activation = 'softmax'), #second layer has 32 filters
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding = 'same', activation = 'softmax'), #third layer has 64 filters
    layers.MaxPooling2D(),
    layers.Dropout(0.2), # drop out 20% output units
    layers.Flatten(),
    layers.Dense(256, activation = 'relu'),
    layers.Dense(num_classes)    
])

#compiling and training the model

model.compile(optimizer = 'adam',
             loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
model.summary()


epochs = 15
history = model.fit(
    train_ds,
    validation_data=vals_ds,
    epochs=epochs)

#visualise the results

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


#testing with a new picture
    #using cloudy picture
    
test_path = Path(r"/home/ADF/axs1603/test_runs/cloud_detector_simstest/AllSkyImage20190701-202200.jpg")
img = keras.preprocessing.image.load_img(
    test_path, target_size = (img_height, img_width))

img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) 

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print("This image most likely belongs to {} with {:.2f} % confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
