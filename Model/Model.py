"""Importer les packages"""
import numpy as np
import pandas as pd

import cv2

import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception



"""Importer les données"""


path_data = ""
input_size = (299,299)
batch_size = 32

nb_couches_rentrainement = 4
nb_epoch = 15

from zipfile import ZipFile
file_name = path_data+"/state-farm-distracted-driver-detection.zip"

with ZipFile(file_name, 'r') as zip:
  zip.extractall()
  print('Done')
  
activity_map = {
    'c0': 'Safe driving', 
    'c1': 'Texting - right', 
    'c2': 'Talking on the phone - right', 
    'c3': 'Texting - left', 
    'c4': 'Talking on the phone - left', 
    'c5': 'Operating the radio', 
    'c6': 'Drinking', 
    'c7': 'Reaching behind', 
    'c8': 'Hair and makeup', 
    'c9': 'Talking to passenger'
}

drivers_imgs_list = pd.read_csv(path_data+"/driver_imgs_list.csv")
classes = drivers_imgs_list.classname.unique()
nb_classes = len(classes)



"""Formater les données """


train_datagen = ImageDataGenerator(validation_split=0.2)

train_data = path_data+'/imgs/train'
test_data = path_data+'/imgs/test'
train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

val_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=input_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')


"""Xception"""


conv_base = Xception(include_top=False,weights="imagenet",input_tensor=None,input_shape=input_size+(3,),pooling='avg')
nb_couches = len(conv_base.layers)
for _,layer in enumerate(conv_base.layers) : 
  if _ < nb_couches-nb_couches_rentrainement: 
    layer.trainable = False
  else : 
    layer.trainable = True


"""Modele complet et compilé"""

model = keras.models.Sequential()
    
model.add(conv_base)
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.3))

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.3))    

model.add(keras.layers.Dense(nb_classes, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


"""Entrainement du modèle"""



history = model.fit_generator(train_generator, steps_per_epoch=len(train_generator),
                    epochs=nb_epoch,
                   validation_data = val_generator, 
                   validation_steps=len(val_generator))

model.save("model_drowsiness_level.h5")