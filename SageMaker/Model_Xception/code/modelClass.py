"""Importer les packages"""
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception


def model(nb_classes, nb_couches_rentrainement, input_size):
    conv_base = Xception(include_top=False,weights="imagenet",input_tensor=None,input_shape=input_size+(3,),pooling='avg')
    nb_couches = len(conv_base.layers)
    for _,layer in enumerate(conv_base.layers) : 
        if _ < nb_couches-nb_couches_rentrainement: 
            layer.trainable = False
        else : 
            layer.trainable = True
        
    model = keras.models.Sequential()
    
    model.add(conv_base)
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))    

    model.add(keras.layers.Dense(nb_classes, activation='softmax'))


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

