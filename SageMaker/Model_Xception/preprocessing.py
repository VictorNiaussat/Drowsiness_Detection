import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile


def unzip_data(config_training):
        with zipfile.ZipFile(os.path.join(config_training.training_directory, config_training.data_name), 'r') as zip_ref:
                zip_ref.extractall(config_training.training_directory)

def preprocessing_data(config_training):
    
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

    drivers_imgs_list = pd.read_csv(os.path.join(config_training.training_directory, "driver_imgs_list.csv"))
    classes = drivers_imgs_list.classname.unique()
    nb_classes = len(classes)

    input_size = (config_training.model_params['input_size'],config_training.model_params['input_size'])
    train_datagen = ImageDataGenerator(validation_split=0.2)

    train_data = os.path.join(config_training.training_directory, 'imgs/train')
    train_generator = train_datagen.flow_from_directory(
            train_data,
            target_size=input_size,
            batch_size=config_training.model_params['batch_size'],
            class_mode='categorical',
            subset='training')

    val_generator = train_datagen.flow_from_directory(
            train_data,
            target_size=input_size,
            batch_size=config_training.model_params['batch_size'],
            class_mode='categorical',
            subset='validation')
    
    return nb_classes, train_generator, val_generator


