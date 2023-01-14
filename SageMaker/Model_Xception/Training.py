from modelClass import model
import os
import tensorflow as tf

def training_model(config_training, nb_classes, traingen, valgen):
    nb_couches_rentrainement, input_size, nb_epochs = config_training.model_params["nb_couches_rentrainement"], config_training.model_params["input_size"], config_training.model_params["nb_epochs"]
    model_ml = model(nb_classes, nb_couches_rentrainement, (input_size,input_size))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join("/opt/ml/model","model_drowsiness_level.h5"),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/opt/ml/output/tensorboard', histogram_freq=1)
    
    history = model_ml.fit(traingen, steps_per_epoch=len(traingen),
                    epochs=nb_epochs,
                   validation_data = valgen, 
                   validation_steps=len(valgen),
                   verbose=2,
                   callbacks=[model_checkpoint_callback, tensorboard_callback])



