from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from flask import Flask, request, Response, json
import numpy as np
import cv2
from modelClass import model



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

nb_classes = len(activity_map)
nb_couches_rentrainement = int(os.getenv('nb_couches_rentrainement'))
input_size = int(os.getenv('input_size'))

model = model(nb_classes, nb_couches_rentrainement, input_size)

app = Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    health = model is not None
    status = 200 if health else 404
    return Response(response= '\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    imgs = [cv2.imdecode(nparr[i], cv2.IMREAD_COLOR) for i in range(nparr.shape[0])]
    inference_datagen = ImageDataGenerator.flow(imgs,
                                                target_size=input_size,)
    predictions = model.predict_from_batch(inference_datagen)
    json_dict = dict(predictions = predictions)
    
    return Response(response=json.dumps(json_dict), status=200, mimetype='application/json')