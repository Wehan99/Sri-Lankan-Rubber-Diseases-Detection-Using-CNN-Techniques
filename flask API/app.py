import tensorflow as tf
import cv2
import os
import numpy as np
import requests
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model
model_filenameTrunk = os.path.join(os.path.dirname(__file__), 'Rubber_trunk_diseases_latest.h5')
modelTrunk = tf.keras.models.load_model(model_filenameTrunk)

model_filenameLeaf = os.path.join(os.path.dirname(__file__), 'Rubber_leaf_diseases_final.h5')
modelLeaf = tf.keras.models.load_model(model_filenameLeaf)

# API endpoint for prediction
@app.route('/predictLeaf', methods=['POST'])
def predictLeaf():
    # Check if the request contains the 'file' key in the JSON body
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in the request'})

    # Get the uploaded file from the request
    file = request.files['file']

    # Read the file into memory as a NumPy array
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.resize(image, (224, 224))

    # Convert the resized image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = np.expand_dims(rgb_image, axis=0)
    image2 = image2 / 255.0  # Normalize pixel values
    
    # Perform prediction using the image
    predictions = modelLeaf.predict(image2)

    class_labels = ['Colletotrichum', 'Corynespora', 'Oidium']

    predicted_disease = class_labels[int(np.argmax(predictions,axis=1))]

    # Determine the prediction label based on the threshold
    value = float(np.amax(predictions))

    # Convert prediction array to nested Python list
    prediction_list = predictions.tolist()

    # Serialize the prediction list to JSON
    prediction_json = json.dumps(prediction_list[0])

    # Return the predictions as a JSON response
    return jsonify({'predicted_disease': predicted_disease, 'Value': value, 'predictions' : prediction_json})


@app.route('/predictTrunk', methods=['POST'])
def predictTrunk():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in the request'})
    
    # Get the uploaded file from the request
    file = request.files['file']
    
    # Read the file into memory as a NumPy array
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Preprocess the image
    image = cv2.resize(image, (224, 224))

    # Convert the resized image to RGB format
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image2 = np.expand_dims(rgb_image, axis=0)
    image2 = image2 / 255.0  # Normalize pixel values

    # Perform prediction using the image
    prediction = modelTrunk.predict(image2)

    class_labels = ['Healthy', 'Unhealthy']

    predicted_disease = class_labels[int(np.argmax(prediction,axis=1))]

    # Determine the prediction label based on the threshold
    value = float(np.amax(prediction))

    # Convert prediction array to nested Python list
    prediction_list = prediction.tolist()

    # Serialize the prediction list to JSON
    prediction_json = json.dumps(prediction_list[0])

    # Return the predictions as a JSON response
    return jsonify({'predicted_disease': predicted_disease, 'Value': value, 'predictions' : prediction_json})

if __name__ == '__main__':
    app.run(debug=True)