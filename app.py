from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow
import pandas as pd

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
from io import BytesIO
from tensorflow.python.keras import backend as K
# for matrix math
import numpy as np
import cv2
import matplotlib as plt
# for regular expressions, saves time dealing with string data
import re

# system level operations (like loading files)
import sys

# for reading operating system data
import os

# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request

# initalize our flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './models/'


json_file = open(MODEL_PATH+'model.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(MODEL_PATH +"model.h5")

#set paths to upload folder
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app.config['IMAGE_UPLOADS'] = os.path.join(APP_ROOT, 'uploads')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def convertDataToImage(image):
    img = np.fromstring(image, np.uint8)
    file = cv2.imdecode(img, cv2.IMREAD_COLOR)
    # cv.imshow('', file)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    file = cv2.resize(file, (32,32))
    file = np.array([file]) / 255.0
    return file

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        image = request.files['file']
        filename = image.filename
        file_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
        image.save(file_path)
        image.stream.seek(0) # seek to the beginning of file
        image_byte = image.read()
        data = convertDataToImage(image_byte)
        preds = model.predict(data)
        # Process your result for human
        pred_class = preds.argmax(axis=-1)
        data_dict = {}
        data_dict = {
             0 : "Airplane",
             1 : "Car",
             2 : "Bird",
             3 : "Cat",
             4 : "Deer",
             5 : "Dog",
             6 : "Frog",
             7 : "Horse",
             8 : "Ship",
             9 : "Truck",
            }
        return data_dict.get(pred_class[0])
    return None


if __name__ == '__main__':
    model=loaded_model
    app.run(host='0.0.0.0', port=5000, debug=True)