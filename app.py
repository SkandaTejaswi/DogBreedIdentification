from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

from itertools import chain
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import pickle

import gc
gc.collect()

y = pickle.load(open('y.dat','rb'))    
    
le = LabelEncoder()
le.fit(y)


app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


MODEL_PATH ='final_model.h5'


model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    
    img = load_img(img_path, color_mode = "rgb", target_size=(299, 299, 3))
    x = img_to_array(img)
    x = x/255.0
    x = np.expand_dims(x, axis=0)
    test_predictions = model.predict(x)
    predictions = le.classes_[np.argmax(test_predictions,axis=1)]
    return predictions[0]


@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'there is no file1 in form!'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        return model_predict(path,model)
    
    return None



if __name__ == '__main__':
    app.run(debug=False)