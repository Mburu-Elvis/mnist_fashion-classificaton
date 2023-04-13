import os
import uuid
import flask
import PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'model.hdf5'))

ALLOWED_EXT = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Skirt', 'Sneaker', 'Bag', 'Ankle boot']

def predict(filename, model):
    img = load_img(filename, target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    