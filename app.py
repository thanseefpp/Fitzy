################################## IMPORTING LIBRARIES ##################################

from flask import Flask, render_template, jsonify, request
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
from numpy.linalg import norm
import os
import pickle
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import requests

################################### APP CREATING ########################################

app = Flask(__name__)

################################### LOADING MODELS ######################################

feature_list = np.array(pickle.load(open('model/trained_image_set.pkl', 'rb')))
filenames = pickle.load(open('model/filenames.pkl', 'rb'))

################################### CREATING MODEL ######################################

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3)) # rgb color(3), top layer setting false.
model.trainable = False
model = tf.keras.Sequential([
  model,
  GlobalMaxPooling2D()
])

################################### CREATING FUNCTIONS ##################################

@app.route('/')
def landing_page():
    return render_template("index.html")

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if request.method == "POST":
        pass

################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    app.run(debug = True)