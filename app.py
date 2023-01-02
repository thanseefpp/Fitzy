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


################################### APP CREATING ########################################
app = Flask(__name__)

################################### LOADING MODELS ######################################

trained_image_model = 's3://fitzy-models/trained_image_set.pkl'
file_names = 's3://fitzy-models/filenames.pkl'

################################### CREATING FUNCTIONS ##################################

@app.route('/')
def landing_page():
    return render_template("index.html")



################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    app.run(debug = True)