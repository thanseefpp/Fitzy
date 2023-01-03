################################## IMPORTING LIBRARIES ##################################

from flask import Flask, request, redirect, url_for, render_template, jsonify
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
from werkzeug.utils import secure_filename

################################### APP CONFIGURATIONS ########################################

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "fitzy-apparels-recommendation"
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

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

################################### FUNCTIONS ###########################################

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('static/uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

@app.route('/')
def landing_page():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
    if 'uploadFile' not in request.files:
        return redirect(request.url)
    files = request.files.get('uploadFile')
    if files and allowed_file(files.filename):
        filename = secure_filename(files.filename)
        files.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        msg = 'File successfully uploaded to /static/uploads!'
    else:
        msg = 'Invalid Upload only png, jpg, jpeg, gif'
    return jsonify({'success_response': render_template('response.html', msg=msg, filename=filename)})

################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    app.run(debug = True)