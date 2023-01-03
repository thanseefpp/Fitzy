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
        with open(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
def extract_features(img_path, model):
    img = load_img(img_path, target_size=(224, 224, 3))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def get_nearest_neighbors(features, feature_list,number_of_recommendation):
    neighbors = NearestNeighbors(n_neighbors=number_of_recommendation, algorithm='brute', metric='euclidean')
    # print(f"neighbors ------------------------------: {neighbors}")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    # print(f'distance :{distances},\nindices :{indices}')
    return indices

@app.route('/')
def landing_page():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
    if 'uploadFile' not in request.files:
        return redirect(request.url)
    user_range = request.form['MyRange']
    filenames_path = []
    file = request.files.get('uploadFile')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        img_saved = save_uploaded_file(file)
        if img_saved != 0:
            #image feature extract using Resnet50 model
            feature_extracted = extract_features(os.path.join(app.config['UPLOAD_FOLDER'], file.filename), model)
            nearest_neighbours_indices = get_nearest_neighbors(features=feature_extracted,feature_list=feature_list,number_of_recommendation=int(user_range))
            for count in range(0,int(user_range)):
                # print(count)
                filenames_path.append(filenames[nearest_neighbours_indices[0][count]])
            # print(f"filenames_path : {filenames_path}")
            msg = 'Successfully Uploaded'
        else:
            msg = 'Invalid Upload only png, jpg, jpeg'
    return jsonify({'success_response': render_template('response.html', msg=msg,filename=filename, filesPathList=filenames_path)})

################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    app.run(debug = True)