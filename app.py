################################## IMPORTING LIBRARIES ##################################

from flask import Flask, request, redirect, render_template, jsonify
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from keras.applications.resnet import ResNet50,preprocess_input
from keras.layers import GlobalMaxPooling2D
# from numpy.linalg import norm
import os
import pickle
from sklearn.neighbors import NearestNeighbors
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
    """
        This Function is will allow only the images with the same data type that we have specified.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(uploaded_file):
    """
        This Function is used to store the file that user uploaded
        'static/uploads' files keep in this folder
    """
    try:
        with open(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0
    
def extract_features(img_path, model):
    """
        Here we take the image that has been uploaded by the user.
        1. load_img - Loads an image into PIL format with the target size(it can be changed)
        2. img_to_array - converting the img to an array
        3. expand_dims - Expand the shape of an array
        4. preprocess_input - The images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling. (encoding a batch of images)
        5. model.predict - Resnet50 model that has been used to predict the result, "flattening adds an extra channel dimension and output shape is (batch, 1)"
        6. normalizing the result - This function can compute several different vector norms (the 1-norm, the Euclidean or 2-norm, the inf-norm, and in general the p-norm for p > 0) and matrix norms (Frobenius, 1-norm, 2-norm and inf-norm). Explicitly supports 'euclidean' norm as the default, including for higher order tensors
    """
    img = load_img(img_path, target_size=(224, 224, 3))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / tf.norm(result)
    return normalized_result

def get_nearest_neighbors(features, feature_list,number_of_recommendation):
    """
        Here using NearestNeighbors algorithm to find the nearest features with our feature_list that we have already trained with 44k images.
        "Returns indices of and distances to the neighbors of each point"
    """
    neighbors = NearestNeighbors(n_neighbors=number_of_recommendation, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

@app.route('/')
def landing_page():
    """
        landing page upload image form will appear on this screen.
    """
    return render_template("index.html")

@app.route('/', methods=['POST'])
def upload_image():
    """
        Main Function 
        -------------
        Uploaded image,range that choose by user both data will taken by this function.
        1. Saving the image to uploads folder function calling.
        2. Creating the list to store the image path to server image from s3( for that i have to take the name in array).
        3. Feature extract function calling
        4. Finding the nearest features
        5. Appending the featured image from the filename model, there we have stored the path of the images while training the model.
        6. return response will server to response.html file
    """
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
                filenames_path.append(filenames[nearest_neighbours_indices[0][count]])
            msg = 'Successfully Uploaded'
        else:
            msg = 'Invalid Upload only png, jpg, jpeg'
    return jsonify({'success_response': render_template('response.html', msg=msg,filename=filename, filesPathList=filenames_path)})

################################### EXECUTE APPLICATION #################################

if __name__ == "__main__":
    app.run(debug = False)