import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from keras.utils import load_img, img_to_array
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import requests

feature_list = np.array(pickle.load(open('model/trained_image_set.pkl', 'rb')))
filenames = pickle.load(open('model/filenames.pkl', 'rb'))
# print(f"filenames :{filenames}")
# trained_model = requests.get('https://fitzy-models.s3.ap-south-1.amazonaws.com/trained_image_set.pkl')
# filenames = requests.get('https://fitzy-models.s3.ap-south-1.amazonaws.com/filenames.pkl')

model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fitzy')


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path, model):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    # print(f"neighbors ------------------------------: {neighbors}")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices


# steps
# file upload -> save
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        # feature extract
        features = feature_extraction(os.path.join(
            "uploads", uploaded_file.name), model)
        st.text(features)
        # recommendation
        indices = recommend(features, feature_list)
        print('indices ::::::::::::::::::::::::::::',indices)
        # show
        col1, col2, col3, col4, col5 = st.beta_columns(5)

        with col1:
            print(f"filenames[0]{filenames[0]},indices[0][0]:{indices[0][0]},filenames[indices[0][0]] :{filenames[indices[0][0]]}")
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])
    else:
        st.header("Some error occurred in file upload")
