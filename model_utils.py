import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained ResNet50 model (cached for performance)
@st.cache_resource
def load_model(model_path="resnet50_fish.h5"):
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess the image for ResNet50
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Predict fish category and confidence scores
def predict_fish(model, img_array, class_names):
    preds = model.predict(img_array)
    predicted_index = np.argmax(preds)
    confidence = preds[0][predicted_index] * 100
    return class_names[predicted_index], confidence, preds[0]
