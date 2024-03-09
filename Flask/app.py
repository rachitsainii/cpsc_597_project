from flask import Flask, render_template, request
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
from keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image


app = Flask(__name__)

# Defining the model path
MODEL_PATH = './models/efficient_net_b0.h5'

# Loading the trained model
model = load_model(MODEL_PATH)
print('Model Loaded.')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150,150))

    #Preprocessing the image
    x = image.img_to_array(img)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(port=3000, debug=True)