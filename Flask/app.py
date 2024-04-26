from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import uuid  # To generate unique filenames

app = Flask(__name__)

# Load the TensorFlow Lite model
MODEL_PATH = './model.tflite'
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define a directory to store the uploaded images
UPLOAD_FOLDER = './uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess and predict
def predict(image_path):
    # Load and preprocess the image
    img = Image.open(image_path).resize((150, 150))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Set the tensor and invoke the model
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get the output
    output = interpreter.get_tensor(output_details[0]['index'])
    result = np.argmax(output[0])

    # Define class labels and messages
    labels = {0: 'Glioma Tumour', 1: 'Meningioma Tumour', 2: 'No Tumour', 3: 'Pituitary Tumour'}
    prediction_text = labels[result]

    message = {
        'Glioma Tumour': "Glioma Tumour detected. Consult a specialist for further advice.",
        'Meningioma Tumour': "Meningioma Tumour detected. Consult a specialist for further advice.",
        'No Tumour': "No tumour detected.",
        'Pituitary Tumour': "Pituitary Tumour detected. Consult a specialist for further advice."
    }.get(prediction_text, "Unknown result")

    return prediction_text, message

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = None
    image_path = None  # To store the path of the uploaded image
    if request.method == 'POST':
        imagefile = request.files.get('imagefile')
        if imagefile:
            # Generate a unique filename to avoid conflicts
            filename = str(uuid.uuid4()) + os.path.splitext(imagefile.filename)[1]
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            imagefile.save(image_path)

            # Get the prediction and message
            prediction, message = predict(image_path)

    # Render the index page with optional prediction, message, and image_path
    return render_template('index.html', prediction=prediction, message=message, image_path=image_path)

# Route to serve the uploaded images
@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
