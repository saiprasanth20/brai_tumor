import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained model
model = load_model('path_to_your_model.h5')  # Update with your actual model path

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Read the image
        img = cv2.imread(file_path)

        # Resize the image to (224, 224) to match the model's expected input size
        img = cv2.resize(img, (224, 224))

        # Normalize the image
        img = img / 255.0

        # Add batch dimension (1, 224, 224, 3)
        img = np.expand_dims(img, axis=0)

        # Ensure the image shape is correct before prediction (optional debug print)
        print("Image shape before prediction:", img.shape)

        # Make prediction
        try:
            prediction = model.predict(img)
            # Assuming binary classification (you can modify this based on your model)
            result = 'Tumor' if prediction[0] > 0.5 else 'No Tumor'
            return jsonify({"prediction": result}), 200
        except Exception as e:
            print(e)
            return jsonify({"error": "Prediction failed"}), 500
    else:
        return jsonify({"error": "Invalid file format. Only PNG, JPG, JPEG are allowed."}), 400

if __name__ == '__main__':
    # Make sure to bind to the correct port (usually 10000 in Render)
    app.run(host='0.0.0.0', port=10000)
