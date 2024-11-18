from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model (make sure the path to the .h5 file is correct)
model = load_model('model/brain_tumor_detector.h5')

# Upload folder for storing images temporarily
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for the image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """Simple endpoint to check if the API is running."""
    return "Brain Tumor Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """Handle file upload and return prediction."""
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = cv2.imread(file_path)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400

        # Resize to (224, 224)
        img = cv2.resize(img, (224, 224))

        # Normalize the image
        img = img / 255.0  # Normalize image to [0, 1]

        # Ensure the image has 3 channels (RGB)
        if img.shape[-1] != 3:
            return jsonify({'error': 'Invalid image format, must have 3 channels (RGB)'}), 400

        # Expand dimensions to match the model's input (batch dimension)
        img = np.expand_dims(img, axis=0)  # Shape becomes (1, 224, 224, 3)

        # Make prediction
        prediction = model.predict(img)
        result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
        probability = prediction[0][0]

        # Return prediction result and probability as JSON
        return jsonify({'prediction': result, 'probability': float(probability)})

    else:
        return jsonify({'error': 'Invalid file type. Allowed formats: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
