from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model (make sure the path to the .h5 file is correct)
try:
    model = load_model('model/brain_tumor_detector.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

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
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the image
            img = cv2.imread(file_path)
            if img is None:
                return jsonify({'error': 'Invalid image file'}), 400

            img = cv2.resize(img, (224, 224))  # Resize to match model input size
            img = img / 255.0  # Normalize image
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(img)
            result = "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"
            probability = prediction[0][0]

            # Return prediction result and probability as JSON
            return jsonify({'prediction': result, 'probability': float(probability)})

        except Exception as e:
            print(f"Error processing the image or prediction: {e}")
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'error': 'Invalid file type. Allowed formats: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
