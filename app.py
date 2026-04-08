from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)

# Load class indices
class_indices = {"0": "Bacterial Leaf Blight", "1": "Brown Spot", "2": "Healthy Rice Leaf", 
                 "3": "Leaf Blast", "4": "Leaf scald", "5": "Narrow Brown Leaf Spot", 
                 "6": "Neck_Blast", "7": "Rice Hispa", "8": "Sheath Blight"}
class_names = {int(k): v for k, v in class_indices.items()}

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='rice_model_mobilenetv2.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug: Print model input and output details
print("=== Model Input Details ===")
print(input_details)
print("=== Model Output Details ===")
print(output_details)

# Configuration
UPLOAD_FOLDER = 'static/upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
current_prediction = {}  # Store current prediction

# Updated dictionaries for medicines and fertilizers (mapped to new class labels)
medicines = {
    'Bacterial Leaf Blight': [
        "Copper oxychloride (fungicide)",
        "Streptomycin (antibiotic)",
        "Plant growth regulators (PGRs)"
    ],
    'Brown Spot': [
        "Carbendazim (fungicide)",
        "Tricyclazole (fungicide)",
        "Tebuconazole (fungicide)"
    ],
    'Leaf Blast': [
        "Tricyclazole (fungicide)",
        "Isoprothiolane (fungicide)",
        "Propiconazole (fungicide)"
    ],
    'Leaf scald': [
        "Mancozeb (fungicide)",
        "Carbendazim (fungicide)",
        "Copper-based fungicides"
    ],
    'Narrow Brown Leaf Spot': [
        "Propiconazole (fungicide)",
        "Mancozeb (fungicide)",
        "Tebuconazole (fungicide)"
    ],
    'Neck_Blast': [
        "Tricyclazole (fungicide)",
        "Isoprothiolane (fungicide)",
        "Azoxystrobin (fungicide)"
    ],
    'Rice Hispa': [
        "Chlorpyrifos (insecticide)",
        "Malathion (insecticide)",
        "Neem-based insecticides"
    ],
    'Sheath Blight': [
        "Validamycin (fungicide)",
        "Hexaconazole (fungicide)",
        "Carbendazim (fungicide)"
    ],
    'Healthy Rice Leaf': []  # No medicines for healthy leaf
}

fertilizers = {
    'Bacterial Leaf Blight': [
        "Nitrogen-rich fertilizers (e.g., Urea)",
        "Potassium fertilizers (e.g., Muriate of Potash)"
    ],
    'Brown Spot': [
        "Balanced NPK fertilizers",
        "Organic fertilizers (e.g., compost)"
    ],
    'Leaf Blast': [
        "Silicate fertilizers (e.g., Potassium silicate)",
        "Nitrogen management"
    ],
    'Leaf scald': [
        "Phosphorus fertilizers (e.g., Superphosphate)",
        "Organic matter"
    ],
    'Narrow Brown Leaf Spot': [
        "Balanced NPK fertilizers",
        "Potassium fertilizers"
    ],
    'Neck_Blast': [
        "Silicate fertilizers",
        "Organic compost"
    ],
    'Rice Hispa': [
        "Nitrogen fertilizers (e.g., Urea)",
        "Organic manure"
    ],
    'Sheath Blight': [
        "Potassium fertilizers",
        "Organic matter to improve soil health"
    ],
    'Healthy Rice Leaf': []  # No fertilizers for healthy leaf
}

# Preprocessing function exactly as in training code
def preprocess_image(image_path):
    IMG_SIZE = (224, 224)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(img_array, axis=0))
    return img_array

# Updated prediction function without confidence threshold
def predict_disease(image_path):
    # Debug: Print image path
    print(f"\n=== Predicting disease for image: {image_path} ===")

    # Preprocess image using the exact training method
    input_data = preprocess_image(image_path)
    print("Image preprocessed successfully.")

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    print("Input tensor set successfully.")

    # Run inference
    interpreter.invoke()
    print("Inference completed successfully.")

    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    print("Raw predictions:", predictions)

    # Process predictions
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx] * 100
    predicted_label = class_names[predicted_class_idx]

    # Debug: Print prediction details
    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted label: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")

    # Return prediction results
    recommended_medicines = medicines.get(predicted_label, [])
    recommended_fertilizers = fertilizers.get(predicted_label, [])
    return predicted_label, confidence, recommended_medicines, recommended_fertilizers

# Weather-based disease mapping (unchanged)
def get_weather_diseases(weather):
    weather_diseases = {
        'Sunny': ['Brown Spot', 'Leaf Blast'],
        'Rainy': ['Bacterial Leaf Blight', 'Sheath Blight'],
        'Cloudy': ['Bacterial Leaf Blight', 'Leaf scald'],
        'Windy': ['Brown Spot', 'Narrow Brown Leaf Spot']
    }
    return weather_diseases.get(weather, [])

# Main route (unchanged structure)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "message": "No file uploaded.",
                "image_path": None,
                "weather": None,
                "date": datetime.now().strftime("%A, %B %d, %Y, %I %p IST")
            }), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({
                "success": False,
                "message": "No file selected.",
                "image_path": None,
                "weather": None,
                "date": datetime.now().strftime("%A, %B %d, %Y, %I %p IST")
            }), 400

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            weather = request.form.get('weather', 'Unknown')
            print(f"Weather condition: {weather}")

            # Predict disease
            predicted_label, confidence, recommended_medicines, recommended_fertilizers = predict_disease(image_path)
            weather_diseases = get_weather_diseases(weather)

            global current_prediction
            current_prediction = {
                'image_path': f"http://192.168.1.33:5000/{image_path}",
                'disease': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'medicines': recommended_medicines,
                'fertilizers': recommended_fertilizers,
                'weather': weather,
                'weather_diseases': weather_diseases
            }

            print("Prediction results stored in current_prediction.")

            return jsonify({
                "success": True,
                "message": "Successfully submitted.",
                "image_path": current_prediction['image_path'],
                "weather": weather,
                "date": datetime.now().strftime("%A, %B %d, %Y, %I %p IST")
            })

    return render_template('index.html')

# Existing API endpoints (unchanged)
@app.route('/results', methods=['GET'])
def results():
    return jsonify(current_prediction)

@app.route('/results/image', methods=['GET'])
def get_image():
    return jsonify({"image_path": current_prediction.get('image_path', None)})

@app.route('/results/disease', methods=['GET'])
def get_disease():
    return jsonify({"disease": current_prediction.get('disease', None)})

@app.route('/results/confidence', methods=['GET'])
def get_confidence():
    return jsonify({"confidence": current_prediction.get('confidence', None)})

@app.route('/results/medicines', methods=['GET'])
def get_medicines():
    return jsonify({"medicines": current_prediction.get('medicines', [])})

@app.route('/results/fertilizers', methods=['GET'])
def get_fertilizers():
    return jsonify({"fertilizers": current_prediction.get('fertilizers', [])})

@app.route('/results/weather', methods=['GET'])
def get_weather():
    return jsonify({"weather": current_prediction.get('weather', None)})

@app.route('/results/weather_diseases', methods=['GET'])
def get_weather_diseases_result():
    return jsonify({"weather_diseases": current_prediction.get('weather_diseases', [])})

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 204 No Content

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)