from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image
import joblib
import os
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Load models and scaler
RF_MODEL_PATH = 'random_forest_hb_model.pkl'  # Path to the Random Forest model
SCALER_PATH = 'scalerRF.pkl'  # Path to the scaler file
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to extract RGB intensity percentages using PIL
def extract_rgb_intensity_pil(image_path):
    image = Image.open(image_path).convert("RGB")
    pixels = np.array(image)

    # Flatten the pixel values
    R, G, B = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    total_pixels = pixels.shape[0] * pixels.shape[1]

    # Calculate the mean intensity for each channel
    R_intensity = np.sum(R) / total_pixels
    G_intensity = np.sum(G) / total_pixels
    B_intensity = np.sum(B) / total_pixels

    # Calculate the total intensity and percentages
    total_intensity = R_intensity + G_intensity + B_intensity
    r_percent = (R_intensity / total_intensity) * 100
    g_percent = (G_intensity / total_intensity) * 100
    b_percent = (B_intensity / total_intensity) * 100

    return np.array([r_percent, g_percent, b_percent]).reshape(1, -1), (R_intensity, G_intensity, B_intensity)

# Function to convert image to base64
def convert_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        try:
            # Extract RGB intensity percentages using PIL
            rgb_percentages, raw_rgb_values = extract_rgb_intensity_pil(file_path)

            # Apply scaling to the extracted RGB percentages
            inputs = scaler.transform(rgb_percentages)

            # Predict hemoglobin level using the Random Forest model
            hb_prediction = rf_model.predict(inputs)[0]

            # Determine anemia status
            status = "Anemic" if hb_prediction < 11 else "Non Anemic"

            # Convert uploaded image to base64
            image = Image.open(file_path)
            image_base64 = convert_image_to_base64(image)

            # Set color for anemia status (green for non-anemic, red for anemic)
            status_color = "green" if status == "Non Anemic" else "red"

            # Render the results in the webpage
            return render_template(
                'result.html',
                uploaded_image=image_base64,
                hb_level=f"{hb_prediction:.2f} g/dL",
                status=status,
                status_color=status_color
            )

        except Exception as e:
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        finally:
            # Clean up temporary files if necessary
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
