from flask import Flask, render_template, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

# ---------- MODEL PATH ----------
# Update this path to your model location
MODEL_PATH = "./asl_CNNalgorithm.keras"
# --------------------------------

# Check if model exists
if not os.path.exists(MODEL_PATH):
    print(f"⚠️  Warning: Model not found at {MODEL_PATH}")
    print("Please update the MODEL_PATH variable with the correct path to your trained model.")
    model = None
else:
    # Load trained ASL CNN model
    try:
        model = keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

# Define labels (A–Z)
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Model expects 64x64 RGB images
IMG_SIZE = (64, 64)

def preprocess_image(image_bytes):
    """Convert image bytes to a normalized NumPy array."""
    try:
        # Open and convert image to RGB
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to expected size
        img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize to 0-1 range
        arr = np.array(img, dtype=np.float32) / 255.0
        
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
        
        return arr
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise

@app.route("/")
def index():
    """Serve the main application page."""
    # Use the upload-only template
    return render_template("index_upload_only.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload and return prediction."""
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            "error": "Model not loaded. Please check the model path."
        }), 500
    
    # Check for file upload
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read image bytes
        image_bytes = file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get the predicted class
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        
        # Get the letter
        predicted_letter = LABELS[predicted_idx] if predicted_idx < len(LABELS) else str(predicted_idx)
        
        # Get top 3 predictions for more detail
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                "label": LABELS[idx] if idx < len(LABELS) else str(idx),
                "confidence": float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        print(f"✅ Prediction: {predicted_letter} (confidence: {confidence:.2%})")
        
        return jsonify({
            "label": predicted_letter,
            "index": predicted_idx,
            "confidence": confidence,
            "top_predictions": top_predictions
        })
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_count": len(LABELS)
    })

if __name__ == "__main__":
    # Configuration
    PORT = 5000
    DEBUG = False  # Set to True for development
    
    print("\n" + "="*50)
    print("🤟 SignSpeak ASL Recognition Server")
    print("="*50)
    print(f"📍 Server URL: http://localhost:{PORT}")
    print(f"📚 Classes: {len(LABELS)} letters (A-Z)")
    if model:
        print("✅ Model loaded and ready")
    else:
        print("⚠️  Model not loaded - predictions will fail")
    print("="*50 + "\n")
    
    # Start Flask server
    app.run(host="127.0.0.1", port=PORT, debug=DEBUG)