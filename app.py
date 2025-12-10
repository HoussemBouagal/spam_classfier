from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===============================
# 📂 Paths
# ===============================
# Define paths to model, tokenizer, and label encoder
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(RESULTS_DIR, "spam-classifier-model.keras")
TOKENIZER_PATH = os.path.join(RESULTS_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(RESULTS_DIR, "label_encoder.pkl")

# ===============================
# 📥 Load Model 
# ===============================
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoder = pickle.load(f)

MAX_LEN = 800  

# ===============================
# 🚀 Flask App
# ===============================
app = Flask(__name__)

def predict_message(message):
    """
    Predict the class (SPAM or HAM) and confidence score of a message.
    """
    # Convert text to sequences and pad
    seq = tokenizer.texts_to_sequences([message])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    # Make prediction
    pred = model.predict(padded)
    pred_idx = np.argmax(pred, axis=1)[0]

    # Map numerical labels to text labels
    label_map = {0: "HAM", 1: "SPAM"}
    prediction = label_map.get(pred_idx, "UNKNOWN")

    # Calculate confidence score
    confidence = float(np.max(pred)) * 100
    return prediction, confidence

@app.route("/", methods=["GET", "POST"])
def home():
    """
    Render the main page with a form to classify a message.
    """
    prediction, confidence = "", 0
    if request.method == "POST":
        message = request.form["message"]
        if message.strip():
            prediction, confidence = predict_message(message)
            print(f"✅ Prediction: {prediction} ({confidence:.2f}%)")
    return render_template("index.html", prediction=prediction, confidence=confidence)

# ✅ API Endpoint (JSON response)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint to classify a message via JSON request.
    """
    data = request.get_json()
    message = data.get("message", "")
    if not message:
        return jsonify({"error": "Message is required"}), 400
    
    prediction, confidence = predict_message(message)
    return jsonify({"prediction": prediction, "confidence": f"{confidence:.2f}%"})

if __name__ == "__main__":
    app.run(debug=True)
