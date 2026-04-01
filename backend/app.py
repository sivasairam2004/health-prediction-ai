from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load all models and scalers
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "..", "models")

def load(name):
    path = os.path.join(MODELS, name)
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    diabetes_model    = load("diabetes_model.pkl")
    diabetes_scaler   = load("diabetes_scaler.pkl")
    heart_model       = load("heart_model.pkl")
    heart_scaler      = load("heart_scaler.pkl")
    parkinsons_model  = load("parkinsons_model.pkl")
    parkinsons_scaler = load("parkinsons_scaler.pkl")
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

@app.route("/")
def home():
    return jsonify({"message": "Health Prediction AI is running!"})

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)
        features = [[
            float(data.get("Pregnancies", 0)),
            float(data.get("Glucose", 0)),
            float(data.get("BloodPressure", 0)),
            float(data.get("SkinThickness", 0)),
            float(data.get("Insulin", 0)),
            float(data.get("BMI", 0)),
            float(data.get("DiabetesPedigreeFunction", 0)),
            float(data.get("Age", 0))
        ]]
        scaled = diabetes_scaler.transform(features)
        prediction = int(diabetes_model.predict(scaled)[0])
        probability = diabetes_model.predict_proba(scaled)[0]
        return jsonify({
            "prediction": prediction,
            "result": "Diabetic" if prediction == 1 else "Not Diabetic",
            "confidence": round(float(max(probability)) * 100, 2)
        })
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 400

@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    try:
        data = request.get_json(force=True)
        features = [[
            float(data.get("age", 0)),
            float(data.get("sex", 0)),
            float(data.get("cp", 0)),
            float(data.get("trestbps", 0)),
            float(data.get("chol", 0)),
            float(data.get("fbs", 0)),
            float(data.get("restecg", 0)),
            float(data.get("thalach", 0)),
            float(data.get("exang", 0)),
            float(data.get("oldpeak", 0)),
            float(data.get("slope", 0)),
            float(data.get("ca", 0)),
            float(data.get("thal", 0))
        ]]
        scaled = heart_scaler.transform(features)
        prediction = int(heart_model.predict(scaled)[0])
        probability = heart_model.predict_proba(scaled)[0]
        return jsonify({
            "prediction": prediction,
            "result": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "confidence": round(float(max(probability)) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/predict/parkinsons", methods=["POST"])
def predict_parkinsons():
    try:
        data = request.get_json(force=True)
        features = [[
            float(data.get("fo", 0)),
            float(data.get("fhi", 0)),
            float(data.get("flo", 0)),
            float(data.get("jitter", 0)),
            float(data.get("shimmer", 0)),
            float(data.get("nhr", 0)),
            float(data.get("hnr", 0)),
            float(data.get("rpde", 0)),
            float(data.get("dfa", 0)),
            float(data.get("spread1", 0)),
            float(data.get("spread2", 0)),
            float(data.get("d2", 0)),
            float(data.get("ppe", 0))
        ]]
        scaled = parkinsons_scaler.transform(features)
        prediction = int(parkinsons_model.predict(scaled)[0])
        probability = parkinsons_model.predict_proba(scaled)[0]
        return jsonify({
            "prediction": prediction,
            "result": "Parkinson's Detected" if prediction == 1 else "No Parkinson's",
            "confidence": round(float(max(probability)) * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)