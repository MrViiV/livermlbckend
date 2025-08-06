from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load model and scaler
model = joblib.load("xgb_model.pkl")  
scaler = joblib.load("scaler.pkl") 

feature_names = [
    "Age", "Gender", "BMI", "AlcoholConsumption", "Smoking", "GeneticRisk",
    "PhysicalActivity", "Diabetes", "Hypertension", "LiverFunctionTest"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert categorical inputs to numerical
        gender = 1 if data["Gender"] == "Male" else 0
        smoking = 1 if data["Smoking"] == "Yes" else 0
        diabetes = 1 if data["Diabetes"] == "Yes" else 0
        hypertension = 1 if data["Hypertension"] == "Yes" else 0
        genetic_risk = {"Low": 0, "Medium": 1, "High": 2}[data["GeneticRisk"]]

        # Numeric features (assume already float/int)
        age = float(data["Age"])
        bmi = float(data["BMI"])
        alcohol = float(data["AlcoholConsumption"])
        activity = float(data["PhysicalActivity"])
        liver_test = float(data["LiverFunctionTest"])

        # Feature vector in model training order
        input_features = [
            age, gender, bmi, alcohol, smoking,
            genetic_risk, activity, diabetes, hypertension, liver_test
        ]

        input_array = np.array([input_features])  # Shape: (1, 10)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)

        return jsonify({
            "result": str(prediction[0])  # Can also map to labels here if needed
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
