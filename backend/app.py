from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Load Model
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)
# -------------------------------
# Home Route
# -------------------------------
@app.route("/")
def home():
    return jsonify({
        "message": "SmartHealth API is running!"
    })

# -------------------------------
# Prediction Route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        df = pd.DataFrame([data])

        # Remove spaces from column names
        df.columns = df.columns.str.strip()

        prediction = model.predict(df)[0]

        return jsonify({
            "risk": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -------------------------------
# Run Locally
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)