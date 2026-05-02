from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("../model/model.pkl")

@app.route("/")
def home():
    return "SmartHealth API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]
    prediction = max(0, min(100, prediction))

    return jsonify({
        "health_risk": round(prediction, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)