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
    try:
        data = request.json

        df = pd.DataFrame([data])

        # fix column names (important)
        df.columns = df.columns.str.strip()

        prediction = model.predict(df)[0]

        return jsonify({
            "risk": float(prediction)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
