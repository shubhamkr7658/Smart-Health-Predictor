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
        print("Received:", data)

        df = pd.DataFrame([data])

        # FIX column spacing issue
        df.columns = df.columns.str.strip()

        prediction = model.predict(df)[0]
        prediction = max(0, min(100, prediction))

        return jsonify({
            "risk": round(prediction, 2)
        })

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
