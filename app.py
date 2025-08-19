from flask import Flask, request, jsonify
import pandas as pd
import joblib
import xgboost as xgb
import os
from flask_cors import CORS

# === Load model and scaler ===
model = xgb.XGBClassifier()
model.load_model("phishing_xgboost_model.json")   # file repo root la irukkanum
scaler = joblib.load("scaler.pkl")

# === Feature names ===
feature_names = ['length_url', 'nb_dots', 'nb_hyphens', 'nb_at', 'nb_slash', 'nb_www', 'nb_com']

# === Feature extraction function ===
def extract_features(url):
    url = str(url)
    return {
        'length_url': len(url),
        'nb_dots': url.count('.'),
        'nb_hyphens': url.count('-'),
        'nb_at': 1 if '@' in url else 0,
        'nb_slash': url.count('/'),
        'nb_www': 1 if 'www' in url else 0,
        'nb_com': 1 if '.com' in url else 0
    }

# === Flask App ===
app = Flask(__name__)
CORS(app)  # allow API calls from extension/browser

@app.route("/")
def home():
    return "✅ Phishing Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        url = data.get("url", "").strip()

        if not url:
            return jsonify({"error": "URL is required"}), 400

        # Extract & scale
        features = extract_features(url)
        df_features = pd.DataFrame([features], columns=feature_names)
        scaled_features = scaler.transform(df_features.values)

        # Predict
        proba = model.predict_proba(scaled_features)[0]
        phishing_confidence = proba[1]
        legit_confidence = proba[0]

        # Custom threshold
        if phishing_confidence > 0.85:
            label = "Phishing"
            confidence = phishing_confidence
        elif legit_confidence > 0.84:
            label = "Legitimate"
            confidence = legit_confidence
        else:
            label = "Uncertain"
            confidence = max(phishing_confidence, legit_confidence)

        return jsonify({
            "url": url,
            "prediction": label,
            "confidence": round(float(confidence), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
