from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import xgboost as xgb
import os

# Load model + scaler (same folder la irukkanum)
model = xgb.XGBClassifier()
model.load_model("phishing_xgboost_model.json")
scaler = joblib.load("scaler.pkl")

feature_names = ['length_url','nb_dots','nb_hyphens','nb_at','nb_slash','nb_www','nb_com']

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

app = Flask(__name__)
# CORS allowed — extension ku fetch allow aagum
CORS(app)

@app.route("/")
def home():
    return "Phishing Detection API is Live!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error":"URL is required"}), 400

    feats = extract_features(url)
    df = pd.DataFrame([feats], columns=feature_names)
    X = scaler.transform(df.values)

    proba = model.predict_proba(X)[0]
    ph = float(proba[1]); lg = float(proba[0])

    if ph > 0.85:
        label, conf = "Phishing", ph
    elif lg > 0.84:
        label, conf = "Legitimate", lg
    else:
        label, conf = "Uncertain", max(ph, lg)

    return jsonify({"url": url, "prediction": label, "confidence": round(conf, 4)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
