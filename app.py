from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from scipy.sparse import hstack

# -----------------------------
# App initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Safe path handling
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# -----------------------------
# Load ML artifacts (EXACT filenames)
# -----------------------------
model = joblib.load(os.path.join(MODEL_DIR, "xgb_expense_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    needs_review = False

    if request.method == "POST":
        merchant = request.form["merchant"]
        description = request.form["description"]
        amount = float(request.form["amount"])
        date = pd.to_datetime(request.form["date"])

        # -------- Feature engineering --------
        text = merchant + " " + description
        log_amount = np.log1p(amount)
        day_of_week = date.dayofweek
        is_weekend = 1 if day_of_week >= 5 else 0
        month = date.month

        # -------- Transform --------
        text_vec = tfidf.transform([text])
        num_vec = scaler.transform([[log_amount, day_of_week, is_weekend, month]])
        final_vec = hstack([text_vec, num_vec])

        # -------- Prediction --------
        probs = model.predict_proba(final_vec)
        pred_idx = np.argmax(probs)
        prediction = label_encoder.inverse_transform([pred_idx])[0]
        confidence = round(float(np.max(probs) * 100), 2)

        if confidence < 60:
            needs_review = True

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        needs_review=needs_review
    )


# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
