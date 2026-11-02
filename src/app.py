"""Flask API serving the credit-risk model with Prometheus metrics."""
import time
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

MODEL_PATH = Path("models/model.joblib")
model = joblib.load(MODEL_PATH)
FEATURES = [c for c in model.feature_names_in_ if c != "credit_per_month"]

REQUESTS = Counter("app_requests_total", "Total HTTP requests", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions by class", ["label"])
ERRORS = Counter("prediction_errors_total", "Failed prediction requests")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")

app = Flask(__name__)


_pre = model.named_steps["pre"]
_cat = dict(zip(_pre.transformers_[1][2], _pre.named_transformers_["cat"].categories_))
CAT_OPTIONS = {c: [str(v) for v in vals] for c, vals in _cat.items()}
NUM_FIELDS = [f for f in FEATURES if f not in CAT_OPTIONS]


@app.get("/")
def home():
    REQUESTS.labels("/").inc()
    return render_template("index.html", cat_options=CAT_OPTIONS, num_fields=NUM_FIELDS)


@app.get("/api")
def api_info():
    return jsonify(service="credit-risk-api", endpoints=["/health", "/predict", "/metrics"])


@app.get("/health")
def health():
    REQUESTS.labels("/health").inc()
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    REQUESTS.labels("/predict").inc()
    start = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in payload]
    if missing:
        ERRORS.inc()
        return jsonify(error="missing fields", fields=missing), 400
    try:
        row = pd.DataFrame([{f: payload[f] for f in FEATURES}])
        row["credit_per_month"] = float(row["credit_amount"][0]) / float(row["duration"][0])
        proba = float(model.predict_proba(row)[0, 1])
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        ERRORS.inc()
        return jsonify(error=str(exc)), 400
    label = "bad" if proba >= 0.5 else "good"
    PREDICTIONS.labels(label).inc()
    LATENCY.observe(time.perf_counter() - start)
    return jsonify(risk=label, probability_bad=round(proba, 4))


@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
