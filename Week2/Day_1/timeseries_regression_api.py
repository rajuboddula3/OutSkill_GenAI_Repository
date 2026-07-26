"""
Flask REST API for the AirPassengers sliding-window regression model.

Loads the TabPFN model + StandardScaler produced by
`timeseries_regression_solution.ipynb` and predicts next month's passenger
count from the previous 6 months.

Run:
    cd Week2
    ./.venv/bin/python Day_1/timeseries_regression_api.py          # port 5001
    PORT=8080 ./.venv/bin/python Day_1/timeseries_regression_api.py

Example:
    curl -X POST http://localhost:5001/predict \
         -H "Content-Type: application/json" \
         -d '{"passengers": [417, 391, 419, 461, 472, 535]}'
"""
import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "tabpfn_timeseries_regression_model.pkl"
SCALER_PATH = BASE_DIR / "timeseries_regression_scaler.pkl"
META_PATH = BASE_DIR / "timeseries_regression_metadata.json"

app = Flask(__name__)

# ---- load artifacts once at startup --------------------------------------
_missing = [p.name for p in (MODEL_PATH, SCALER_PATH, META_PATH) if not p.exists()]
if _missing:
    raise FileNotFoundError(
        f"Missing artifact(s): {', '.join(_missing)}. "
        "Run timeseries_regression_solution.ipynb first."
    )

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
METADATA = json.loads(META_PATH.read_text())

WINDOW_SIZE = METADATA["window_size"]
FEATURE_NAMES = METADATA["feature_names"]


def _validate(payload):
    """Return (values, None) on success or (None, (json_error, status))."""
    if not isinstance(payload, dict):
        return None, ({"error": "Request body must be a JSON object."}, 400)

    if "passengers" not in payload:
        return None, ({"error": "Missing required field 'passengers'.",
                       "expected": f"a list of {WINDOW_SIZE} numbers, oldest -> newest"}, 400)

    values = payload["passengers"]
    if not isinstance(values, (list, tuple)):
        return None, ({"error": "'passengers' must be a list of numbers.",
                       "received_type": type(values).__name__}, 400)

    if len(values) != WINDOW_SIZE:
        return None, ({"error": f"'passengers' must contain exactly {WINDOW_SIZE} values.",
                       "received_length": len(values)}, 400)

    clean = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, ({"error": f"Value at index {i} is not numeric.",
                           "value": v}, 400)
        v = float(v)
        if not math.isfinite(v):
            return None, ({"error": f"Value at index {i} is not finite (NaN/inf)."}, 400)
        if v < 0:
            return None, ({"error": f"Value at index {i} is negative; "
                                    "passenger counts must be >= 0.", "value": v}, 400)
        clean.append(v)

    return clean, None


@app.get("/")
def index():
    return jsonify({
        "service": "AirPassengers next-month forecaster",
        "model": METADATA["model"],
        "window_size": WINDOW_SIZE,
        "feature_order": METADATA["feature_order"],
        "trained_on": METADATA["train_period"],
        "test_r2": round(METADATA["metrics"]["test"]["R2"], 4),
        "endpoints": {
            "GET /health": "health check",
            "POST /predict": {
                "body": {"passengers": [f"<{WINDOW_SIZE} numbers, oldest -> newest>"]},
                "returns": {"predicted_next_month": "<float>"},
            },
        },
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None,
                    "scaler_loaded": SCALER is not None,
                    "window_size": WINDOW_SIZE})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Body must be valid JSON with "
                                 "Content-Type: application/json."}), 400

    values, err = _validate(payload)
    if err is not None:
        body, status = err
        return jsonify(body), status

    try:
        frame = pd.DataFrame([values], columns=FEATURE_NAMES)
        prediction = float(MODEL.predict(SCALER.transform(frame))[0])
    except Exception as exc:                       # inference failure
        app.logger.exception("prediction failed")
        return jsonify({"error": "Prediction failed.", "detail": str(exc)}), 500

    last = values[-1]
    return jsonify({
        "input_passengers": values,
        "predicted_next_month": round(prediction, 2),
        "change_vs_last_month": round(prediction - last, 2),
        "change_vs_last_month_pct": round((prediction / last - 1) * 100, 2) if last else None,
        "window_size": WINDOW_SIZE,
        "model": METADATA["model"],
    })


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found.",
                    "available": ["GET /", "GET /health", "POST /predict"]}), 404


@app.errorhandler(405)
def not_allowed(_):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))   # 5000 is taken by AirPlay on macOS
    app.run(host="0.0.0.0", port=port, debug=False)
