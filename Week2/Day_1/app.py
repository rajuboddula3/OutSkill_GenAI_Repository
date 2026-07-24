"""
Air Passengers Forecasting REST API
====================================

A Flask service that serves the TabPFNRegressor model trained in
`air_passengers_analysis.ipynb`. Time series forecasting is framed as a
supervised regression problem using a sliding window of 6 lag features.

The client sends the last 6 months of passenger counts (oldest -> newest)
and the service returns the predicted passenger count for the next month.

Run:
    python app.py
    # or
    flask --app app run --port 5001

Example:
    curl -X POST http://localhost:5001/predict \
         -H "Content-Type: application/json" \
         -d '{"passengers": [417, 391, 419, 461, 472, 535]}'
"""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template_string, request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "tabpfn_air_passengers_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Number of lag features the model was trained on (window_size in the notebook).
WINDOW_SIZE = 6

app = Flask(__name__)


# ---------------------------------------------------------------------------
# HTML docs page (served at GET /docs) with a live test form for /predict.
# ---------------------------------------------------------------------------
DOCS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Air Passengers Forecasting API — Docs</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
           Helvetica, Arial, sans-serif; max-width: 760px; margin: 2rem auto;
           padding: 0 1rem; line-height: 1.55; }
    h1 { margin-bottom: .2rem; }
    .sub { color: #888; margin-top: 0; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
    pre { background: rgba(127,127,127,.12); padding: .8rem 1rem;
          border-radius: 8px; overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { text-align: left; padding: .5rem .6rem;
             border-bottom: 1px solid rgba(127,127,127,.3); }
    .method { font-weight: 700; }
    .get { color: #1a7f37; } .post { color: #9a6700; }
    .card { border: 1px solid rgba(127,127,127,.3); border-radius: 10px;
            padding: 1.2rem; margin: 1.5rem 0; }
    label { font-weight: 600; display: block; margin-bottom: .4rem; }
    input[type=text] { width: 100%; padding: .6rem; font-size: 1rem;
            border-radius: 6px; border: 1px solid rgba(127,127,127,.5);
            background: transparent; color: inherit; }
    button { margin-top: .8rem; padding: .6rem 1.2rem; font-size: 1rem;
             border: 0; border-radius: 6px; background: #2563eb; color: #fff;
             cursor: pointer; }
    button:hover { background: #1d4ed8; }
    #result { margin-top: 1rem; white-space: pre-wrap; }
    .hint { color: #888; font-size: .9rem; }
  </style>
</head>
<body>
  <h1>Air Passengers Forecasting API</h1>
  <p class="sub">TabPFNRegressor · sliding-window regression · window size =
     {{ window_size }}</p>

  <h2>Endpoints</h2>
  <table>
    <tr><th>Method</th><th>Path</th><th>Description</th></tr>
    <tr><td class="method get">GET</td><td><code>/</code></td>
        <td>JSON service description</td></tr>
    <tr><td class="method get">GET</td><td><code>/docs</code></td>
        <td>This page</td></tr>
    <tr><td class="method get">GET</td><td><code>/health</code></td>
        <td>Health check</td></tr>
    <tr><td class="method post">POST</td><td><code>/predict</code></td>
        <td>Predict next month from the last {{ window_size }} months</td></tr>
  </table>

  <h2>POST /predict</h2>
  <p>Send exactly {{ window_size }} numeric, non-negative values ordered
     <strong>oldest &rarr; newest</strong>.</p>
  <pre>curl -X POST http://localhost:5001/predict \\
     -H "Content-Type: application/json" \\
     -d '{"passengers": [417, 391, 419, 461, 472, 535]}'</pre>

  <div class="card">
    <h3>Try it</h3>
    <label for="vals">Last {{ window_size }} months (comma-separated,
      oldest &rarr; newest)</label>
    <input id="vals" type="text" value="417, 391, 419, 461, 472, 535">
    <p class="hint">Example above uses real values from the dataset.</p>
    <button onclick="runPredict()">Predict next month</button>
    <div id="result"></div>
  </div>

  <script>
    async function runPredict() {
      const out = document.getElementById('result');
      const raw = document.getElementById('vals').value;
      const passengers = raw.split(',')
        .map(s => s.trim()).filter(s => s.length)
        .map(Number);
      out.textContent = 'Predicting...';
      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({passengers})
        });
        const data = await res.json();
        if (res.ok) {
          out.textContent =
            'Predicted next month: ' + data.predicted_next_month +
            ' passengers\\n\\n' + JSON.stringify(data, null, 2);
        } else {
          out.textContent = 'Error (' + res.status + '): ' + data.error;
        }
      } catch (e) {
        out.textContent = 'Request failed: ' + e;
      }
    }
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Load model + scaler once at startup
# ---------------------------------------------------------------------------
def _load_artifacts():
    """Load the trained model and the fitted StandardScaler from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = _load_artifacts()
    print(f"Loaded model from {MODEL_PATH}")
    print(f"Loaded scaler from {SCALER_PATH}")
except FileNotFoundError as exc:
    # Fail loudly at startup rather than on the first request.
    raise RuntimeError(
        "Could not load model/scaler artifacts. Run the notebook "
        "`air_passengers_analysis.ipynb` first to generate them."
    ) from exc


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_passengers(payload):
    """
    Validate the request body and return a clean list of floats.

    The features must be ordered oldest -> newest so they line up with the
    model's lag columns [lag_6, lag_5, lag_4, lag_3, lag_2, lag_1], where
    lag_6 is 6 months ago and lag_1 is the most recent month.

    Raises ValueError with a human-readable message on any problem.
    """
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")

    if "passengers" not in payload:
        raise ValueError("Missing required field 'passengers'.")

    values = payload["passengers"]

    if not isinstance(values, (list, tuple)):
        raise ValueError("'passengers' must be a list of numbers.")

    if len(values) != WINDOW_SIZE:
        raise ValueError(
            f"'passengers' must contain exactly {WINDOW_SIZE} values "
            f"(got {len(values)}). Provide the last {WINDOW_SIZE} months "
            "ordered oldest -> newest."
        )

    cleaned = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"Value at position {i} ('{v}') is not a number."
            )
        if np.isnan(v) or np.isinf(v):
            raise ValueError(f"Value at position {i} is not finite.")
        if v < 0:
            raise ValueError(
                f"Value at position {i} ('{v}') is negative; passenger "
                "counts must be non-negative."
            )
        cleaned.append(float(v))

    return cleaned


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """Basic service description."""
    return jsonify(
        {
            "service": "Air Passengers Forecasting API",
            "model": "TabPFNRegressor (sliding-window regression)",
            "window_size": WINDOW_SIZE,
            "endpoints": {
                "GET /docs": "Interactive HTML docs + in-browser test form.",
                "GET /health": "Health check.",
                "POST /predict": (
                    "Predict next month's passengers from the last "
                    f"{WINDOW_SIZE} months (oldest -> newest)."
                ),
            },
        }
    )


@app.get("/docs")
def docs():
    """Human-friendly HTML docs with an in-browser form to test /predict."""
    return render_template_string(DOCS_HTML, window_size=WINDOW_SIZE)


@app.get("/health")
def health():
    """Liveness/readiness probe confirming artifacts are loaded."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.post("/predict")
def predict():
    """Predict the next month's passenger count."""
    # Parse JSON safely (silent=True avoids a raw 400 HTML page).
    payload = request.get_json(silent=True)
    if payload is None:
        return (
            jsonify(
                {
                    "error": "Request body must be valid JSON with "
                    "Content-Type: application/json."
                }
            ),
            400,
        )

    # Validate input.
    try:
        passengers = _validate_passengers(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    # Scale features exactly as during training, then predict.
    try:
        features = np.array(passengers, dtype=float).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = float(model.predict(features_scaled)[0])
    except Exception as exc:  # noqa: BLE001 - surface any inference failure
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(
        {
            "input_passengers": passengers,
            "predicted_next_month": round(prediction, 2),
            "window_size": WINDOW_SIZE,
        }
    )


# ---------------------------------------------------------------------------
# Error handlers (return JSON instead of HTML)
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


if __name__ == "__main__":
    # Default to port 5001 to avoid clashing with the macOS AirPlay
    # receiver, which occupies port 5000.
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
