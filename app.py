#!/usr/bin/env python3
"""
Flask ML service for spam detection.

Loads a trained sklearn Pipeline (TF-IDF + MultinomialNB) from a pickle file
and exposes classification and health endpoints.

Environment variables:
    MODEL_PATH  - Path to the model pickle file (default: Model_Store/model_v1.pkl)
    HOST        - Host to bind to (default: 0.0.0.0)
    PORT        - Port to listen on (default: 5000)
"""

import json
import logging
import os
import pickle
import sys
import time
import traceback

from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (read from environment, no hard-coded values)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model_Store", "model_v1.pkl")
MODEL_PATH = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))

# ---------------------------------------------------------------------------
# Model loading (Req 3.2, 3.3, 3.5)
# ---------------------------------------------------------------------------
def load_model(path: str):
    """Load the sklearn Pipeline from a pickle file. Exits on failure."""
    if not os.path.exists(path):
        logger.error("Model file not found: %s", path)
        sys.exit(1)

    try:
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
    except Exception:
        logger.error("Failed to load model from '%s':\n%s", path, traceback.format_exc())
        sys.exit(1)

    # Basic sanity check — must have predict and predict_proba
    if not (hasattr(pipeline, "predict") and hasattr(pipeline, "predict_proba")):
        logger.error("Loaded object from '%s' is not a valid classifier pipeline.", path)
        sys.exit(1)

    logger.info("Model loaded successfully from %s", path)
    return pipeline


def load_model_metadata(model_path: str) -> dict:
    """
    Load sidecar JSON metadata file if present.
    The sidecar is expected at <model_path>.json  (e.g. model_v1.pkl.json).
    Returns defaults when the file is absent or unreadable.
    """
    meta_path = model_path + ".json"
    defaults = {
        "name": "spam-classifier",
        "version": "1",
        "training_date": "unknown",
    }
    if not os.path.exists(meta_path):
        return defaults
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "name": data.get("name", defaults["name"]),
            "version": str(data.get("version", defaults["version"])),
            "training_date": data.get("training_date", defaults["training_date"]),
        }
    except Exception:
        logger.warning("Could not read model metadata from '%s'; using defaults.", meta_path)
        return defaults


# Load model at import time so startup failures are caught before the server
# starts accepting requests (Req 3.2, 3.3).
pipeline = load_model(MODEL_PATH)
model_metadata = load_model_metadata(MODEL_PATH)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint (Req 2.2 / 3.2)."""
    return jsonify({"status": "ok"}), 200


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classify a text as spam or ham.

    Request body:  {"text": "..."}
    Response body: {"label": "spam"|"ham", "confidence": float}

    Req 1.2, 6.2
    """
    start = time.perf_counter()

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string."}), 400

    # Predict
    label = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    confidence = float(proba[classes.index(label)])

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Req 6.2 — log input length, label, confidence, processing time (not raw text)
    logger.info(
        "classify | input_length=%d label=%s confidence=%.4f processing_time_ms=%.2f",
        len(text),
        label,
        confidence,
        elapsed_ms,
    )

    return jsonify({"label": label, "confidence": confidence}), 200


@app.route("/model/info", methods=["GET"])
def model_info():
    """Return model name, version, and training date."""
    return jsonify(model_metadata), 200


# ---------------------------------------------------------------------------
# Global error handler (Req 6.4)
# ---------------------------------------------------------------------------

@app.errorhandler(Exception)
def handle_unhandled_exception(exc):
    logger.error("Unhandled exception:\n%s", traceback.format_exc())
    return jsonify({"error": "An internal server error occurred."}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)
