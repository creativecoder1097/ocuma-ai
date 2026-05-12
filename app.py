from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from PIL import Image

# =========================================================
# Logging
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("retinalscreen")

# =========================================================
# Flask
# =========================================================
app = Flask(__name__)

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model"

# CHANGE THIS IF YOUR FILE NAME IS DIFFERENT
MODEL_PATH = MODEL_DIR / "retinal_model_v2.h5"
META_PATH = MODEL_DIR / "model_meta.json"
DISEASE_MAP_PATH = MODEL_DIR / "disease_map.json"

# =========================================================
# Globals
# =========================================================
model = None

meta = {}
disease_map = {}

IMG_SIZE = 224

# =========================================================
# Load Model + Metadata
# =========================================================
def load_everything():

    global model
    global meta
    global disease_map
    global IMG_SIZE

    try:

        logger.info("Loading retinal model...")

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False,
            safe_mode=False
        )

        logger.info("Loading metadata...")

        if META_PATH.exists():

            with open(META_PATH, "r") as f:
                meta = json.load(f)

        else:

            meta = {}

        if DISEASE_MAP_PATH.exists():

            with open(DISEASE_MAP_PATH, "r") as f:
                disease_map = json.load(f)

        else:

            disease_map = {}

        IMG_SIZE = meta.get("img_size", 224)

        logger.info("Model loaded successfully")

        return True

    except Exception:

        logger.exception("MODEL LOAD FAILED")

        return False


MODEL_READY = load_everything()

# =========================================================
# CLAHE
# =========================================================
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# =========================================================
# Image Preprocessing
# =========================================================
def preprocess_image(img_rgb):

    img = cv2.resize(
        img_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # CLAHE enhancement
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    img = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    # Normalize
    img = img.astype(np.float32) / 255.0

    return img

# =========================================================
# GradCAM
# =========================================================
def generate_gradcam(img_array):

    try:
        # Find last convolution layer
        last_conv_layer = None

        for layer in reversed(model.layers):

            try:
                output_shape = layer.output.shape

                if len(output_shape) == 4:
                    last_conv_layer = layer.name
                    break

            except Exception:
                continue

        if last_conv_layer is None:
            logger.warning("No conv layer found")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        logger.info(f"Using GradCAM layer: {last_conv_layer}")

        # Use ONLY risk output
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer).output,
                model.output[0]
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(img_array)

            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            logger.warning("Gradients are None")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(heatmap, 0)

        max_val = tf.reduce_max(heatmap)

        if max_val == 0:
            logger.warning("Heatmap max is zero")
            return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        heatmap /= max_val

        heatmap = heatmap.numpy()

        heatmap = cv2.resize(
            heatmap,
            (IMG_SIZE, IMG_SIZE)
        )

        return heatmap

    except Exception:
        logger.exception("GradCAM failed")
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

# =========================================================
# Overlay Heatmap
# =========================================================
def overlay_heatmap(img, heatmap):

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        img,
        0.7,
        heatmap,
        0.5,
        0
    )

    return overlay
# =========================================================
# Convert Image → Base64
# =========================================================
def to_base64(img):

    pil_img = Image.fromarray(
        img.astype(np.uint8)
    )

    buf = io.BytesIO()

    pil_img.save(
        buf,
        format="PNG"
    )

    return base64.b64encode(
        buf.getvalue()
    ).decode()

# =========================================================
# Home
# =========================================================
@app.route("/")
def home():

    return render_template(
        "index.html",
        model_loaded=MODEL_READY,
        meta=meta
    )

# =========================================================
# Status API
# =========================================================
@app.route("/api/status")
def status():

    return jsonify({

        "model_loaded": MODEL_READY,

        "model_info": meta
    })

# =========================================================
# Predict API
# =========================================================
@app.route("/api/predict", methods=["POST"])
def predict():

    if not MODEL_READY:

        return jsonify({
            "error": "Model failed to load"
        }), 500

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400

    try:

        start = time.time()

        # =====================================================
        # Read uploaded image
        # =====================================================

        file = request.files["image"]

        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        img_bgr = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if img_bgr is None:

            return jsonify({
                "error": "Invalid image"
            }), 400

        img_rgb = cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        )

        # =====================================================
        # Preprocess
        # =====================================================

        processed = preprocess_image(img_rgb)

        input_tensor = np.expand_dims(
            processed,
            axis=0
        )

        # =====================================================
        # Predict
        # =====================================================

        outputs = model.predict(
            input_tensor,
            verbose=0
        )

        # Multi-output model
        if isinstance(outputs, list):

            risk_output = outputs[0]
            disease_output = outputs[1]

            risk_score = float(risk_output[0][0])

            disease_probs = disease_output[0].tolist()

        # Single-output model fallback
        else:

            risk_score = float(outputs[0][0])

            disease_probs = []

        # =====================================================
        # Risk Label
        # =====================================================

        label = (
            "At Risk"
            if risk_score > 0.5
            else "No Risk"
        )

        confidence = (
            risk_score
            if risk_score > 0.5
            else (1 - risk_score)
        )

        # =====================================================
        # Top Diseases
        # =====================================================

        top_diseases = []

        for idx, prob in enumerate(disease_probs):

            top_diseases.append({

                "name": disease_map.get(
                    str(idx),
                    f"Disease {idx}"
                ),

                "probability": float(prob)
            })

        top_diseases = sorted(
            top_diseases,
            key=lambda x: x["probability"],
            reverse=True
        )[:5]

        # =====================================================
        # GradCAM
        # =====================================================

        heatmap = generate_gradcam(input_tensor)

        display_img = cv2.resize(
            img_rgb,
            (IMG_SIZE, IMG_SIZE)
        )

        overlay = overlay_heatmap(
            display_img,
            heatmap
        )

        # =====================================================
        # Timing
        # =====================================================

        elapsed = int(
            (time.time() - start) * 1000
        )

        # =====================================================
        # Response
        # =====================================================

        return jsonify({

            "prediction": {

                "label": label,

                "probability": risk_score,

                "confidence": confidence
            },

            "top_diseases": top_diseases,

            "images": {

                "original": to_base64(display_img),

                "heatmap": to_base64(
                    np.uint8(heatmap * 255)
                ),

                "overlay": to_base64(overlay)
            },

            "meta": {

                "model": meta.get(
                    "model_name",
                    "RetinalScreenV2"
                ),

                "base_model": meta.get(
                    "base_model",
                    "EfficientNetB3"
                ),

                "test_auc": meta.get("test_auc"),

                "test_f1": meta.get("test_f1"),

                "inference_ms": elapsed
            }
        })

    except Exception:

        logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed"
        }), 500

# =========================================================
# Main
# =========================================================
import os

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )