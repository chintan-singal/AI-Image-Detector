# app.py
# ---------------------------------------------------------
# Flask Backend API for AI Image Detector
# Local Hosting Ready
#
# Run from root folder:
#   python app.py
#
# API:
#   GET  /
#   GET  /upload
#   POST /predict
# ---------------------------------------------------------

import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from model_api import predict_image


# =========================================================
# PATH CONFIG (ROOT SAFE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# FLASK CONFIG
# =========================================================
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}


# =========================================================
# HELPERS
# =========================================================
def allowed_file(filename):
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# =========================================================
# SERVE FRONTEND
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")

@app.route("/<path:path>")
def serve_static(path):
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return app.send_static_file(path)
    else:
        return app.send_static_file("index.html")


# =========================================================
# MAIN PREDICTION ROUTE
# =========================================================
@app.route("/predict", methods=["POST"])
def predict():

    temp_path = None

    try:
        # Check file present
        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "Empty filename."
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "success": False,
                "error": "Unsupported file type."
            }), 400

        # Safe name
        filename = secure_filename(file.filename)

        # Unique temp filename
        ext = filename.rsplit(".", 1)[1].lower()
        temp_name = f"{uuid.uuid4().hex}.{ext}"

        temp_path = os.path.join(app.config["UPLOAD_FOLDER"], temp_name)

        # Save upload
        file.save(temp_path)

        # Predict
        result = predict_image(temp_path)

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )