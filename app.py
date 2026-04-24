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
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from model_api import predict_image


# =========================================================
# PATH CONFIG (ROOT SAFE)
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# FLASK CONFIG
# =========================================================
app = Flask(__name__)
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
# HEALTH CHECK
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "AI Image Detection API is running."
    })


# =========================================================
# SIMPLE TEST UI
# =========================================================
@app.route("/upload", methods=["GET"])
def upload_page():
    return """
    <html>
    <head>
        <title>AI Image Detector</title>
    </head>
    <body style="font-family:Arial;padding:40px;">
        <h2>AI Image Detector</h2>
        <p>Select an image and test the model.</p>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br><br>
            <button type="submit">Upload & Predict</button>
        </form>
    </body>
    </html>
    """


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