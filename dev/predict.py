# dev/predict.py
# ---------------------------------------------------------
# CLI Prediction Tester
#
# Run from root folder:
#   python dev/predict.py
#
# or from dev folder:
#   python predict.py
# ---------------------------------------------------------

import os
import json
import numpy as np
import torch
import torch.nn as nn
import timm
import joblib

from PIL import Image
from torchvision import transforms, models


# =========================================================
# ROOT-SAFE PATH CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(ROOT_DIR, "models")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFF_PATH = os.path.join(MODELS_DIR, "flagship_effnet.pth")
SWIN_PATH = os.path.join(MODELS_DIR, "flagship_swin.pth")
XGB_PATH = os.path.join(MODELS_DIR, "final_flagship_xgb.pkl")


# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# LOAD MODELS
# =========================================================
def load_effnet():
    model = models.efficientnet_b0(weights=None)

    num_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_f, 2)
    )

    model.load_state_dict(torch.load(EFF_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def load_swin():
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=2
    )

    model.load_state_dict(torch.load(SWIN_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model


def load_xgb():
    return joblib.load(XGB_PATH)


# =========================================================
# STARTUP LOAD
# =========================================================
print("Loading models...")

eff_model = load_effnet()
swin_model = load_swin()
xgb_model = load_xgb()

print("Models loaded successfully.")


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def extract_eff_features(img_tensor):
    with torch.no_grad():
        feats = eff_model.features(img_tensor)
        feats = eff_model.avgpool(feats)
        feats = torch.flatten(feats, 1)
        return feats.cpu().numpy()


def extract_swin_features(img_tensor):
    with torch.no_grad():
        feats = swin_model.forward_features(img_tensor)

        if feats.ndim == 4:
            feats = feats.mean(dim=(1, 2))
        elif feats.ndim == 3:
            feats = feats.mean(dim=1)

        return feats.cpu().numpy()


# =========================================================
# MESSAGE
# =========================================================
def generate_message(label, confidence):

    if label == "AI Generated":
        if confidence >= 90:
            return "This image is highly likely AI-generated."
        elif confidence >= 75:
            return "This image is likely AI-generated."
        elif confidence >= 55:
            return "This image may be AI-generated."
        else:
            return "Prediction uncertain."

    else:
        if confidence >= 90:
            return "This image is highly likely authentic."
        elif confidence >= 75:
            return "This image appears authentic."
        elif confidence >= 55:
            return "This image may be authentic."
        else:
            return "Prediction uncertain."


# =========================================================
# MAIN PREDICTION
# =========================================================
def predict_image(image_path):

    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(DEVICE)

    eff_feat = extract_eff_features(img)
    swin_feat = extract_swin_features(img)

    X = np.concatenate([eff_feat, swin_feat], axis=1)

    probs = xgb_model.predict_proba(X)[0]

    ai_prob = float(probs[0])
    real_prob = float(probs[1])

    if ai_prob >= real_prob:
        label = "AI Generated"
        confidence = round(ai_prob * 100, 2)
    else:
        label = "Real"
        confidence = round(real_prob * 100, 2)

    return {
        "prediction": label,
        "confidence": confidence,
        "message": generate_message(label, confidence)
    }


# =========================================================
# CLI TEST
# =========================================================
if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    result = predict_image(image_path)

    print(json.dumps(result, indent=4))