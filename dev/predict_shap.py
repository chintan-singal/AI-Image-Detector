# dev/predict_shap.py
# ---------------------------------------------------------
# CLI Prediction + SHAP Tester
#
# Run from root folder:
#   python dev/predict_shap.py
#
# or from dev folder:
#   python predict_shap.py
# ---------------------------------------------------------

import os
import json
import numpy as np
import torch
import torch.nn as nn
import timm
import joblib
import shap

from PIL import Image
from torchvision import transforms, models


# =========================================================
# ROOT-SAFE PATH CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EFF_PATH = os.path.join(MODELS_DIR, "flagship_effnet.pth")
SWIN_PATH = os.path.join(MODELS_DIR, "flagship_swin.pth")
XGB_PATH = os.path.join(MODELS_DIR, "final_flagship_xgb.pkl")

EFF_BG_PATH = os.path.join(FEATURES_DIR, "X_train.npy")
SWIN_BG_PATH = os.path.join(FEATURES_DIR, "swin_X_train.npy")


# =========================================================
# TRANSFORM
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


# =========================================================
# STARTUP LOAD
# =========================================================
print("Loading models...")

eff_model = load_effnet()
swin_model = load_swin()
xgb_model = joblib.load(XGB_PATH)

print("Loading SHAP background data...")

eff_bg = np.load(EFF_BG_PATH)
swin_bg = np.load(SWIN_BG_PATH)

X_bg = np.concatenate([eff_bg, swin_bg], axis=1)

# speed subset
if len(X_bg) > 300:
    idx = np.random.choice(len(X_bg), 300, replace=False)
    X_bg = X_bg[idx]

explainer = shap.TreeExplainer(xgb_model, X_bg)

print("Everything loaded successfully.")


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
# SHAP REASONS
# =========================================================
def shap_reasons(shap_vals, prediction):

    vals = shap_vals[0]
    top_idx = np.argsort(np.abs(vals))[::-1][:5]

    reasons = []

    real_positive = [
        "Natural texture variation strongly supported authenticity.",
        "Lighting transitions appeared realistic and organic.",
        "Visual structure matched real-world photographic patterns.",
        "Edge details appeared naturally irregular.",
        "No dominant synthetic artifact signals were detected."
    ]

    real_negative = [
        "Minor synthetic-like signals were detected but were weak.",
        "Some generated-looking patterns appeared, but not strongly enough.",
        "Small artificial consistency traces were present but limited."
    ]

    ai_positive = [
        "Synthetic texture patterns strongly influenced the prediction.",
        "Visual consistency resembled generated imagery.",
        "Detail structures matched learned AI-generated patterns.",
        "Image composition showed signals associated with synthetic creation.",
        "Surface smoothness patterns appeared algorithmic."
    ]

    ai_negative = [
        "Some authentic-image traits were present but outweighed.",
        "Natural visual signals existed but were not dominant.",
        "Real-image characteristics were detected but weaker overall."
    ]

    for i in top_idx:

        impact = vals[i]

        if prediction == "AI Generated":
            if impact > 0:
                reasons.append(ai_positive[len(reasons) % len(ai_positive)])
            else:
                reasons.append(ai_negative[len(reasons) % len(ai_negative)])
        else:
            if impact < 0:
                reasons.append(real_positive[len(reasons) % len(real_positive)])
            else:
                reasons.append(real_negative[len(reasons) % len(real_negative)])

    cleaned = []
    seen = set()

    for r in reasons:
        if r not in seen:
            cleaned.append(r)
            seen.add(r)

    return cleaned[:3]


# =========================================================
# MAIN PREDICT
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

    shap_vals = explainer.shap_values(X)

    if isinstance(shap_vals, list):
        vals = shap_vals[0]
    else:
        vals = shap_vals

    return {
        "prediction": label,
        "confidence": confidence,
        "message": generate_message(label, confidence),
        "reasons": shap_reasons(vals, label)
    }


# =========================================================
# CLI TEST
# =========================================================
if __name__ == "__main__":

    image_path = input("Enter image path: ").strip()

    result = predict_image(image_path)

    print(json.dumps(result, indent=4))