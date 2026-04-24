# research/hard_example_xg.py
# ---------------------------------------------------------
# Hard Example Mining XGBoost Retraining
#
# Run from project root:
#   python research/hard_example_xg.py
#
# Uses:
#   features/*.npy
#   models/fusion_xgboost_ai_detector.pkl
#
# Saves:
#   models/hard_mined_fusion_xgboost.pkl
# ---------------------------------------------------------

import os
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ======================================================
# ROOT SAFE PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

FEATURES_DIR = os.path.join(ROOT_DIR, "features")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Input model
BASE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fusion_xgboost_ai_detector.pkl"
)

# Output model
SAVE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "hard_mined_fusion_xgboost.pkl"
)


# ======================================================
# CONFIG
# ======================================================
LOW_CONFIDENCE = 0.60
HARD_WEIGHT = 3.0
NORMAL_WEIGHT = 1.0


# ======================================================
# LOAD FEATURES
# ======================================================
print("Loading features...")

# EfficientNet
eff_X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

eff_X_val = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
y_val = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))

eff_X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

# Swin
swin_X_train = np.load(os.path.join(FEATURES_DIR, "swin_X_train.npy"))
swin_X_val = np.load(os.path.join(FEATURES_DIR, "swin_X_val.npy"))
swin_X_test = np.load(os.path.join(FEATURES_DIR, "swin_X_test.npy"))


# ======================================================
# FUSION
# ======================================================
X_train = np.concatenate([eff_X_train, swin_X_train], axis=1)
X_val = np.concatenate([eff_X_val, swin_X_val], axis=1)
X_test = np.concatenate([eff_X_test, swin_X_test], axis=1)

print("Train Shape:", X_train.shape)


# ======================================================
# LOAD CURRENT MODEL
# ======================================================
print("Loading current fusion model...")

base_model = joblib.load(BASE_MODEL_PATH)


# ======================================================
# FIND HARD EXAMPLES
# ======================================================
print("Mining hard examples...")

train_probs = base_model.predict_proba(X_train)
train_pred = np.argmax(train_probs, axis=1)
train_conf = np.max(train_probs, axis=1)

weights = np.ones(len(y_train)) * NORMAL_WEIGHT

hard_count = 0

for i in range(len(y_train)):

    correct = (train_pred[i] == y_train[i])

    # Wrong prediction
    if not correct:
        weights[i] = HARD_WEIGHT
        hard_count += 1

    # Correct but low confidence
    elif train_conf[i] < LOW_CONFIDENCE:
        weights[i] = HARD_WEIGHT
        hard_count += 1

print("Hard examples found:", hard_count)
print(
    "Percent hard:",
    round(100 * hard_count / len(y_train), 2),
    "%"
)


# ======================================================
# RETRAIN XGBOOST
# ======================================================
print("\nTraining weighted XGBoost...")

model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.035,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train,
    sample_weight=weights
)


# ======================================================
# VALIDATION
# ======================================================
val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print(f"\nVAL Accuracy : {val_acc:.4f}")


# ======================================================
# TEST
# ======================================================
test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)

print(f"TEST Accuracy: {test_acc:.4f}")

print("\nTEST Classification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["ai", "real"],
    digits=4
))


# ======================================================
# SAVE MODEL
# ======================================================
joblib.dump(model, SAVE_MODEL_PATH)

print("\nSaved:", SAVE_MODEL_PATH)