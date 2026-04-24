# research/xg_boost.py
# ---------------------------------------------------------
# Train XGBoost on EfficientNet extracted features
#
# Run from project root:
#   python research/xg_boost.py
#
# Uses:
#   features/X_train.npy
#   features/y_train.npy
#   features/X_val.npy
#   features/y_val.npy
#   features/X_test.npy
#   features/y_test.npy
#
# Saves:
#   models/xgboost_ai_detector.pkl
# ---------------------------------------------------------

import os
import numpy as np
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# =====================================================
# ROOT SAFE PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

FEATURES_DIR = os.path.join(ROOT_DIR, "features")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

SAVE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "xgboost_ai_detector.pkl"
)


# =====================================================
# LOAD FEATURES
# =====================================================
X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

X_val = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
y_val = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))

X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

print("Loaded Features:")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)


# =====================================================
# MODEL
# =====================================================
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# =====================================================
# TRAIN
# =====================================================
print("\nTraining XGBoost...")

model.fit(X_train, y_train)


# =====================================================
# EVALUATE
# =====================================================
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

print("\nVAL Accuracy:", accuracy_score(y_val, val_pred))
print("TEST Accuracy:", accuracy_score(y_test, test_pred))

print("\nTEST Classification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=["ai", "real"],
    digits=4
))


# =====================================================
# SAVE MODEL
# =====================================================
joblib.dump(model, SAVE_MODEL_PATH)

print("\nSaved:", SAVE_MODEL_PATH)