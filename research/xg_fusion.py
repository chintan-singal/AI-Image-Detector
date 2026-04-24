# research/xg_fusion.py
# ---------------------------------------------------------
# EfficientNet Features + Swin Features -> XGBoost
#
# Run from project root:
#   python research/xg_fusion.py
#
# Uses:
#   features/*.npy
#
# Saves:
#   models/fusion_xgboost_ai_detector.pkl
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

SAVE_MODEL_PATH = os.path.join(
    MODELS_DIR,
    "fusion_xgboost_ai_detector.pkl"
)


# ======================================================
# LOAD SAVED FEATURES
# ======================================================
print("Loading feature files...")

# EfficientNet features
eff_X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
eff_y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

eff_X_val = np.load(os.path.join(FEATURES_DIR, "X_val.npy"))
eff_y_val = np.load(os.path.join(FEATURES_DIR, "y_val.npy"))

eff_X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
eff_y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

# Swin features
swin_X_train = np.load(os.path.join(FEATURES_DIR, "swin_X_train.npy"))
swin_y_train = np.load(os.path.join(FEATURES_DIR, "swin_y_train.npy"))

swin_X_val = np.load(os.path.join(FEATURES_DIR, "swin_X_val.npy"))
swin_y_val = np.load(os.path.join(FEATURES_DIR, "swin_y_val.npy"))

swin_X_test = np.load(os.path.join(FEATURES_DIR, "swin_X_test.npy"))
swin_y_test = np.load(os.path.join(FEATURES_DIR, "swin_y_test.npy"))


# ======================================================
# SAFETY CHECKS
# ======================================================
assert np.array_equal(
    eff_y_train, swin_y_train
), "Train labels mismatch!"

assert np.array_equal(
    eff_y_val, swin_y_val
), "Validation labels mismatch!"

assert np.array_equal(
    eff_y_test, swin_y_test
), "Test labels mismatch!"

print("Labels aligned.")


# ======================================================
# FEATURE FUSION
# EfficientNet (1280) + Swin (768)
# Total = 2048 dims
# ======================================================
print("Concatenating features...")

X_train = np.concatenate(
    [eff_X_train, swin_X_train],
    axis=1
)

X_val = np.concatenate(
    [eff_X_val, swin_X_val],
    axis=1
)

X_test = np.concatenate(
    [eff_X_test, swin_X_test],
    axis=1
)

y_train = eff_y_train
y_val = eff_y_val
y_test = eff_y_test

print("Train Shape:", X_train.shape)
print("Val Shape  :", X_val.shape)
print("Test Shape :", X_test.shape)


# ======================================================
# XGBOOST MODEL
# ======================================================
clf = XGBClassifier(
    n_estimators=700,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ======================================================
# TRAIN
# ======================================================
print("\nTraining XGBoost on fused features...")

clf.fit(X_train, y_train)


# ======================================================
# VALIDATION
# ======================================================
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)

print(f"\nVAL Accuracy : {val_acc:.4f}")


# ======================================================
# TEST
# ======================================================
test_pred = clf.predict(X_test)
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
joblib.dump(clf, SAVE_MODEL_PATH)

print("\nSaved:", SAVE_MODEL_PATH)