# research/swin.py
# ---------------------------------------------------------
# Full Dataset:
# Swin Transformer Tiny Feature Extractor + XGBoost
#
# Run from project root:
#   python research/swin.py
#
# Uses:
#   dataset/
#
# Saves:
#   models/swin_tiny_feature_extractor.pth
#   models/swin_xgboost_ai_detector.pkl
#   features/swin_*.npy
# ---------------------------------------------------------

import os
import numpy as np
import torch
import timm
import joblib

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ======================================================
# ROOT SAFE PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)


# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0

SAVE_FEATURES = True
TRAIN_XGB = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ======================================================
# DATASETS
# ======================================================
train_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=transform
)

val_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=transform
)

test_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=transform
)

class_names = train_ds.classes
print("Classes:", class_names)


# ======================================================
# LOADERS
# ======================================================
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ======================================================
# LOAD SWIN
# num_classes=0 removes classifier head
# embedding output = 768
# ======================================================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=0
)

model = model.to(device)
model.eval()

print("Swin Tiny loaded.")


# ======================================================
# SAVE WEIGHTS
# ======================================================
SWIN_SAVE_PATH = os.path.join(
    MODELS_DIR,
    "swin_tiny_feature_extractor.pth"
)

torch.save(model.state_dict(), SWIN_SAVE_PATH)

print("Saved:", SWIN_SAVE_PATH)


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(loader, split_name):

    features_all = []
    labels_all = []

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            if batch_idx % 50 == 0:
                print(
                    f"{split_name}: "
                    f"Batch {batch_idx}/{len(loader)}"
                )

            images = images.to(device)

            feats = model(images)

            features_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())

    X = np.concatenate(features_all, axis=0)
    y = np.concatenate(labels_all, axis=0)

    return X, y


# ======================================================
# EXTRACT FEATURES
# ======================================================
print("\nExtracting TRAIN features...")
X_train, y_train = extract_features(train_loader, "TRAIN")

print("\nExtracting VAL features...")
X_val, y_val = extract_features(val_loader, "VAL")

print("\nExtracting TEST features...")
X_test, y_test = extract_features(test_loader, "TEST")

print("\nFeature Shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ======================================================
# SAVE FEATURES
# ======================================================
if SAVE_FEATURES:

    np.save(os.path.join(FEATURES_DIR, "swin_X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "swin_y_train.npy"), y_train)

    np.save(os.path.join(FEATURES_DIR, "swin_X_val.npy"), X_val)
    np.save(os.path.join(FEATURES_DIR, "swin_y_val.npy"), y_val)

    np.save(os.path.join(FEATURES_DIR, "swin_X_test.npy"), X_test)
    np.save(os.path.join(FEATURES_DIR, "swin_y_test.npy"), y_test)

    print("\nSaved feature files to /features")


# ======================================================
# TRAIN XGBOOST
# ======================================================
if TRAIN_XGB:

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("\nTraining XGBoost...")
    clf.fit(X_train, y_train)

    # Validation
    val_pred = clf.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nVAL Accuracy : {val_acc:.4f}")

    # Test
    test_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"TEST Accuracy: {test_acc:.4f}")

    print("\nTEST Classification Report:")
    print(classification_report(
        y_test,
        test_pred,
        target_names=class_names,
        digits=4
    ))

    # Save XGB
    XGB_SAVE_PATH = os.path.join(
        MODELS_DIR,
        "swin_xgboost_ai_detector.pkl"
    )

    joblib.dump(clf, XGB_SAVE_PATH)

    print("\nSaved:", XGB_SAVE_PATH)