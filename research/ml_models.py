# research/ml_models.py
# ---------------------------------------------------------
# Extract EfficientNet-B0 features from full dataset
# Then train multiple ML models
#
# Run from project root:
#   python research/ml_models.py
#
# Uses:
#   dataset/
#   models/baseline_ai_detector.pth
#
# Saves:
#   features/*.npy
# ---------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


# ======================================================
# ROOT SAFE PATHS
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

os.makedirs(FEATURES_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "baseline_ai_detector.pth"
)


# ======================================================
# CONFIG
# ======================================================
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0
SAVE_FEATURES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# ======================================================
# TRANSFORMS
# ======================================================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
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
# LOAD MODEL
# ======================================================
model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

print("Model Loaded.")


# ======================================================
# FEATURE EXTRACTION
# ======================================================
def extract_features(loader, split_name):

    features_list = []
    labels_list = []

    with torch.no_grad():

        for batch_idx, (images, labels) in enumerate(loader):

            if batch_idx % 50 == 0:
                print(
                    f"{split_name}: "
                    f"Batch {batch_idx}/{len(loader)}"
                )

            images = images.to(device)

            feats = model.features(images)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)

            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())

    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(labels_list, axis=0)

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

print("\nTrain Shape:", X_train.shape)
print("Val Shape:", X_val.shape)
print("Test Shape:", X_test.shape)


# ======================================================
# SAVE FEATURES
# ======================================================
if SAVE_FEATURES:

    np.save(os.path.join(FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)

    np.save(os.path.join(FEATURES_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(FEATURES_DIR, "y_val.npy"), y_val)

    np.save(os.path.join(FEATURES_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)

    print("\nSaved feature files to /features")


# ======================================================
# MODELS
# ======================================================
models_dict = {

    "Logistic Regression":
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),

    "Linear SVM":
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=5000))
        ]),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            n_jobs=-1,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "XGBoost":
        XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss"
        )
}


# ======================================================
# TRAIN + TEST
# ======================================================
results = []

for name, clf in models_dict.items():

    print("\n" + "=" * 60)
    print("Training:", name)

    clf.fit(X_train, y_train)

    pred_val = clf.predict(X_val)
    pred_test = clf.predict(X_test)

    val_acc = accuracy_score(y_val, pred_val)
    test_acc = accuracy_score(y_test, pred_test)

    print(f"\n{name} VAL Accuracy : {val_acc:.4f}")
    print(f"{name} TEST Accuracy: {test_acc:.4f}")

    print("\nTEST Classification Report:")
    print(classification_report(
        y_test,
        pred_test,
        target_names=class_names,
        digits=4
    ))

    results.append((name, val_acc, test_acc))


# ======================================================
# LEADERBOARD
# ======================================================
print("\n" + "=" * 60)
print("FINAL RESULTS")

results = sorted(
    results,
    key=lambda x: x[2],
    reverse=True
)

for r in results:
    print(
        f"{r[0]:20s} "
        f"VAL={r[1]:.4f} "
        f"TEST={r[2]:.4f}"
    )