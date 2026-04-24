# research/final.py
# ==========================================================
# SAFE FINAL STAGE PIPELINE
#
# Run from project root:
#   python research/final.py
#
# Uses:
#   models/flagship_effnet.pth
#   models/flagship_swin.pth
#   dataset/
#
# Saves:
#   models/final_flagship_xgb.pkl
#   features/*.npy
# ==========================================================

import os
import numpy as np
import torch
import torch.nn as nn
import timm
import joblib

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


# ==========================================================
# ROOT-SAFE PATH CONFIG
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

EFF_MODEL_PATH = os.path.join(MODELS_DIR, "flagship_effnet.pth")
SWIN_MODEL_PATH = os.path.join(MODELS_DIR, "flagship_swin.pth")
FINAL_XGB_PATH = os.path.join(MODELS_DIR, "final_flagship_xgb.pkl")


# ==========================================================
# CONFIG
# ==========================================================
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# ==========================================================
# TRANSFORM
# ==========================================================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ==========================================================
# SAFE DATASET
# ==========================================================
class SafeImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.base = datasets.ImageFolder(root)
        self.samples = self.base.samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            return img, label

        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))


# ==========================================================
# DATASETS
# ==========================================================
train_ds = SafeImageFolder(os.path.join(DATA_DIR, "train"), transform)
val_ds   = SafeImageFolder(os.path.join(DATA_DIR, "val"), transform)
test_ds  = SafeImageFolder(os.path.join(DATA_DIR, "test"), transform)

class_names = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train")
).classes

print("Classes:", class_names)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


# ==========================================================
# LOAD EFFICIENTNET
# ==========================================================
eff = models.efficientnet_b0(weights=None)

num_f = eff.classifier[1].in_features
eff.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_f, 2)
)

eff.load_state_dict(torch.load(EFF_MODEL_PATH, map_location=device))
eff = eff.to(device)
eff.eval()

print("Loaded flagship_effnet.pth")


# ==========================================================
# LOAD SWIN
# ==========================================================
swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=2
)

swin.load_state_dict(torch.load(SWIN_MODEL_PATH, map_location=device))
swin = swin.to(device)
swin.eval()

print("Loaded flagship_swin.pth")


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_eff(loader, split):

    feats_all = []
    labels_all = []

    with torch.no_grad():

        for batch_idx, (imgs, labels) in enumerate(loader):

            if batch_idx % 50 == 0:
                print(f"{split} EfficientNet Batch {batch_idx}/{len(loader)}")

            imgs = imgs.to(device)

            feats = eff.features(imgs)
            feats = eff.avgpool(feats)
            feats = torch.flatten(feats, 1)

            feats_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(feats_all), np.concatenate(labels_all)


def extract_swin(loader, split):

    feats_all = []
    labels_all = []

    with torch.no_grad():

        for batch_idx, (imgs, labels) in enumerate(loader):

            if batch_idx % 50 == 0:
                print(f"{split} Swin Batch {batch_idx}/{len(loader)}")

            imgs = imgs.to(device)

            feats = swin.forward_features(imgs)

            if feats.ndim == 4:
                feats = feats.mean(dim=(1, 2))
            elif feats.ndim == 3:
                feats = feats.mean(dim=1)

            feats_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(feats_all), np.concatenate(labels_all)


# ==========================================================
# EXTRACT FEATURES
# ==========================================================
print("\nExtracting TRAIN features...")
eff_train, y_train = extract_eff(train_loader, "TRAIN")
swin_train, _ = extract_swin(train_loader, "TRAIN")

print("\nExtracting VAL features...")
eff_val, y_val = extract_eff(val_loader, "VAL")
swin_val, _ = extract_swin(val_loader, "VAL")

print("\nExtracting TEST features...")
eff_test, y_test = extract_eff(test_loader, "TEST")
swin_test, _ = extract_swin(test_loader, "TEST")


# ==========================================================
# SAVE BACKGROUND FEATURES
# ==========================================================
np.save(os.path.join(FEATURES_DIR, "X_train.npy"), eff_train)
np.save(os.path.join(FEATURES_DIR, "swin_X_train.npy"), swin_train)


# ==========================================================
# FUSION
# ==========================================================
X_train = np.concatenate([eff_train, swin_train], axis=1)
X_val   = np.concatenate([eff_val, swin_val], axis=1)
X_test  = np.concatenate([eff_test, swin_test], axis=1)

print("\nFusion Shape:", X_train.shape)


# ==========================================================
# HARD EXAMPLE MINING
# ==========================================================
print("\nMining hard examples...")

base = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

base.fit(X_train, y_train)

probs = base.predict_proba(X_train)
preds = np.argmax(probs, axis=1)
conf = np.max(probs, axis=1)

weights = np.ones(len(y_train))
hard = 0

for i in range(len(y_train)):

    if preds[i] != y_train[i]:
        weights[i] = 3.0
        hard += 1

    elif conf[i] < 0.60:
        weights[i] = 3.0
        hard += 1

print("Hard examples:", hard)


# ==========================================================
# FINAL XGBOOST
# ==========================================================
print("\nTraining FINAL weighted XGBoost...")

clf = XGBClassifier(
    n_estimators=900,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train, sample_weight=weights)


# ==========================================================
# EVALUATE
# ==========================================================
val_pred = clf.predict(X_val)
test_pred = clf.predict(X_test)

print("\nVAL Accuracy :", accuracy_score(y_val, val_pred))
print("TEST Accuracy:", accuracy_score(y_test, test_pred))

print("\nTEST Classification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=class_names,
    digits=4
))


# ==========================================================
# SAVE FINAL MODEL
# ==========================================================
joblib.dump(clf, FINAL_XGB_PATH)

print(f"\nSaved: {FINAL_XGB_PATH}")