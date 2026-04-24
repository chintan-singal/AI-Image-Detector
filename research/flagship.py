# research/flagship.py
# ==========================================================
# FLAGSHIP MODEL PIPELINE
#
# Run from project root:
#   python research/flagship.py
#
# Uses:
#   dataset/
#   models/hard_mined_fusion_xgboost.pkl
#   features/*.npy
#
# Saves:
#   models/flagship_effnet.pth
#   models/flagship_swin.pth
#   models/flagship_xgb.pkl
#   features/*.npy
# ==========================================================

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import joblib

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier


# ==========================================================
# ROOT SAFE PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FEATURES_DIR = os.path.join(ROOT_DIR, "features")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Existing inputs
OLD_XGB_PATH = os.path.join(MODELS_DIR, "hard_mined_fusion_xgboost.pkl")

OLD_EFF_TRAIN = os.path.join(FEATURES_DIR, "X_train.npy")
OLD_SWIN_TRAIN = os.path.join(FEATURES_DIR, "swin_X_train.npy")
OLD_Y_TRAIN = os.path.join(FEATURES_DIR, "y_train.npy")

# Outputs
EFF_SAVE = os.path.join(MODELS_DIR, "flagship_effnet.pth")
SWIN_SAVE = os.path.join(MODELS_DIR, "flagship_swin.pth")
XGB_SAVE = os.path.join(MODELS_DIR, "flagship_xgb.pkl")


# ==========================================================
# CONFIG
# ==========================================================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 5
LR = 1e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# ==========================================================
# TRANSFORMS
# ==========================================================
train_tf = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# ==========================================================
# DATASETS
# ==========================================================
train_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_tf
)

val_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=test_tf
)

test_ds = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_tf
)

class_names = train_ds.classes


# ==========================================================
# LOAD HARD WEIGHTS
# ==========================================================
print("Loading old fused features...")

eff_X_train = np.load(OLD_EFF_TRAIN)
swin_X_train = np.load(OLD_SWIN_TRAIN)
y_train_np = np.load(OLD_Y_TRAIN)

X_old = np.concatenate([eff_X_train, swin_X_train], axis=1)

old_model = joblib.load(OLD_XGB_PATH)

probs = old_model.predict_proba(X_old)
preds = np.argmax(probs, axis=1)
conf = np.max(probs, axis=1)

sample_weights = []

for i in range(len(y_train_np)):

    hard = False

    if preds[i] != y_train_np[i]:
        hard = True
    elif conf[i] < 0.60:
        hard = True

    sample_weights.append(3.0 if hard else 1.0)

sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ==========================================================
# LOADERS
# ==========================================================
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
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


# ==========================================================
# TRAIN FUNCTION
# ==========================================================
def train_model(model, optimizer, criterion, save_name):

    best_acc = 0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        model.train()

        for imgs, labels in train_loader:

            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = model(imgs)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for imgs, labels in val_loader:

                imgs = imgs.to(device)
                labels = labels.to(device)

                out = model(imgs)
                pred = torch.argmax(out, dim=1)

                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print("VAL ACC:", acc)

        if acc > best_acc:
            best_acc = acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)

    torch.save(model.state_dict(), save_name)
    print("Saved:", save_name)

    return model


# ==========================================================
# STAGE 1 - EFFICIENTNET
# ==========================================================
print("\n=== Fine-tuning EfficientNet ===")

eff = models.efficientnet_b0(weights="DEFAULT")

num_f = eff.classifier[1].in_features
eff.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_f, 2)
)

for p in eff.features[:-2].parameters():
    p.requires_grad = False

eff = eff.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, eff.parameters()),
    lr=LR
)

eff = train_model(eff, optimizer, criterion, EFF_SAVE)


# ==========================================================
# STAGE 2 - SWIN
# ==========================================================
print("\n=== Fine-tuning Swin ===")

swin = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=2
)

for name, p in swin.named_parameters():
    if "layers.3" not in name and "head" not in name:
        p.requires_grad = False

swin = swin.to(device)

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, swin.parameters()),
    lr=LR
)

swin = train_model(swin, optimizer, criterion, SWIN_SAVE)


# ==========================================================
# FEATURE EXTRACTION LOADERS
# ==========================================================
plain_train = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=test_tf
)

plain_val = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=test_tf
)

plain_test = datasets.ImageFolder(
    os.path.join(DATA_DIR, "test"),
    transform=test_tf
)

train_loader2 = DataLoader(plain_train, batch_size=64, shuffle=False)
val_loader2 = DataLoader(plain_val, batch_size=64, shuffle=False)
test_loader2 = DataLoader(plain_test, batch_size=64, shuffle=False)


# ==========================================================
# FEATURE HELPERS
# ==========================================================
def extract_eff(model, loader):

    feats_all = []
    labels_all = []

    model.eval()

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            feats = model.features(imgs)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)

            feats_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(feats_all), np.concatenate(labels_all)


def extract_swin(model, loader):

    feats_all = []
    labels_all = []

    model.eval()

    with torch.no_grad():

        for imgs, labels in loader:

            imgs = imgs.to(device)

            feats = model.forward_features(imgs)

            if feats.ndim == 4:
                feats = feats.mean(dim=(1, 2))
            elif feats.ndim == 3:
                feats = feats.mean(dim=1)

            feats_all.append(feats.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(feats_all), np.concatenate(labels_all)


# ==========================================================
# EXTRACT NEW FEATURES
# ==========================================================
print("\nExtracting new EfficientNet features...")

eff_train, y_train = extract_eff(eff, train_loader2)
eff_val, y_val = extract_eff(eff, val_loader2)
eff_test, y_test = extract_eff(eff, test_loader2)

print("Extracting new Swin features...")

swin_train, _ = extract_swin(swin, train_loader2)
swin_val, _ = extract_swin(swin, val_loader2)
swin_test, _ = extract_swin(swin, test_loader2)

# Save features
np.save(os.path.join(FEATURES_DIR, "X_train.npy"), eff_train)
np.save(os.path.join(FEATURES_DIR, "X_val.npy"), eff_val)
np.save(os.path.join(FEATURES_DIR, "X_test.npy"), eff_test)

np.save(os.path.join(FEATURES_DIR, "swin_X_train.npy"), swin_train)
np.save(os.path.join(FEATURES_DIR, "swin_X_val.npy"), swin_val)
np.save(os.path.join(FEATURES_DIR, "swin_X_test.npy"), swin_test)

np.save(os.path.join(FEATURES_DIR, "y_train.npy"), y_train)
np.save(os.path.join(FEATURES_DIR, "y_val.npy"), y_val)
np.save(os.path.join(FEATURES_DIR, "y_test.npy"), y_test)


# ==========================================================
# FUSION
# ==========================================================
X_train = np.concatenate([eff_train, swin_train], axis=1)
X_val   = np.concatenate([eff_val, swin_val], axis=1)
X_test  = np.concatenate([eff_test, swin_test], axis=1)

print("Fusion Shape:", X_train.shape)


# ==========================================================
# XGBOOST
# ==========================================================
print("\nTraining flagship XGBoost...")

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

clf.fit(X_train, y_train)

val_pred = clf.predict(X_val)
test_pred = clf.predict(X_test)

print("\nVAL ACC:", accuracy_score(y_val, val_pred))
print("TEST ACC:", accuracy_score(y_test, test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    test_pred,
    target_names=class_names,
    digits=4
))

joblib.dump(clf, XGB_SAVE)

print("\nSaved:", XGB_SAVE)