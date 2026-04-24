# research/metrics.py
# ---------------------------------------------------------
# Full diagnostics for baseline_ai_detector.pth
#
# Run from project root:
#   python research/metrics.py
#
# Uses:
#   dataset/
#   models/baseline_ai_detector.pth
# ---------------------------------------------------------

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)


# =========================================================
# ROOT SAFE PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "baseline_ai_detector.pth"
)


# =========================================================
# CONFIG
# =========================================================
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)


# =========================================================
# TRANSFORMS
# =========================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# DATASETS
# =========================================================
datasets_dict = {
    split: datasets.ImageFolder(
        os.path.join(DATA_DIR, split),
        transform=transform
    )
    for split in ["train", "val", "test"]
}

loaders = {
    split: DataLoader(
        datasets_dict[split],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )
    for split in ["train", "val", "test"]
}

class_names = datasets_dict["train"].classes
print("Classes:", class_names)


# =========================================================
# LOAD MODEL
# =========================================================
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

print("Model Loaded Successfully")


# =========================================================
# EVALUATION
# =========================================================
def evaluate_split(split="test", save_wrong=True):

    y_true = []
    y_pred = []
    y_prob = []
    wrong_images = []

    loader = loaders[split]
    dataset = datasets_dict[split]

    with torch.no_grad():

        for batch_idx, (inputs, labels) in enumerate(loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            confs, preds = torch.max(probs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(confs.cpu().numpy())

            # Save wrong predictions
            if save_wrong:

                for i in range(len(labels)):

                    if preds[i] != labels[i]:

                        global_idx = batch_idx * BATCH_SIZE + i

                        if global_idx < len(dataset.samples):

                            path, _ = dataset.samples[global_idx]

                            wrong_images.append(
                                (
                                    path,
                                    class_names[labels[i]],
                                    class_names[preds[i]],
                                    float(confs[i])
                                )
                            )

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{split.upper()} Accuracy: {acc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Per class accuracy
    print("\nPer-Class Accuracy:")

    for i in range(len(class_names)):
        cls_acc = cm[i][i] / cm[i].sum()
        print(f"{class_names[i]}: {cls_acc:.4f}")

    # Confidence Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(y_prob, bins=20)
    plt.title(f"{split.upper()} Prediction Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.show()

    # Wrong predictions
    if save_wrong:

        print(f"\nTotal Wrong Predictions: {len(wrong_images)}")
        print("\nTop 20 Wrong Samples:")

        for item in wrong_images[:20]:
            print(
                f"File: {item[0]} | "
                f"True: {item[1]} | "
                f"Pred: {item[2]} | "
                f"Conf: {item[3]:.4f}"
            )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    evaluate_split("train", save_wrong=False)
    evaluate_split("val", save_wrong=True)
    evaluate_split("test", save_wrong=True)