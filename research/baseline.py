# research/baseline_model.py
# ---------------------------------------------------------
# AI Image Detector Baseline Model
# EfficientNet-B0 + GPU Support + Root Folder Safe
#
# Run from project root:
#   python research/baseline_model.py
# ---------------------------------------------------------

import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# =========================================================
# ROOT-SAFE PATH CONFIG
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
SAVE_PATH = os.path.join(ROOT_DIR, "models", "baseline_ai_detector.pth")

# Ensure models folder exists
os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)


# =========================================================
# CONFIG
# =========================================================
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4

# Windows safe multiprocessing
NUM_WORKERS = 0


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# =========================================================
# TRANSFORMS
# =========================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================================================
# DATASETS
# Expected:
# dataset/
#   train/
#   val/
#   test/
# =========================================================
image_datasets = {
    "train": datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"),
        transform=train_transform
    ),
    "val": datasets.ImageFolder(
        os.path.join(DATA_DIR, "val"),
        transform=val_test_transform
    ),
    "test": datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform=val_test_transform
    )
}


# =========================================================
# DATALOADERS
# =========================================================
dataloaders = {
    x: DataLoader(
        image_datasets[x],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    for x in ["train", "val", "test"]
}

dataset_sizes = {
    x: len(image_datasets[x])
    for x in ["train", "val", "test"]
}

class_names = image_datasets["train"].classes

print("Classes:", class_names)
print("Train:", dataset_sizes["train"])
print("Val:", dataset_sizes["val"])
print("Test:", dataset_sizes["test"])


# =========================================================
# MODEL
# =========================================================
model = models.efficientnet_b0(weights="DEFAULT")

# Freeze backbone
for param in model.features.parameters():
    param.requires_grad = False

num_features = model.classifier[1].in_features

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 2)
)

model = model.to(device)


# =========================================================
# LOSS / OPTIMIZER
# =========================================================
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# =========================================================
# TRAIN FUNCTION
# =========================================================
def train_model(model, epochs):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:

            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(
                f"{phase.upper()} Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f}"
            )

            if phase == "val":

                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    print("\nBest Validation Accuracy:", best_acc.item())

    model.load_state_dict(best_model_wts)
    return model


# =========================================================
# TEST FUNCTION
# =========================================================
def evaluate_test(model):

    model.eval()
    running_corrects = 0

    with torch.no_grad():

        for inputs, labels in dataloaders["test"]:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes["test"]

    print(f"\nTEST ACCURACY: {acc:.4f}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    start = time.time()

    model = train_model(model, EPOCHS)

    torch.save(model.state_dict(), SAVE_PATH)

    print(f"\nModel Saved: {SAVE_PATH}")

    evaluate_test(model)

    end = time.time()

    print(f"\nTotal Time: {(end - start)/60:.2f} minutes")