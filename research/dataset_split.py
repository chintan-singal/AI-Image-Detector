# research/dataset_split.py
# ---------------------------------------------------------
# Dataset Train / Val / Test Splitter
#
# Run from project root:
#   python research/dataset_split.py
#
# Expected source:
#   MASTER_DATASET_FINAL/
#       real/
#       ai/
#
# Creates:
#   dataset/
#       train/
#       val/
#       test/
# ---------------------------------------------------------

import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm


# =========================================================
# ROOT-SAFE PATH CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

SOURCE = ROOT_DIR / "MASTER_DATASET_FINAL"
DEST = ROOT_DIR / "dataset"


# =========================================================
# CONFIG
# =========================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

IMG_EXT = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================
for split in ["train", "val", "test"]:
    for cls in ["real", "ai"]:
        (DEST / split / cls).mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def get_images(folder):
    return [
        x for x in folder.iterdir()
        if x.is_file() and x.suffix.lower() in IMG_EXT
    ]


# =========================================================
# SPLIT + COPY
# =========================================================
def split_and_copy(class_name):

    src = SOURCE / class_name

    if not src.exists():
        print(f"Missing folder: {src}")
        return

    files = get_images(src)

    random.shuffle(files)

    total = len(files)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, split_files in splits.items():

        print(
            f"\nCopying {class_name} -> {split_name} "
            f"({len(split_files)} images)"
        )

        for file in tqdm(split_files):

            target = DEST / split_name / class_name / file.name
            shutil.copy2(file, target)


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Source:", SOURCE)
    print("Destination:", DEST)

    split_and_copy("real")
    split_and_copy("ai")

    print("\nDONE!")