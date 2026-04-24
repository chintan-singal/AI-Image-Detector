# AI Image Detector

A hybrid deep learning + machine learning system for detecting whether an image is **AI-generated** or **real**.

This project combines:

* EfficientNet-B0 (CNN feature extractor)
* Swin Transformer (Vision Transformer feature extractor)
* Weighted XGBoost Classifier
* Hard Example Mining
* SHAP Explainability
* Flask API for local deployment

---

# Project Structure

```text
AI-Image-Detector/
│── app.py
│── model_api.py
│── requirements.txt
│── README.md
│
├── models/
│   ├── flagship_effnet.pth
│   ├── flagship_swin.pth
│   └── final_flagship_xgb.pkl
│
├── features/
│   ├── X_train.npy
│   └── swin_X_train.npy
│
├── dev/
├── research/
└── notes/
```

---

# Features

* Detects AI-generated vs Real images
* Confidence score output
* Human-readable prediction reasoning
* Browser image upload testing
* JSON API response
* Fully local deployment (no cloud required)

---

# Model Architecture

```text
Input Image
↓
EfficientNet-B0 Features
↓
Swin Transformer Features
↓
Feature Fusion
↓
Hard-Mined Weighted XGBoost
↓
Prediction + Confidence + SHAP Explanation
```

---
# Model Evolution & Experimental Results

This project was developed through multiple training stages, progressively improving performance from a baseline CNN model to a final hybrid flagship architecture. Results below are summarized from training logs and experiments. 

---

## Stage 1 — Baseline CNN (EfficientNet-B0 End-to-End)

Initial model trained directly as a binary image classifier.

### Architecture

* EfficientNet-B0
* Final classification head (2 classes)

### Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 71.06% |
| Test Accuracy       | 71.00% |

### Key Observation

Good starting point, but struggled on difficult and photorealistic AI images. 

---

## Stage 2 — CNN Features + Classical ML Models

EfficientNet was converted into a feature extractor (1280-dim embeddings), then multiple ML models were tested.

### Results

| Model               | Validation |   Test |
| ------------------- | ---------: | -----: |
| Logistic Regression |     77.29% | 77.72% |
| Linear SVM          |     77.40% | 77.80% |
| Random Forest       |     79.28% | 79.22% |
| Decision Tree       |     63.58% | 63.69% |
| XGBoost             |     80.77% | 80.87% |

### Key Observation

XGBoost significantly outperformed the baseline neural classifier. 

---

## Stage 3 — Swin Transformer Features + XGBoost

Added a transformer-based vision encoder for stronger global visual understanding.

### Architecture

* Swin Tiny Transformer
* 768-dim feature embeddings
* XGBoost classifier

### Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 84.87% |
| Test Accuracy       | 85.31% |

### Key Observation

Large jump in accuracy versus CNN-only pipeline. 

---

## Stage 4 — Feature Fusion (EfficientNet + Swin)

Combined local texture features (CNN) with global semantic features (Transformer).

### Architecture

* EfficientNet features (1280)
* Swin features (768)
* Concatenated to 2048-dim vector
* XGBoost classifier

### Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 88.41% |
| Test Accuracy       | 88.59% |

### Key Observation

Feature fusion created a more robust representation than either model alone. 

---

## Stage 5 — Hard Example Mining + Weighted XGBoost

Misclassified and low-confidence samples were given higher training importance.

### Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 89.84% |
| Test Accuracy       | 90.29% |

### Gain Over Previous Stage

+1.70% absolute test accuracy improvement. 

---

## Stage 6 — Final Flagship Production Model

Further refinement of the hard-mined hybrid pipeline with optimized weighting and feature extraction.

### Final Architecture

```text id="6y0sm7"
Input Image
↓
EfficientNet-B0 Feature Extractor
+
Swin Transformer Feature Extractor
↓
2048-Dim Feature Fusion
↓
Weighted XGBoost
↓
Confidence + Explanation Output
```

### Final Performance

| Metric              |  Score |
| ------------------- | -----: |
| Validation Accuracy | 96.14% |
| Test Accuracy       | 96.20% |

### Final Classification Quality

* AI Precision: 95.95%
* AI Recall: 96.35%
* Real Precision: 96.43%
* Real Recall: 96.04%



---

## Performance Progression Summary

| Stage                  | Test Accuracy |
| ---------------------- | ------------: |
| Baseline EfficientNet  |        71.00% |
| EfficientNet + XGBoost |        80.87% |
| Swin + XGBoost         |        85.31% |
| Fusion XGBoost         |        88.59% |
| Hard-Mined Fusion      |        90.29% |
| Final Flagship Model   |    **96.20%** |



---

## Engineering Takeaways

* CNNs captured strong texture-level artifacts.
* Transformers improved global consistency detection.
* XGBoost outperformed simple dense neural heads on extracted embeddings.
* Feature fusion substantially improved robustness.
* Hard example mining was critical for edge-case performance.
* Final system achieved production-grade accuracy on the curated test set.

---

## Reproducibility

Training scripts for each stage are included in:

```text id="fmg6gr"
research/
```

Inference API is available in:

```text id="0m3uq8"
app.py
model_api.py
```
---

# Download Required Model Files

Large trained model files and extracted feature matrices are hosted separately due to GitHub file size limits.

## Download Link

https://drive.google.com/file/d/17cYU4K6t6ibIcRTWef9WwSTAERCdorMc/view?usp=sharing

---

# How to Add the Files

## Step 1: Download the ZIP file

Download the archive from the Google Drive link above.

## Step 2: Extract the ZIP

After downloading, right-click the ZIP file and choose:

```text
Extract All...
```

## Step 3: Copy folders into project root

Move the extracted folders:

```text
models/
features/
```

into the main project folder where:

```text
app.py
model_api.py
README.md
```

already exist.

## Final Structure Should Look Like:

```text
AI-Image-Detector/
│── app.py
│── model_api.py
│── requirements.txt
│── README.md
│
├── models/
├── features/
├── dev/
├── research/
└── notes/
```

---

# Dataset Sources

The training dataset was created by combining multiple publicly available datasets from Kaggle containing both **real** and **AI-generated** images.

## Primary Sources

1. CIFAKE – Real and AI Generated Synthetic Images
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

2. Midjourney CIFAKE Inspired
   https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired

3. RVF10K
   https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k

4. HardFake vs Real Faces
   https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces

5. AI Generated Images vs Real Images
   https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images

6. AI Generated Images vs Real Images
   https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images

7. Shoes Dataset – Real and AI Generated Images
   https://www.kaggle.com/datasets/sunnykakar/shoes-dataset-real-and-ai-generated-images

8. AI vs Real Images Dataset
   https://www.kaggle.com/datasets/rhythmghai/ai-vs-real-images-dataset

9. 200K Real vs AI Visuals
   https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal

10. Real vs Fake Faces
    https://www.kaggle.com/datasets/uditsharma72/real-vs-fake-faces

11. MiniImageNet
    https://www.kaggle.com/datasets/deeptrial/miniimagenet

---

# Final Dataset Summary

* Approximate total size: **350,000+ images**
* Classes:

  * Real Images
  * AI-Generated Images

## Included Categories

* Faces
* Human portraits
* Objects
* Shoes
* Landscapes
* Mixed real-world scenes
* Stylized AI content
* Photorealistic synthetic images

## Preprocessing

* RGB conversion
* Resize to **224 x 224**
* Train / Validation / Test split

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repo-url>
cd AI-Image-Detector
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Project

```bash
python app.py
```

Server starts at:

```text
http://127.0.0.1:5000
```

---

# Browser Upload Testing

Open:

```text
http://127.0.0.1:5000/upload
```

Upload an image and receive prediction output.

---

# API Endpoint

## POST `/predict`

Send image as form-data.

Key:

```text
image
```

Returns:

```json
{
  "success": true,
  "prediction": "Real",
  "confidence": 97.14,
  "message": "This image is highly likely authentic.",
  "reasons": [
    "Natural texture variation strongly supported authenticity."
  ]
}
```

---

# Limitations

Model performance may vary on:

* unseen object categories
* highly photorealistic AI generations
* heavily edited images
* future generation models not present in training data

---

# Technologies Used

* Python
* PyTorch
* Torchvision
* TIMM
* XGBoost
* SHAP
* Flask

---

# Author

Chintan Kumar Singal
