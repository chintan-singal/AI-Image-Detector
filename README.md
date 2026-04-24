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
AI Image Detector/
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
│── research/
│── notes/
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
