# 🫁 Pneumonia Detection from Chest X-Ray Images using CNN and VGG16

<p align="center">
  <b>Deep Learning • Computer Vision • TensorFlow • Streamlit</b>
</p>

------------------------------------------------------------------------

## 📖 Overview

This project classifies chest X-ray images into **NORMAL** and
**PNEUMONIA** classes using two deep learning approaches:

-   **Custom Convolutional Neural Network (CNN)**
-   **Transfer Learning with VGG16**

A **Streamlit** web application allows users to upload chest X-ray
images and obtain predictions with a confidence score.

> **Repository:**
> https://github.com/funistic25/Pneumonia-Detection-from-X-Ray-Images

------------------------------------------------------------------------

## ✨ Features

-   Custom CNN architecture
-   VGG16 transfer learning
-   Image preprocessing and normalization
-   Binary image classification
-   Streamlit web application
-   Confidence score visualization
-   Clean and modular code structure

------------------------------------------------------------------------

## 🧠 Workflow

``` text
Chest X-Ray Image
        │
        ▼
 Image Preprocessing
 (Resize + Normalize)
        │
        ▼
 CNN / VGG16 Model
        │
        ▼
 Binary Classification
        │
        ▼
 NORMAL / PNEUMONIA
```

------------------------------------------------------------------------

## 🏗️ Project Structure

``` text
Pneumonia-Detection-from-X-Ray-Images/
│
├── Pneumonia_Detection.ipynb
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── Dataset/
│   └── Pneumonia_Detection.ipynb
│
├── Models/
│   ├── custom_pre_trained_model_10.keras
│   └── model_10.keras
│
└── Pneumonia-Detection-Streamlit/
    ├── app.py
    └── custom_pre_trained_model_10.keras
```

------------------------------------------------------------------------

## ⚙️ Technologies

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   OpenCV
-   Pillow
-   Matplotlib
-   Streamlit

------------------------------------------------------------------------

## 📊 Model Comparison

  ------------------------------------------------------------------------
  Model                                       Accuracy Notes
  ------------------- -------------------------------- -------------------
  Custom CNN                                 \~70--78% Trained from
                                                       scratch; varies
                                                       slightly between
                                                       runs

  VGG16                                          \~92% Stable transfer
                                                       learning model
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/funistic25/Pneumonia-Detection-from-X-Ray-Images.git
cd Pneumonia-Detection-from-X-Ray-Images
pip install -r requirements.txt
```

Run the application:

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📷 Screenshots

Add these images inside the **screenshots/** folder:

-   homepage.png
-   upload.png
-   prediction_normal.png
-   prediction_pneumonia.png

They will greatly improve the presentation of the repository.

------------------------------------------------------------------------

## 📁 Dataset

The dataset is intentionally excluded because of its size.

Expected structure:

``` text
dataset/
└── chest_xray/
    ├── train/
    ├── test/
    └── val/
```

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Grad-CAM Explainability
-   DenseNet121
-   ResNet50
-   MobileNetV3
-   Docker Support
-   Cloud Deployment
-   REST API

------------------------------------------------------------------------

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**
and must not be used as a medical diagnostic system.
