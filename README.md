# 👁️ Retinal Image Analysis System

A Flask-based web application for retinal image preprocessing, blood vessel segmentation, and feature extraction for Diabetic Retinopathy analysis.

---

## 📌 Project Overview

This project processes retinal fundus images using classical image processing techniques. It enhances image quality, segments blood vessels, and extracts statistical and texture-based features that can be used for retinal disease analysis.

This project is being developed as the preprocessing module for a Diabetic Retinopathy detection system. A CNN-based 5-class classifier will be integrated in the next phase.

---

## 🚀 Features

- Upload Retinal Fundus Image
- Green Channel Extraction
- CLAHE Enhancement
- Histogram Equalization
- Gaussian Filtering
- Median Filtering
- Morphological Opening
- Morphological Closing
- Adaptive Thresholding
- Blood Vessel Segmentation
- Blood Vessel Overlay
- Statistical Feature Extraction
- GLCM Texture Feature Extraction
- Processing Time Measurement
- Responsive Bootstrap UI

---

## 🔬 Image Processing Pipeline

```
Input Image
      ↓
Green Channel
      ↓
CLAHE
      ↓
Histogram Equalization
      ↓
Gaussian Filter
      ↓
Median Filter
      ↓
Morphological Opening
      ↓
Morphological Closing
      ↓
Adaptive Threshold
      ↓
Blood Vessel Segmentation
      ↓
Blood Vessel Overlay
      ↓
Feature Extraction
```

---

## 📊 Extracted Features

### Statistical Features

- Mean Intensity
- Standard Deviation
- Minimum Intensity
- Maximum Intensity
- Median Intensity
- Variance
- RMS Intensity

### Vessel Features

- Vessel Pixels
- Total Pixels
- Vessel Density (%)
- Background Pixels
- Background Density (%)

### Texture Features (GLCM)

- Contrast
- Energy
- Homogeneity
- Correlation
- Dissimilarity

---

## 🛠 Technologies Used

- Python
- Flask
- OpenCV
- NumPy
- Scikit-image
- Bootstrap 5
- HTML5
- CSS3

---

## 📂 Project Structure

```
Retinal-Image-Processing/

│── app.py
│── processing.py
│── requirements.txt
│── README.md

│── templates/
│     └── index.html

│── static/
│     ├── uploads/
│     └── outputs/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/saimul380/Retinal-Image-Processing.git
```

Move into the project

```bash
cd Retinal-Image-Processing
```

Create virtual environment

```bash
python3 -m venv venv
```

Activate environment

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install packages

```bash
pip install -r requirements.txt
```

Run

```bash
python3 app.py
```

Open

```
http://127.0.0.1:5000
```

---

## 🔮 Future Work

- CNN-based Diabetic Retinopathy Classification
- APTOS 2019 Dataset Integration
- 5-Class DR Detection
- Deep Learning Model Deployment
- Explainable AI (Grad-CAM)

---

## 👨‍💻 Developer

**Md. Saimul Hoque Sawon**

Department of Computer Science & Engineering

International Islamic University Chittagong

GitHub:
https://github.com/saimul380

---

## ⭐ Project Status

🚧 Under Development

Current Phase:
Image Processing & Feature Extraction

Next Phase:
CNN-based 5-Class Diabetic Retinopathy Detection





## Screenshots

### Home Page

<img src="/screenshots/home.png" width="800">

### Processing Result

<img src="/screenshots/output1.png" width="800">
<img src="/screenshots/output2.png" width="800">
<img src="/screenshots/processing time.png" width="800">
<img src="/screenshots/features extraction table.png" width="800">