# 👁️ Retinal OCT Disease Classification using EfficientNet-B0

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Model%20Accuracy-93.20%25-brightgreen.svg)]()

An end-to-end deep learning system for automated detection and multi-class diagnosis of retinal diseases from Optical Coherence Tomography (OCT) images using **Transfer Learning with EfficientNet-B0** and an interactive **Streamlit web application**.

---

## 📌 Table of Contents
- [Clinical Background](#-clinical-background)
- [Key Features](#-key-features)
- [Model Architecture & Methodology](#-model-architecture--methodology)
- [Performance & Benchmark Results](#-performance--benchmark-results)
- [Project Structure](#-project-structure)
- [Installation & Quick Start](#-installation--quick-start)
- [Usage Guide](#-usage-guide)
  - [1. Streamlit Web Interface](#1-launching-the-streamlit-web-app)
  - [2. Command-Line Inference](#2-command-line-inference)
  - [3. Model Training](#3-model-training)
- [Dataset Details](#-dataset-details)
- [Medical Disclaimer](#-medical-disclaimer)
- [References & Citations](#-references--citations)

---

## 🩺 Clinical Background

Optical Coherence Tomography (OCT) is a non-invasive, micrometer-resolution optical imaging technique widely used in ophthalmology to visualize cross-sectional structures of the human retina. Timely detection of macular diseases is paramount to preventing irreversible vision loss.

This system classifies retinal OCT B-scans into four distinct clinical categories:

| Class | Full Medical Name | Clinical Description |
|---|---|---|
| **CNV** | Choroidal Neovascularization | Growth of abnormal, leaky blood vessels beneath the retina (characteristic of wet Age-Related Macular Degeneration / AMD). Requires prompt anti-VEGF therapy. |
| **DME** | Diabetic Macular Edema | Retinal swelling and intraretinal fluid accumulation resulting from microvascular breakdown in diabetic retinopathy. |
| **DRUSEN** | Drusen Deposits | Acellular lipid/proteinaceous deposits under the retinal pigment epithelium (RPE), indicating early-to-intermediate dry AMD. |
| **NORMAL** | Healthy Retina | Well-preserved foveal depression and intact continuous retinal layer architecture with no fluid or subretinal pathology. |

---

## ✨ Key Features

- **Transfer Learning with EfficientNet-B0**: Leverages compound-scaled convolutional features pre-trained on ImageNet for superior parameter efficiency and convergence.
- **Robust Generalization**: Achieves **93.20% classification accuracy** on a standardized 1,000-image multi-class test benchmark.
- **Data Augmentation**: Incorporates subtle rotational ($\pm 3\%$) and zoom ($\pm 5\%$) augmentations to safeguard against scanner variances.
- **Interactive Web App**: Modern Streamlit web dashboard allowing clinicians and researchers to drag-and-drop OCT images, inspect predicted classes, view confidence scores, and observe probability distributions across all conditions.
- **Automated CLI Prediction**: Lightweight Python script for batch pipelines or terminal-based quick inference.
- **Dynamic Learning Rate & Early Stopping**: Employs `ReduceLROnPlateau` and `EarlyStopping` to prevent overfitting.

---

## 🧠 Model Architecture & Methodology

```
Input OCT Scan (224x224x3)
           │
           ▼
┌──────────────────────────────────────────┐
│  Data Augmentation Layer                 │
│  • RandomRotation(factor=0.03)           │
│  • RandomZoom(factor=0.05)               │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  EfficientNet-B0 Backbone (ImageNet)     │
│  • Frozen feature extractor              │
│  • Compound scaling (depth, width, res)  │
└──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Classification Head                     │
│  • GlobalAveragePooling2D                │
│  • Dropout (rate = 0.3)                  │
│  • Dense (4 units, Softmax activation)   │
└──────────────────────────────────────────┘
           │
           ▼
Output: Class Probabilities [CNV, DME, DRUSEN, NORMAL]
```

---

## 📊 Performance & Benchmark Results

The model was evaluated against an independent, balanced test dataset of **1,000 OCT scans** (250 scans per clinical condition).

### Classification Report

```text
              Precision    Recall  F1-Score   Support

         CNV     0.8710    0.9720    0.9187       250
         DME     0.9563    0.9640    0.9602       250
      DRUSEN     0.9583    0.8280    0.8884       250
      NORMAL     0.9526    0.9640    0.9583       250

    Accuracy                         0.9320      1000
   Macro Avg     0.9346    0.9320    0.9314      1000
Weighted Avg     0.9346    0.9320    0.9314      1000
```

- **Overall Test Accuracy**: **93.20%**
- **Test Loss**: **0.2009**
- **Confusion Matrix**: Saved in [`outputs/confusion_matrix.png`](outputs/confusion_matrix.png)

---

## 📂 Project Structure

```bash
retinal/
├── app.py                      # Interactive Streamlit Web Application
├── predict.py                  # Standalone CLI inference script
├── train.py                    # Complete training, validation, and evaluation pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (excludes large data/envs)
├── outputs/
│   ├── best_model.keras        # Trained EfficientNet-B0 model checkpoint
│   ├── class_names.json        # Class label definitions
│   ├── classification_report.txt # Test evaluation metrics
│   └── confusion_matrix.png    # Evaluation confusion matrix plot
└── dataset/                    # Kermany et al. dataset (local only, excluded from git)
    └── CellData/OCT/
        ├── train/              # Training set (split 80/20 train/val)
        └── test/               # Balanced test set (250 images/class)
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/retinal-oct-classifier.git
cd retinal-oct-classifier
```

### 2. Set Up Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage Guide

### 1. Launching the Streamlit Web App
Run the following command to start the web dashboard locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`. You can upload any standard OCT image (`.jpg`, `.jpeg`, `.png`) to inspect instant classifications with confidence metrics and probability breakdowns.

### 2. Command-Line Inference
To classify a single OCT scan directly from the terminal:
```bash
python predict.py path/to/oct_image.jpeg
```

**Example output:**
```text
Loading saved model...

--- OCT Classification Result ---
Image: dataset/CellData/OCT/test/CNV/CNV-1016042-1.jpeg
Predicted class: CNV
Model confidence: 99.41%

All class probabilities:
CNV: 99.41%
DME: 0.32%
DRUSEN: 0.18%
NORMAL: 0.09%
```

### 3. Model Training
To retrain the model on the full or custom OCT dataset:
```bash
python train.py
```
The script will:
1. Load and augment scans from `dataset/CellData/OCT/train`.
2. Train with Adam optimizer and checkpoint the best weights based on `val_loss`.
3. Evaluate against `dataset/CellData/OCT/test`.
4. Output the confusion matrix image and classification report into `outputs/`.

---

## 📚 Dataset Details

This project uses the publicly available retinal OCT dataset from:
- **Source**: Kermany et al., *Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images*, Mendeley Data / Cell (2018).
- **Structure**:
  - Total Training Scans: ~108,300+ images across CNV, DME, DRUSEN, and NORMAL.
  - Test Scans: 1,000 images (250 scans per class).

> **Note**: Due to size constraints (~8.1 GB), the dataset directory is ignored by Git. To obtain the dataset, download it from [Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2) and extract it into `dataset/CellData/OCT/`.

---

## ⚠️ Medical Disclaimer

> **IMPORTANT**: This software is developed strictly for academic, educational, and research purposes. It is **not** an FDA/CE-approved medical device and must **not** be used as a primary diagnostic tool in clinical settings without physician supervision and appropriate regulatory certification. Always consult licensed ophthalmologists or healthcare professionals for clinical diagnoses and treatment decisions.

---

## 📖 References & Citations

1. **Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. L., ... & Zhang, K. (2018).** *Identifying medical diagnoses and treatable diseases by image-based deep learning.* **Cell**, 172(5), 1122-1131. [DOI: 10.1016/j.cell.2018.02.010](https://doi.org/10.1016/j.cell.2018.02.010)
2. **Tan, M., & Le, Q. (2019).** *EfficientNet: Rethinking model scaling for convolutional neural networks.* **ICML 2019**, PMLR, 6105-6114.
