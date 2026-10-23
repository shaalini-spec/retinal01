# 📄 Technical & Clinical Project Report: Retinal OCT Disease Classification Using Deep Transfer Learning

**Project Title:** Multi-Class Retinal Disease Detection and Diagnosis from Optical Coherence Tomography (OCT) Scans using EfficientNet-B0  
**Domain:** Biomedical Imaging, Ophthalmic Diagnostics, Deep Learning, Computer Vision  
**Author / Team:** Shaalini Chandrasekaran  
**Frameworks & Libraries:** TensorFlow 2.x, Keras 3, Streamlit, Scikit-Learn, Matplotlib, NumPy  
**Repository Status:** Git Initialized, Initial Commit Completed  

---

## Executive Summary

Retinal diseases such as Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), and Drusen represent leading causes of irreversible vision impairment and blindness worldwide. Early, precise diagnosis via Optical Coherence Tomography (OCT) enables timely clinical interventions (e.g., anti-VEGF injections, glycemic control, laser photocoagulation) that preserve visual acuity.

This project delivers an automated, high-precision deep learning diagnostic pipeline utilizing **EfficientNet-B0** transfer learning, paired with a web application built in **Streamlit** and a command-line interface for automated batch workflows. Evaluated on a benchmark test set of 1,000 independent clinical OCT images across four balanced classes, the trained model achieved an overall **test accuracy of 93.20%**, a **macro-average F1-score of 93.14%**, and a **test loss of 0.2009**.

---

## 1. Introduction & Clinical Significance

### 1.1 The Role of Optical Coherence Tomography (OCT)
Optical Coherence Tomography is a non-invasive, cross-sectional optical imaging technique analogous to ultrasound, but utilizing low-coherence near-infrared light instead of sound waves. OCT produces cross-sectional B-scans with axial resolutions on the order of 5–10 micrometers, allowing ophthalmologists to examine individual retinal layers: the inner limiting membrane, ganglion cell layer, inner and outer plexiform layers, and the retinal pigment epithelium (RPE).

### 1.2 Target Pathologies

| Condition | Pathophysiology & Etiology | Clinical Urgency & Treatment |
|---|---|---|
| **CNV (Choroidal Neovascularization)** | Pathological blood vessels originating in the choroid break through Bruch's membrane into the sub-RPE or subretinal space, often leading to hemorrhage and exudation. Principal hallmark of "wet" Age-Related Macular Degeneration (AMD). | **High / Urgent**: Requires immediate intravitreal injections of anti-VEGF agents (e.g., Ranibizumab, Aflibercept) to prevent permanent macular scarring. |
| **DME (Diabetic Macular Edema)** | Chronic hyperglycemia compromises the blood-retinal barrier in retinal capillaries, causing fluid accumulation and intraretinal cystoid spaces within the macula. Major complication of diabetic retinopathy. | **Moderate to High**: Treated via anti-VEGF therapy, focal laser photocoagulation, and systemic blood glucose/pressure control. |
| **DRUSEN** | Focal extracellular deposits of lipids, proteins, and cellular debris between the basal lamina of the RPE and the inner collagenous layer of Bruch's membrane. Defining sign of early-to-intermediate "dry" AMD. | **Moderate / Monitoring**: Requires regular surveillance, lifestyle modifications, and antioxidant supplementation (AREDS2 formulation) to monitor progression toward wet AMD or geographic atrophy. |
| **NORMAL** | Intact foveal depression, continuous parallel retinal layer laminations, uniform RPE band, and no intra- or sub-retinal fluid or structural disorganization. | **Baseline**: Standard routine screening. |

---

## 2. Dataset Architecture & Preprocessing

### 2.1 Dataset Provenance
This project utilizes the validated clinical dataset compiled by **Kermany et al. (Cell, 2018)**. The dataset consists of spectral-domain OCT (SD-OCT) B-scans acquired from clinical centers including the Shiley Eye Institute at the University of California San Diego, the California Retinal Consultants, and medical centers in China.

### 2.2 Data Distribution
- **Training Cohort:** ~108,312 images partitioned dynamically using an 80/20 train-to-validation split with fixed random seed ($SEED = 42$).
  - `CNV`: ~37,205 scans
  - `DME`: ~11,348 scans
  - `DRUSEN`: ~8,616 scans
  - `NORMAL`: ~51,140 scans
- **Test Benchmark:** Exactly 1,000 independent images uniformly balanced with 250 images per class (`CNV`, `DME`, `DRUSEN`, `NORMAL`).

### 2.3 Preprocessing & Augmentation Pipeline
1. **Resolution Standardization:** Images are resized to $224 \times 224 \times 3$ to match the standard input dimensions of the neural backbone.
2. **Data Augmentation:** Real-time stochastic augmentation applied during training to enhance invariance against patient head tilt and scanner alignment variations:
   - `layers.RandomRotation(factor=0.03)`: Up to $\pm 10.8^\circ$ rotational perturbation.
   - `layers.RandomZoom(height_factor=0.05, width_factor=0.05)`: Subtle scaling perturbations mimicking varying scan depths.
3. **Optimized I/O:** Prefetching using `tf.data.AUTOTUNE` to eliminate GPU/CPU data starvation bottlenecks.

---

## 3. Deep Learning Architecture & Training Strategy

### 3.1 Model Selection: EfficientNet-B0
Rather than training a standard convolutional network from scratch or employing excessively heavy architectures like ResNet-152 or VGG-19, this system employs **EfficientNet-B0**:
- **Compound Scaling:** Uniformly balances network depth, width, and input resolution using a fixed compound coefficient ($\phi$).
- **MBConv Blocks:** Employs mobile inverted bottleneck convolutions with squeeze-and-excitation (SE) optimization, extracting rich structural boundaries and microstructural textures with only ~4.05 million parameters.
- **Transfer Learning:** Initialized with weights pre-trained on ImageNet, capturing low-level edge features, high-contrast layer transitions, and texture gradients without requiring millions of retinal images for scratch convergence.

### 3.2 Network Topology

```text
=================================================================
Layer (type)                Output Shape              Param #   
=================================================================
input_image (InputLayer)    [(None, 224, 224, 3)]     0         
augmentation (Sequential)   (None, 224, 224, 3)       0         
efficientnetb0 (Backbone)   (None, 7, 7, 1280)        4,049,571 
global_avg_pool (GAP2D)     (None, 1280)              0         
dropout_30 (Dropout 0.3)    (None, 1280)              0         
dense_output (Dense 4, SM)  (None, 4)                 5,124     
=================================================================
Total params: 4,054,695 (15.47 MB)
Trainable params: 5,124 (20.02 KB)
Non-trainable params: 4,049,571 (15.45 MB)
=================================================================
```

### 3.3 Training Hyperparameters & Callbacks
- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, Initial learning rate $\eta = 1 \times 10^{-3}$)
- **Batch Size:** 16
- **Epochs:** Up to 8 with dynamic early-stopping
- **Callbacks:**
  - `ModelCheckpoint`: Automatically tracks `val_loss` and persists only the best weights (`outputs/best_model.keras`).
  - `EarlyStopping`: Halts training if `val_loss` fails to improve over 2 consecutive epochs and restores best model weights.
  - `ReduceLROnPlateau`: Scales learning rate down by factor of $0.3$ upon loss stagnation (`patience=1`).

---

## 4. Experimental Results & Performance Analysis

### 4.1 Quantitative Benchmark (1,000 Test Scans)

| Metric | Measured Value |
|---|---|
| **Overall Test Accuracy** | **93.20%** |
| **Test Categorical Loss** | **0.2009** |
| **Macro Average Precision** | **93.46%** |
| **Macro Average Recall** | **93.20%** |
| **Macro Average F1-Score** | **93.14%** |
| **Weighted Average F1-Score** | **93.14%** |

### 4.2 Per-Class Diagnostic Performance

| Disease Class | Precision | Recall (Sensitivity) | F1-Score | Test Samples |
|---|---|---|---|---|
| **CNV** | 87.10% | **97.20%** | 91.87% | 250 |
| **DME** | 95.63% | **96.40%** | 96.02% | 250 |
| **DRUSEN** | **95.83%** | 82.80% | 88.84% | 250 |
| **NORMAL** | 95.26% | **96.40%** | 95.83% | 250 |

### 4.3 In-Depth Metric Interpretation
1. **Extremely High Sensitivity on Critical Conditions:**
   - **CNV Recall = 97.20%**: In ophthalmology, false negatives for CNV are dangerous because untreated choroidal neovascular membranes can lead to disciform scarring within weeks. A 97.20% sensitivity ensures nearly all wet AMD cases are flagged for clinical review.
   - **DME Recall = 96.40%**: Demonstrates strong discrimination of intraretinal cystic fluid collections.
2. **High Normal Discrimination (96.40% Recall, 95.26% Precision):**
   - The model reliably distinguishes healthy retinal architecture from diseased tissue, minimizing unnecessary follow-up visits or alarm for healthy patients.
3. **Drusen vs. Early CNV Boundary:**
   - Drusen recall was 82.80%, with a very high precision of 95.83%. Analysis of misclassifications reveals that subtle drusen elevations can resemble early flat neovascular membranes or sub-RPE deposits, which is a known challenge even among expert retinal specialists.

---

## 5. Software Deliverables & Web Interface

The system is delivered with two operational modalities:

### 5.1 Interactive Web Dashboard (`app.py`)
- Built using **Streamlit**.
- Features drag-and-drop file upload supporting `.jpeg`, `.jpg`, and `.png`.
- Renders the uploaded OCT B-scan alongside real-time inference latency.
- Displays:
  - Highest confidence diagnostic class.
  - Percentage confidence score.
  - Visual breakdown bars of prediction probabilities across all four classes.
  - Explicit educational and clinical disclaimer banners.

### 5.2 Command-Line Automated Inference (`predict.py`)
- Standalone execution: `python predict.py <path_to_oct_scan>`
- Designed for rapid integration into hospital PACS (Picture Archiving and Communication System) or research batch pipelines.
- Outputs human-readable terminal summaries and percentage distributions.

---

## 6. Clinical Risk Analysis, Limitations, and Safety

1. **Software as a Medical Device (SaMD) Classification:**
   - This implementation is intended for investigational and research purposes only. Prior to deployment in hospital or clinic workflows, prospective clinical trials and regulatory clearances (FDA 510(k) or De Novo, CE Mark under MDR EU 2017/745) are mandatory.
2. **Device & Manufacturer Generalization:**
   - Retinal OCT imaging varies across hardware vendors (e.g., Zeiss Cirrus, Heidelberg Spectralis, Topcon 3D OCT, Optoview). Cross-scanner domain adaptation techniques should be evaluated when expanding deployment to different clinical sites.
3. **Artifact Handling:**
   - Poor pupil dilation, cataracts, or patient eye motion can induce shadow artifacts or signal attenuation on OCT scans. A quality-assurance filtering step should precede diagnostic classification in future releases.

---

## 7. Roadmap & Recommendations for Future Iterations

1. **Fine-Tuning Deeper Layers:** Unfreezing the top 20–30 layers of EfficientNet-B0 with a smaller learning rate ($1 \times 10^{-5}$) to learn domain-specific optical interference features.
2. **Visual Explainability (Grad-CAM):** Integrate Gradient-weighted Class Activation Mapping into `app.py` so clinicians can visually confirm that the network is focusing on intraretinal fluid pockets or RPE elevations rather than scanner borders.
3. **Segmentation Integration:** Combine multi-class classification with a U-Net / SAM backbone to segment fluid volume (subretinal fluid vs. intraretinal fluid) in nanoliters.
4. **Cloud Containerization:** Containerize via Docker and deploy on AWS ECS or Google Cloud Run for scalable remote tele-ophthalmology screening.

---

## 8. Conclusion

This project successfully implemented, evaluated, and packaged a deep learning retinal OCT diagnostic platform:
- Model achieved **93.20% accuracy** on a standardized multi-class benchmark.
- Developed full source code with modular separation between training (`train.py`), inference (`predict.py`), and web deployment (`app.py`).
- Prepared standard repository configurations, clean git commit history, and comprehensive clinical documentation.

---
*Report compiled on October 23, 2026.*
