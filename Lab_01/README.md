# Lab 01: Skin Cancer Classification (ISIC Dataset)

Benchmarking transfer learning models, deep feature extractors, and classical machine learning classifiers on the 9-class ISIC skin cancer dataset[cite: 3].

## Experimental Results

### Table 1: Transfer Learning Model Performance
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 48.31[cite: 3] | 48.29[cite: 3] | 48.61[cite: 3] | 41.66[cite: 3] | 87.32[cite: 3] |
| **resnet18** | 33.90[cite: 3] | 15.13[cite: 3] | 27.78[cite: 3] | 19.08[cite: 3] | 79.36[cite: 3] |
| **efficientnet_b0** | 47.46[cite: 3] | 42.87[cite: 3] | 47.92[cite: 3] | 42.56[cite: 3] | 82.34[cite: 3] |

### Table 2: Deep Feature Extraction with Classical Classifiers
| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 50.00[cite: 3] | 47.74[cite: 3] | 50.00[cite: 3] | 44.43[cite: 3] |
| **Decision Tree** | 23.73[cite: 3] | 17.19[cite: 3] | 25.46[cite: 3] | 19.11[cite: 3] |
| **Random Forest** | 37.29[cite: 3] | 32.83[cite: 3] | 39.58[cite: 3] | 28.52[cite: 3] |
| **K-Nearest Neighbors (KNN)** | 28.81[cite: 3] | 33.03[cite: 3] | 29.63[cite: 3] | 28.32[cite: 3] |
| **Linear SVM** | 45.76[cite: 3] | 45.09[cite: 3] | 46.53[cite: 3] | 42.63[cite: 3] |
| **RBF-SVM** | 49.15[cite: 3] | 52.82[cite: 3] | 49.31[cite: 3] | 44.43[cite: 3] |
| **XGBoost** | 43.22[cite: 3] | 38.15[cite: 3] | 44.44[cite: 3] | 37.53[cite: 3] |

### Table 3: Computational Efficiency & Performance
| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 134.30[cite: 3] | 512.32[cite: 3] | 30.93[cite: 3] | 67.49[cite: 3] | 48.31[cite: 3] |
| **resnet18** | 11.18[cite: 3] | 42.74[cite: 3] | 3.65[cite: 3] | 55.49[cite: 3] | 33.90[cite: 3] |
| **efficientnet_b0** | 3.98[cite: 3] | 15.73[cite: 3] | 0.77[cite: 3] | 62.94[cite: 3] | 47.46[cite: 3] |
