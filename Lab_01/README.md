# Lab 01: Skin Cancer Classification (ISIC Dataset)

Benchmarking transfer learning models, deep feature extractors, and classical machine learning classifiers on the 9-class ISIC skin cancer dataset[cite: 3].

## Experimental Results

### Table 1: Transfer Learning Model Performance
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 48.31 | 48.29 | 48.61 | 41.66 | 87.32 |
| **resnet18** | 33.90 | 15.13 | 27.78 | 19.08 | 79.36 |
| **efficientnet_b0** | 47.46 | 42.87 | 47.92 | 42.56 | 82.34 |

### Table 2: Deep Feature Extraction with Classical Classifiers
| Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 50.00 | 47.74 | 50.00 | 44.43 |
| **Decision Tree** | 23.73 | 17.19 | 25.46 | 19.11 |
| **Random Forest** | 37.29 | 32.83 | 39.58 | 28.52 |
| **K-Nearest Neighbors (KNN)** | 28.81 | 33.03 | 29.63 | 28.32 |
| **Linear SVM** | 45.76 | 45.09 | 46.53 | 42.63 |
| **RBF-SVM** | 49.15 | 52.82 | 49.31 | 44.43 |
| **XGBoost** | 43.22 | 38.15 | 44.44 | 37.53 |

### Table 3: Computational Efficiency & Performance
| Model | Parameters (M) | Model Size (MB) | FLOPs (G) | Inference Time (ms) | Accuracy (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **vgg16** | 134.30 | 512.32 | 30.93 | 67.49 | 48.31 |
| **resnet18** | 11.18 | 42.74 | 3.65 | 55.49 | 33.90 |
| **efficientnet_b0** | 3.98 | 15.73 | 0.77 | 62.94 | 47.46 |
