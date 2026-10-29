# Lab Mid: HOG vs SIFT on Blood-Cell Images

**Name:** Ali Sher Ahmad  
**Reg No:** 032  
**Course:** Computer Vision Lab

## Task
Build two classical blood-cell classifiers, **HOG + SVM** and **SIFT + SVM**, that run on a normal laptop, and find out which one is more reliable when images are rotated or noisy.

## Files
| File | Description |
|---|---|
| `032_ALISHER_LabMid.ipynb` | Executed notebook with all outputs (Q1–Q5) |
| `032_ALISHER_LabMid.pdf` | PDF copy of the notebook |

## Personal settings (from reg no 032)
| Setting | Rule | Value |
|---|---|---|
| Random seed | Last two digits | **32** |
| HOG cell size | Last digit 2 is even | **8 × 8** |
| SIFT grid | Last digit 2 is in 0–4 | **4 × 4 (16 keypoints)** |

## Setup
- **Dataset:** BloodMNIST (`medmnist`), size 64
- **Classes:** eosinophil (1), monocyte (5), neutrophil (6), platelet (7)
- **Split:** 100 train / 30 validation / 30 test images per class → 400 / 120 / 120
- **Preprocessing:** grayscale, resized to 128 × 128
- **HOG:** 9 orientations, 8 × 8 cells, 2 × 2 cells per block, L2-Hys normalisation
- **SIFT:** fixed 4 × 4 grid of keypoints (size 32) over the image, descriptors from `sift.compute`
- **Classifier:** StandardScaler + LinearSVC, C chosen from {0.1, 1, 10} on the validation set

## Results

### Feature lengths
| Features | Calculation | Length |
|---|---|---|
| HOG | 15 × 15 blocks × (2 × 2 cells × 9 bins) | **8100** |
| Grid SIFT | 16 keypoints × 128 | **2048** |

### Validation (C chosen on validation, best C = 0.1 for both)
| Model | Val accuracy | Macro-F1 |
|---|---|---|
| HOG + SVM | 0.725 | 0.721 |
| SIFT + SVM | 0.708 | 0.712 |

### Normal SIFT detection (training set)
Average 71.8 keypoints per image (min 7, max 324), with 0 images having zero keypoints. Because the number of keypoints varies, the descriptors can't be stacked into one fixed-length vector for the SVM, which is why the fixed grid was used.

### Robustness on the test set (frozen models)
| Model | Clean | Rotated 20° | Noisy (σ = 0.05) |
|---|---|---|---|
| HOG + SVM | 0.725 | 0.717 | 0.250 |
| SIFT + SVM | **0.775** | **0.775** | **0.783** |

## Conclusion
- **SIFT + SVM was the more reliable model.** It scored highest on clean, rotated and noisy test images, and its accuracy hardly changed.
- **Noise:** HOG + SVM dropped to 0.250, which is chance level for 4 classes. HOG's small 8 × 8 cells pick up false gradients from noise in the smooth background. Each SIFT descriptor pools gradients over a 32-pixel patch, so the noise averages out.
- **Rotation (20°):** both models were barely affected. HOG lost 0.008 (one test image) and SIFT lost nothing.
- **Most confused classes:** eosinophil and neutrophil for both models. These two cells differ mainly in granule colour, which is lost after converting to grayscale.

## How to run
1. Open the notebook in Google Colab.
2. Runtime → Run all. The first cell installs `medmnist`, and the dataset downloads automatically.
3. To use another registration number, change `REG_NO` in the first cell.

## Libraries
`medmnist`, `numpy`, `opencv-python` (`cv2`), `scikit-image`, `scikit-learn`, `matplotlib`
