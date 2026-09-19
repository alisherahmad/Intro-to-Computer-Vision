# Lab 02 — Effect of Image Filtering on Skin-Lesion Classification

## Overview
This experiment measures how five classical spatial-domain filters (Average,
Gaussian, Median, Sharpening, Sobel) affect the classification performance of
three pretrained CNNs on the **HAM10000** skin-lesion dataset, compared
against an unfiltered baseline.

**Models (carried over from Lab Activity 1 / Task 01):**
- VGG16
- ResNet18
- EfficientNet-B0

All three are loaded with ImageNet-pretrained weights, the convolutional
backbone is frozen, and only the final classification layer is fine-tuned.

## Repository structure
```
Lab 02/
├── README.md                 # this file
├── answers.md                 # written answers + results table
└── ICV_BAI_032_LAB2.ipynb    # full experiment notebook
```

## Requirements
Run in Google Colab (recommended — GPU + `kagglehub` cache), or locally with:

```bash
pip install torch torchvision timm opencv-python kagglehub pandas scikit-learn tqdm matplotlib
```

A Kaggle account/API token is needed the first time `kagglehub` downloads the
`kmader/skin-cancer-mnist-ham10000` dataset (Colab handles this via its
built-in Kaggle integration; locally, place `kaggle.json` in `~/.kaggle/`).

## How to run
Open the notebook and run the cells top to bottom:

1. **Dataset preparation** — downloads HAM10000 via `kagglehub`, builds the
   `image_id → file path` map, encodes the 7 diagnosis classes
   (`bkl, nv, df, mel, vasc, bcc, akiec`) as integer labels, and creates an
   80/20 stratified train/test split (`random_state=42`).
2. **Image filtering** — defines the five OpenCV-based filters (Average,
   Gaussian, Median, Sharpening, Sobel) plus a "No Filter" baseline, and
   plots one sample image through all six for visual inspection.
3. **Dataset/model loading** — a `FilteredHAMDataset` applies the selected
   filter on the fly, then resizes to 224×224 and applies standard ImageNet
   normalization. `get_model()` loads VGG16 / ResNet18 / EfficientNet-B0
   with a frozen backbone and a replaced classification head.
4. **Training** — for each of the 6 filter conditions × 3 models (18 runs
   total), trains for 2 epochs with AdamW (`lr=1e-3`) on only the unfrozen
   head, using cross-entropy loss.
5. **Evaluation** — on the held-out test set, computes Accuracy, macro
   Precision/Recall/F1, and macro-averaged one-vs-rest AUC.
6. **Comparative analysis** — all 18 results are collected into a single
   summary table (`lab2_summary_df`), reproduced in `answers.md`.

Everything runs sequentially — no external args or config files are needed;
just run the notebook.

## Notes / limitations
- Training uses only **2 epochs per run** and a **frozen backbone**, to keep
  the 18-run comparison tractable for a lab session — absolute accuracy is
  well below what full fine-tuning would achieve.
- The current notebook reports **macro-averaged** Precision/Recall/F1 only,
  so the "F1-score" and "Macro-F1" columns in the results table are
  identical; **balanced accuracy, per-class precision/recall/F1, confusion
  matrices, and training/validation accuracy & loss curves are not yet
  generated** and would need to be added (e.g. `sklearn.metrics
  .classification_report`, `confusion_matrix`, and logging per-epoch
  train/val accuracy) to fully satisfy the "Additional Analysis"
  requirements of Task 02.
