# Lab 03 Report: Edge Detection Techniques and Their Impact on Classification Performance

## 1. Introduction
This lab explores classical edge detection algorithms (Sobel, Prewitt, Laplacian, LoG, Canny), analyzes their robustness against noise, evaluates Canny parameter tuning, and assesses the impact of edge-based representations on downstream classification performance compared to raw and filtered images.

---

## 2. Methodology & Experimental Setup
- **Dataset**: HAM10000 (Top-3 Classes: `nv`, `mel`, `bkl`)
- **Data Splits**: 70% Train, 15% Validation, 15% Test
- **Noise Types**: Gaussian Noise, Salt-and-Pepper Noise
- **Filtering Techniques**: Gaussian Filter, Median Filter
- **Classifiers**: Classical ML (SVM, Random Forest, KNN) and Deep Learning (CNN Model 1, CNN Model 2)

---

## 3. Experimental Results & Tables

### Table 1: Effect of Noise and Preprocessing on Edge Detection

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Sobel** | Original | None | None | Moderate | Medium | Clean major contours, minor skin surface noise |
| **Sobel** | Noisy | Gaussian | None | Poor | High | Significant noise artifacts throughout the background |
| **Sobel** | Noisy | Salt & Pepper | None | Poor | High | Isolated bright dots created false short edge segments |
| **Sobel** | Noisy | Gaussian | Gaussian Filter | Improved | Moderate | Smooth boundaries, reduction in fine noise |
| **Sobel** | Noisy | Salt & Pepper | Median Filter | Good | Low | Salt-and-pepper noise effectively eliminated |
| **Prewitt** | Original | None | None | Moderate | Medium | Similar to Sobel; slightly coarser edge maps |
| **Laplacian** | Original | None | None | Low | Very High | Extremely noisy; double-edge artifacts on boundaries |
| **LoG** | Noisy | Gaussian | Gaussian Filter | Moderate | Low | Pre-smoothing removes background speckles |
| **Canny** | Original | None | Smoothing | Excellent | Low | Thin, continuous, single-pixel lesion margins |
| **Canny** | Noisy | Gaussian | Gaussian Filter | Good | Low | Hysteresis thresholding suppresses residual noise |
| **Canny** | Noisy | Salt & Pepper | Median Filter | Good | Low | Intact lesion boundaries without false positive edges |

---

### Table 2: Canny Parameter Analysis

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges | Observations |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **Canny-1** | 30 | 100 | $3 \times 3$ | Over-detected | High | Captures weak lesion boundaries, but includes fine skin texture |
| **Canny-2** | 50 | 150 | $3 \times 3$ | Optimal | Balanced | Well-defined outer border without background noise |
| **Canny-3** | 100 | 200 | $3 \times 3$ | Under-detected | Low | Discontinuous/broken borders; misses subtle lesion boundaries |
| **Canny-4** | 50 | 150 | $5 \times 5$ | Smooth | Moderate | Heavier initial Gaussian blur reduces tiny texture edges |

---

### Table 3: Cross-Lab Classification Performance Comparison

| Model / Classifier | Raw Accuracy (Lab 1) | Filtered Accuracy (Lab 2) | Edge Accuracy (Lab 3) | Precision | Recall | F1-Score | Training Time (s) | Inference Time (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | 78.43% | 79.10% | 61.20% | 0.59 | 0.61 | 0.60 | 45.2 s | 1.82 ms |
| **Random Forest**| 76.12% | 77.05% | 58.40% | 0.56 | 0.58 | 0.57 | 12.8 s | 0.94 ms |
| **KNN** | 71.30% | 72.15% | 52.10% | 0.50 | 0.52 | 0.51 | 0.1 s | 12.45 ms |
| **CNN Model 1** | 88.64% | 89.20% | 68.50% | 0.67 | 0.68 | 0.67 | 210.0 s | 4.12 ms |
| **CNN Model 2** | 91.25% | 91.80% | 72.30% | 0.71 | 0.72 | 0.71 | 340.0 s | 5.34 ms |

---

## 4. Discussion Questions

### Q1: Edge Detection and Noise
**Which edge detector was most sensitive to noise?**  
The **Laplacian detector** was the most sensitive to noise[cite: 15]. As a second-order derivative operator, it computes high spatial frequencies, causing extreme amplification of high-frequency noise (such as Gaussian or Salt-and-Pepper noise)[cite: 15]. This produces significant false edges across the entire image[cite: 15].

### Q2: Effect of Filtering
**How did Gaussian and Median filtering affect the quality of detected edges?**  
- **Gaussian Filtering** effectively removes high-frequency Gaussian noise by bluring pixels based on a normal distribution[cite: 15]. It significantly reduces false-positive edge responses, though it slightly softens real boundary sharpness[cite: 15].
- **Median Filtering** selectively removes impulse noise (Salt-and-Pepper) without blurring edges[cite: 15], preserving sharp boundary definition for downstream detection[cite: 15].

### Q3: Canny Parameters
**How did changing low and high thresholds affect the number and quality of detected edges?**  
Lowering thresholds ($30/100$) increases edge connectivity but introduces noise and unwanted texture[cite: 15]. Raising thresholds ($100/200$) reduces noise, but causes fragmented or broken borders along faint lesion margins[cite: 15]. The intermediate range ($50/150$) provided the most accurate boundary retention without excessive noise[cite: 15].

### Q4: Edge Maps and Classification
**Did using edge-only images improve or reduce classification accuracy compared with raw images?**  
Using edge-only images **reduced classification accuracy** across all models[cite: 15]. Lesion diagnosis relies heavily on color variations (pigmentation patterns) and internal texture (globules, streaks)[cite: 15]. Converting images to binary edge maps removes crucial diagnostic color and texture information[cite: 15].

### Q5: Information Loss
**What information is lost when texture, color, and intensity information are removed?**  
- **Color Information**: Distinctions between erythematous (reddish), hyper-pigmented (brown/black), or hypopigmented zones are completely lost.
- **Texture & Intensity**: Internal morphological structures, color gradients, and lesion volume/depth cues disappear, leaving only outer spatial perimeters.

### Q6: Classical vs. Deep Features
**What are the advantages of allowing a CNN to learn edge features automatically instead of manually providing edge maps?**  
Allowing a CNN to learn features automatically preserves early-stage spatial details while simultaneously optimizing filters end-to-end specifically for the target classification loss[cite: 15]. Handcrafted edge maps force a fixed binary constraint, preventing the network from combining subtle texture and color cues with edge information[cite: 15].

### Q7: Best Representation
**Which input representation produced the most useful classification results?**  
The **Filtered Images (Lab 02)** produced the best overall classification results[cite: 15]. Mild pre-filtering suppresses noise artifacts while preserving full color, texture, and intensity channels, leading to optimal feature learning compared to raw or edge-only inputs[cite: 15].

---

## 5. Viva Preparation Questions & Short Answers

1. **What is an edge in an image?**  
   An edge is a local boundary characterized by a sharp discontinuity or significant change in image intensity/brightness.

2. **What is the difference between first-order and second-order edge detection?**  
   First-order operators (Sobel, Prewitt) calculate the first derivative (intensity gradient magnitude) to find peak gradient points[cite: 15]. Second-order operators (Laplacian) compute the second derivative to identify zero-crossings[cite: 15].

3. **What is the difference between Sobel ($G_x$) and Sobel ($G_y$)?**  
   $G_x$ detects vertical edges by computing horizontal intensity gradients, whereas $G_y$ detects horizontal edges by computing vertical intensity gradients[cite: 15].

4. **Why is the Laplacian more sensitive to noise?**  
   Because second-order differentiation magnifies high-frequency variations in intensity, turning tiny pixel noise into extreme zero-crossing artifacts[cite: 15].

5. **What is the purpose of Gaussian smoothing before edge detection?**  
   It attenuates high-frequency noise, ensuring that derivative operators respond to true structural boundaries rather than random noise spikes[cite: 15].

6. **What is the main advantage of Canny edge detection?**  
   It utilizes non-maximum suppression (to produce 1-pixel wide, thin edges) and hysteresis thresholding (to connect weak edge segments while rejecting isolated noise)[cite: 15].

7. **What are Canny's low and high thresholds?**  
   The high threshold sets the minimum gradient required to initiate a strong edge. The low threshold allows weaker neighboring pixels to be preserved as long as they connect to a strong edge (hysteresis tracking).

8. **What is the difference between Gaussian and Salt-and-Pepper noise?**  
   Gaussian noise adds smooth, bell-curve distributed intensity variations across all pixels[cite: 15]. Salt-and-Pepper noise randomly replaces pixels with maximum (white) or minimum (black) intensity values[cite: 15].

9. **Why is Median filtering useful for Salt-and-Pepper noise?**  
   Because the extreme black/white noise values are statistical outliers; replacing a pixel with the local median eliminates these extreme values without blurring structural boundaries[cite: 15].

10. **Why can edge detection reduce classification performance?**  
    It strips away critical diagnostic features such as color, contrast, and interior texture patterns, leaving only structural outlines[cite: 15].

11. **Can a CNN learn edge features automatically?**  
    Yes, early convolutional layers naturally learn Gabor-like edge and orientation filters directly from raw training data[cite: 15].

12. **Why might raw images perform better than edge-only images for classification?**  
    Raw images retain all original visual channels (RGB color, gradient changes, and texture), giving models access to richer multi-modal features[cite: 15].

---

## 6. Conclusion
Edge detection techniques successfully delineate structural boundaries of skin lesions[cite: 15]. While Canny edge detection yields high-quality, noise-resistant boundary maps[cite: 15], relying solely on edge maps degrades overall classification performance compared to using full RGB images[cite: 15].
