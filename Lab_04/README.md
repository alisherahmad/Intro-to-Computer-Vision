# Lab 4 — Skin Lesion Boundary Detection Using Canny Edge Detection

**Course:** Introduction to Computer Vision (ICV) · **Notebook:** `ICV_BAI_032_LAB4.ipynb` (runs entirely in Google Colab, no uploads needed)

## Overview
A classical computer-vision pipeline that tries to find the boundary of a skin lesion and measure its area and perimeter:

**Original → Grayscale → Gaussian filter → Canny edges → Lesion boundary → Area & perimeter**

## Data
Five melanocytic-nevus dermoscopy images from the public HAM10000 dataset (`marmal88/skin_cancer` on Hugging Face), streamed directly in Colab and resized so the longest side is at most 600 px. All pixel measurements are at that scale.

## Method
1. **Preprocessing:** grayscale, then Gaussian blur (5×5 kernel, σ = 1.4).
2. **Canny:** three threshold pairs, **50–100, 100–200 and 150–250**.
3. **Best threshold:** for each image the notebook picks the setting whose resulting outline lies on the strongest edges (mean gradient along the boundary).
4. **Boundary:** morphological closing and dilation join broken edge fragments, `findContours` extracts the outer contours, and the largest contour that does not touch the image border is kept and filled.
5. **Measurements:** area = pixels inside the filled mask; perimeter = contour arc length (`cv2.arcLength`).

## Results

### Lesion area and perimeter

| Image   | Best Filter   | Edge Method   |   Area (pixels) |   Perimeter (pixels) |
|:--------|:--------------|:--------------|----------------:|---------------------:|
| Image 1 | Gaussian      | Canny 50-100  |            1128 |                529.2 |
| Image 2 | Gaussian      | Canny 50-100  |             803 |                378.2 |
| Image 3 | Gaussian      | Canny 100-200 |             296 |                226.5 |
| Image 4 | Gaussian      | Canny 50-100  |               0 |                  0   |
| Image 5 | Gaussian      | Canny 50-100  |             668 |                235.4 |

**These numbers are not reliable lesion measurements.** Checking the overlays against the original images:
- **Image 1:** the outline follows a long hair strand, not the lesion.
- **Image 2:** the outline follows a hair strand next to the lesion.
- **Image 3:** the outline is a short sliver on the right edge of the lesion, so the area is far smaller than the real lesion.
- **Image 4:** no edges survived, so no boundary was found (area 0).
- **Image 5:** the outline is a small fragment at the top edge of the lesion.

None of the five outlines covers its lesion. The reported areas and perimeters describe whatever edge fragment was selected, not the lesion itself. On these images, Canny at 50–250 does not produce a usable lesion boundary.

### Final comparison (mean over the 5 images)

| Method           |   Noise (fragments) |   Edge Quality |   Boundary IoU |   Overall (mean rank, 1=best) |
|:-----------------|--------------------:|---------------:|---------------:|------------------------------:|
| Original + Sobel |              1141.8 |          0.121 |          0.332 |                         6.667 |
| Original + Canny |               122.2 |          0.14  |          0.199 |                         5.333 |
| Average + Sobel  |               319.2 |          0.18  |          0.549 |                         3.333 |
| Average + Canny  |                 4.8 |          0.208 |          0.001 |                         3.333 |
| Gaussian + Sobel |               382.4 |          0.173 |          0.534 |                         4.667 |
| Gaussian + Canny |                 8.4 |          0.192 |          0.001 |                         4.333 |
| Median + Sobel   |               631.8 |          0.176 |          0.48  |                         5     |
| Median + Canny   |                14.6 |          0.265 |          0.025 |                         3.333 |

Qualitative summary (lab's final table):

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | Poor (most fragments) | Poor (lowest) | Fair (IoU 0.33) | Worst (rank 6.7) |
| Original + Canny | Fair | Poor | Poor (IoU 0.20) | Below average |
| Average + Sobel | Poor | Fair | **Best (IoU 0.55)** | Tied best (3.3) |
| Average + Canny | **Very good** (4.8 fragments) | Fair | Fails (IoU ≈ 0) | Tied best (3.3) |
| Gaussian + Sobel | Poor | Fair | Good (IoU 0.53) | Mid (4.7) |
| Gaussian + Canny | **Very good** (8.4) | Fair | Fails (IoU ≈ 0) | Mid (4.3) |
| Median + Sobel | Poor | Fair | Fair (IoU 0.48) | Below average (5.0) |
| Median + Canny | Very good (14.6) | **Best** (0.265) | Fails (IoU 0.03) | Tied best (3.3) |

How the metrics are defined (there are no ground-truth masks, so these are proxies against a reference mask made with inverse Otsu thresholding):
- **Noise Handling:** number of tiny isolated edge fragments (fewer is better).
- **Edge Quality:** fraction of edge pixels within 3 px of the reference boundary.
- **Boundary Detection:** IoU between the detected lesion mask and the reference mask.

How to read the table:
- **Canny** was far cleaner (5–15 fragments vs 300–1100 for Sobel) but found almost no lesion boundary (IoU ≈ 0 for Average/Gaussian), because the 50–100 thresholds were too high for the soft lesion edges.
- **Sobel** gave noisy edge maps, but closing and filling them usually covered the lesion region, giving IoU of 0.48–0.55 with Average and Gaussian smoothing.
- **Overall rank:** the best methods are tied (3.3), and Average + Sobel is the only one that scored well on boundary detection.
- **Reference bias:** the reference mask comes from intensity thresholding, so it naturally agrees better with region-like edge maps (Sobel). The ranking is not a fair judgement of Canny, and all scores are low in absolute terms.

## Questions

**1. Why is Gaussian filtering applied before Canny detection?**
Edge detectors respond to any intensity change, and noise, skin texture and fine hairs are intensity changes too. Without smoothing, Canny turns them into many short false edges. A Gaussian filter replaces each pixel with a distance-weighted average of its neighbours, which suppresses this high-frequency noise while keeping the stronger lesion border. The comparison shows the effect: Original + Canny produced 122 edge fragments, versus 8 for Gaussian + Canny.

**2. How did the three Canny threshold settings affect the result?**
Thresholds set how strong a gradient must be to count as an edge. These lesions have soft, gradual borders, so their gradients are weak.
- **50–100:** the most sensitive of the three. It still gave only sparse fragments, mostly hairs and a few pieces of the lesion edge.
- **100–200:** almost nothing survived (Image 1's map has a single short segment).
- **150–250:** the edge map was essentially empty.

Higher thresholds therefore removed noise but also removed the lesion boundary. None of the three produced a closed ring around a lesion, and the measured areas came from small edge fragments.

**3. Which threshold produced the best lesion boundary?**
No setting produced a good lesion boundary on these images. **50–100** was selected for four of the five images (and 100–200 for Image 3), but only because it was the least bad: the selection rule picks the setting with the strongest edge along the outline, not the correct one. Judged by the overlays, none of the three captured the lesion.

**4. Why are edges useful for detecting skin lesions?**
A lesion usually differs from the surrounding skin in pigment and brightness, so its border shows up as an intensity discontinuity. Edges give a compact description of its shape without needing a trained model, and from the outline we can compute area, perimeter and border irregularity, which matter clinically (the "Border" and "Diameter" parts of the ABCD rule).

**5. What problems did you observe in detecting the lesion boundary?**
- **Thresholds too high for soft borders:** the lesions fade gradually into the skin, so Canny found almost no edges on them (Image 4 gave an empty edge map and area 0).
- **Hairs:** long hairs create strong, continuous edges that were selected as the "boundary" in Images 1 and 2.
- **Wrong contour selected:** the selection step chose small fragments (Images 3 and 5), so the outline did not cover the lesion.
- **Background structures:** Image 5 has a dark circular vignette and many small red/brown spots that compete with the lesion.
- **Fixed thresholds:** one absolute threshold pair does not suit images with different contrast.
- **Sobel:** it gave thick, noisy edge maps with hundreds of fragments, though its filled region overlapped the reference better.
- **Metrics:** with no ground-truth masks, the proxy scores are only rough guides.

**6. How could your method be improved?**
- **Lower, adaptive thresholds:** for example Otsu- or median-based Canny, instead of fixed 50–250 values.
- **Hair removal:** remove hair first (for example black-hat filtering plus inpainting).
- **Better colour channel:** detect edges on saturation or the L\* channel of L\*a\*b\* instead of plain grayscale.
- **Crop the vignette** before processing.
- **Prefer central, compact contours:** score candidate contours by closeness to the image centre and shape, so hairs and fragments are rejected.
- **Closing alternatives:** use active contours, watershed or GrabCut to close gaps in a broken edge ring.
- **Proper evaluation:** compare against the ISIC ground-truth masks with Dice/IoU, and consider a learned segmentation model such as U-Net.

## How to run
1. Open `ICV_BAI_032_LAB4.ipynb` in Google Colab.
2. Runtime → Run all.
3. The last cell writes `results_table.csv` and `final_comparison.csv`.

## Limitations
Only five images were used and no ground-truth masks were available. In this run the pipeline failed to outline the lesion in all five images, so the area and perimeter values should be read as properties of the selected edge fragments, not as lesion measurements.
