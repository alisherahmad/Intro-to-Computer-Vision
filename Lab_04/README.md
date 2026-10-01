# Lab 4 — Skin Lesion Boundary Detection Using Canny Edge Detection

**Course:** Introduction to Computer Vision (ICV) · **Notebook:** `ICV_BAI_032_LAB4.ipynb` (runs entirely in Google Colab, no uploads needed)

## Overview
A simple classical computer-vision pipeline that finds the boundary of a skin lesion and measures its area and perimeter:

**Original → Grayscale → Gaussian filter → Canny edges → Lesion boundary → Area & perimeter**

## Data
Five melanocytic-nevus dermoscopy images from the public HAM10000 dataset (`marmal88/skin_cancer` on Hugging Face), streamed directly inside Colab. Images are resized so the longest side is at most 600 px, so all pixel measurements are at that scale.

## Method
1. **Preprocessing:** convert to grayscale, then Gaussian blur (5×5 kernel, σ = 1.4).
2. **Canny:** run with three threshold pairs, **5–15, 10–30 and 20–60**, and compare the edge maps.
3. **Best threshold:** chosen by visually comparing the three maps and the resulting outlines (see Q3).
4. **Boundary:** morphological closing and dilation join broken edge fragments into a closed shape. `findContours` then extracts the outer contours, and the contour with the largest area (penalised if it touches the image border or lies far from the centre) is kept as the lesion. The mask is filled, eroded back, and opened to remove thin hair-like tails.
5. **Measurements:** area = number of pixels inside the filled mask; perimeter = contour arc length (`cv2.arcLength`).

> The lab suggests 50–100, 100–200 and 150–250 only as examples. On these images those settings found almost no closed lesion outline (see Q2), so lower thresholds were used.

## Results

### Lesion area and perimeter
<!-- Paste the table from results.md / results_table.csv here -->

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---|
| Image 1 | Gaussian | Canny 10–30 | | |
| Image 2 | Gaussian | Canny 10–30 | | |
| Image 3 | Gaussian | Canny 10–30 | | |
| Image 4 | Gaussian | Canny 10–30 | | |
| Image 5 | Gaussian | Canny 10–30 | | |

### Final comparison (mean over the 5 images)
<!-- Paste the table from final_comparison.csv here -->

| Method | Noise (fragments, lower = better) | Edge Quality | Boundary IoU | Overall (mean rank, 1 = best) |
|---|---|---|---|---|
| Original + Sobel | | | | |
| Original + Canny | | | | |
| Average + Sobel | | | | |
| Average + Canny | | | | |
| Gaussian + Sobel | | | | |
| Gaussian + Canny | | | | |
| Median + Sobel | | | | |
| Median + Canny | | | | |

No ground-truth masks were used, so these scores are proxies, measured against a reference lesion mask made with inverse Otsu thresholding:
- **Noise Handling:** number of tiny isolated edge fragments (fewer is better).
- **Edge Quality:** fraction of edge pixels within 3 px of the reference boundary.
- **Boundary Detection:** IoU between the detected lesion mask and the reference mask.

## Questions

**1. Why is Gaussian filtering applied before Canny detection?**
Edge detectors respond to intensity changes, and noise, skin texture and fine hairs are also intensity changes. Without smoothing, Canny turns them into many short false edges. A Gaussian filter averages each pixel with its neighbours (weighted by distance), which suppresses this high-frequency noise while keeping the stronger lesion border. In the comparison, Original + Canny produced far more edge fragments than the smoothed versions.

**2. How did the three Canny threshold settings affect the result?**
Thresholds control how strong a gradient must be to count as an edge. The lesion borders in these images are soft, so their gradients are weak.
- **Very high thresholds (the lab's 50–100 and above):** almost nothing survived. Only a few hair strands and isolated fragments appeared, no closed ring formed around the lesion, and the measured area was near zero.
- **5–15 (too low):** edges appeared everywhere, including skin texture, hair and background spots. Closing then merged them into very large regions covering much of the image.
- **10–30:** a mostly continuous ring formed around the lesion with limited background clutter.
- **20–60:** cleaner, but the ring broke on faint or fuzzy borders, giving small or empty masks.

**3. Which threshold produced the best lesion boundary?**
**Canny 10–30** (with the 5×5, σ = 1.4 Gaussian filter). It gave the best balance: strong enough to ignore most texture, sensitive enough to follow the soft lesion border and form a closed outline for the clearer lesions (Images 1–3). Confirm this against your own edge-map figures before submitting.

**4. Why are edges useful for detecting skin lesions?**
A lesion usually differs from the surrounding skin in pigment and brightness, so its border appears as a strong intensity discontinuity. Edges therefore give a compact description of the lesion's shape without needing a trained model. From that outline we can compute area, perimeter and border irregularity, which matter clinically (the "Border" and "Diameter" parts of the ABCD rule).

**5. What problems did you observe in detecting the lesion boundary?**
- **Hair:** long hairs create strong, continuous edges that the pipeline can mistake for the lesion boundary or attach to it as thin tails.
- **Soft, fuzzy borders:** pigment fades gradually into the skin, so gradients are weak and the edge ring has gaps.
- **Faint lesions:** a low-contrast lesion (Image 4) was largely missed or merged with hair.
- **Background structures:** a dark circular vignette and many small red or brown spots (Image 5) produced competing edges and a wrong outline.
- **Fixed thresholds:** one pair of absolute thresholds does not suit every image, because contrast varies.
- **Sobel:** it gave thick, noisy edge maps with many fragments.

**6. How could your method be improved?**
- Remove hair first (for example black-hat filtering plus inpainting, as in DullRazor).
- Use adaptive thresholds, such as Otsu- or median-based Canny, instead of fixed values.
- Detect edges on a better channel (saturation, or L\*a\*b\* L) instead of plain grayscale.
- Crop out the dark vignette before processing.
- Replace the closing step with active contours, watershed or GrabCut, which can close gaps in a ring.
- Evaluate against the ISIC ground-truth masks with Dice/IoU instead of proxy metrics, and consider a learned segmentation model such as U-Net.

## How to run
1. Open `ICV_BAI_032_LAB4.ipynb` in Google Colab.
2. Runtime → Run all.
3. The last cell writes `results_table.csv`, `final_comparison.csv` and `results.md`.

## Limitations
Only five images were used and no ground-truth masks were available, so all conclusions are qualitative or based on proxy metrics. Images 4 and 5 are failure cases that the classical pipeline does not handle reliably.
