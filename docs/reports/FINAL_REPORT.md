# Final Project Report  
**Cascaded Liver & Tumor Segmentation**

**Author:** Youssef Hamed  
**Hardware:** local development laptop

---

## 1. Introduction

This project implements an 8-phase pipeline for liver and liver-tumor segmentation on abdominal CT scans.  
I developed every component of the system from scratch, including preprocessing, U-Net models, ROI extraction, cascade inference, and evaluation.

The goal was to build a clear, modular research pipeline and document both the successful phases and the limitations honestly.

---

## 2. Dataset

I used the **Liver Tumor Segmentation** dataset from **Kaggle**, which originates from the **LiTS (Liver Tumor Segmentation Challenge)** dataset.

- Full dataset size: **130 CT volumes**  
- For this project, due to storage and hardware limits, I used **around 50 volumes**  
- Each volume includes:  
  - A CT scan  
  - A segmentation mask (background / liver / tumor)

Important dataset characteristics:
- Tumors are typically small and irregular  
- Many slices contain *no tumor*, causing severe class imbalance  
- Volumes are very large (100–300 MB each), increasing GPU load  

**Note:**  
The dataset is not included in this repository due to license restrictions.

---

## 3. Pipeline Overview (8 Phases)

### **Phase 1 — Dataset Preparation**
Validated shapes, orientations, and standardized all NIfTI filenames.

### **Phase 2 — Preprocessing**
HU clipping, intensity normalization, resampling, and slice filtering.

### **Phase 3 — Liver Model Training**
2D U-Net (384×384), BCE + Dice loss, with basic augmentations.

### **Phase 4 — Liver Inference**
Full-volume prediction followed by connected component filtering.

### **Phase 5 — Tumor ROI Extraction**
Used the liver mask to crop liver regions and isolate tumor-relevant areas.

### **Phase 6 — ROI Validation**
Checked ROI integrity, alignment, and tumor slice preservation.

### **Phase 7 — Tumor Model Training**
Multiple training attempts were made, but the model did not learn meaningful tumor features.

### **Phase 8 — Final Evaluation**
Computed Dice scores, detection rates, best/worst slices, and exported summary reports.

All phases executed successfully from start to finish.

---

## 4. Implementation Details

- **Frameworks:** Python, PyTorch, NumPy, nibabel, scikit-image, scipy, pandas, matplotlib  
- **Models:** 2D U-Nets for liver and tumor segmentation  
- **Losses:** BCE + Dice (liver), Dice/Focal/weighted BCE experiments for tumor  
- **Preprocessing:** HU windowing, normalization, resampling, ROI cropping  
- **Evaluation:** Slice-wise Dice, detection rate, and automated best/worst slice analysis  

All evaluation files are saved in `results/phase8_eval/`.

---

## 5. Results

### ✔ Liver Segmentation — Successful

| Metric | Value |
|--------|--------|
| **Mean Dice** | **0.715** |
| Median | 0.798 |
| Max | 0.955 |
| Std | 0.221 |

The liver model performs reliably and produces clean masks for ROI extraction.

---

### ❌ Tumor Segmentation — Failed

| Metric | Value |
|--------|--------|
| **Mean Dice** | **0.0029** |
| Median | 0.0 |
| Max | 0.320 |
| GT tumor slices | 2093 |
| Predicted tumor slices | 711 |
| Detection rate | ~34% |
| Missed slices | 1382 |

The tumor model outputs near-zero predictions for most slices.  
The evaluation pipeline is correct — the model simply failed to learn tumor features under the project’s constraints.

---

## 6. Why the Tumor Model Failed

Main contributing factors:

### **1. Hardware limitations**
My laptop GPU could not support:
- deeper U-Nets  
- 3D architectures  
- higher resolutions  
- larger batch sizes  

### **2. Limited dataset usage**
I used **~50 of the 130 LiTS volumes**.  
This reduced the number of tumor examples and weakened the model’s ability to generalize.

### **3. Extremely small tumors**
Many tumors occupy just a few pixels, requiring deeper or multi-scale networks.

### **4. Class imbalance**
Most slices contain no tumors → strong bias toward predicting background.

### **5. 2D architecture limits**
The model lacked volumetric context necessary for tumor detection.

### **6. ROI cropping edge cases**
Some tumors near liver boundaries may have been cut off.

These combined factors prevented the tumor model from learning meaningful features, despite multiple training attempts.

---

## 7. What Worked Well

- Full 8-phase pipeline implemented and reproducible  
- Strong liver segmentation results  
- Clean NIfTI processing and ROI extraction  
- Reliable evaluation with plots and data summaries  
- Best/worst slice identification  
- Organized code structure ready for extension  

---

## 8. Future Work (once I have access to stronger hardware)

I plan to significantly extend this work using a more powerful GPU.  
The key improvements I will introduce are:

### **1. Use the full LiTS dataset (all 130 volumes)**  
This will increase the number of tumor examples and help reduce overfitting.

### **2. Train deeper and more advanced models**
- Attention U-Net  
- ResNet-based U-Net  
- U-Net++  
- Swin-UNet  

### **3. Move to 3D segmentation**
3D U-Nets or patch-based 3D models will capture volumetric tumor structure.

### **4. Better loss functions**
Dice + Focal loss for extreme imbalance.

### **5. Higher resolution**
Move from 384 → 512/576 to capture small lesions.

### **6. Improved ROI strategies**
Use adaptive bounding boxes or larger context padding.

These improvements are expected to dramatically increase tumor segmentation performance.

---

## 9. Deliverables & Files

- `code/` – training, inference, evaluation  
- `data/` – processed CTs/masks (dataset not included)  
- `results/` – metrics, plots, detection stats  
- `docs/` – final report + short summary  

---

## 10. Conclusion

I successfully built a full cascaded liver–tumor segmentation pipeline from scratch.  
The liver segmentation model performs well.  
The tumor model failed due to hardware constraints, limited dataset usage, and the difficulty of the task.  
This failure is documented honestly and accurately.

The pipeline is clean, reproducible, and ready for major improvements once I have access to stronger hardware and can use the full 130-volume LiTS dataset.

