# Project Short Summary  
**Cascaded Liver & Tumor Segmentation Pipeline**  
**Author:** Youssef Hamed

This project implements an 8-phase medical imaging pipeline for liver and tumor segmentation from abdominal CT scans. I built the entire system myself, covering preprocessing, ROI extraction, model training, cascade inference, evaluation, and reporting.

The dataset used is the **Liver Tumor Segmentation** dataset from Kaggle (derived from LiTS).  
Due to storage and hardware limits, I used **~50 out of the 130 volumes**, which influenced tumor model performance.

---

## What I Built
- Full preprocessing pipeline (HU clipping, normalization, resampling)  
- Automated liver ROI and tumor ROI extraction  
- 2D liver U-Net (successful)  
- 2D tumor U-Net (attempted; failed to learn)  
- Cascade inference producing liver → tumor predictions  
- Evaluation tools (Dice, slice detection, best/worst slices, CSVs, plots)  
- Fully reproducible workflow with organized code and outputs

---

## Results

### ✔ Liver segmentation — **successful**
- Mean Dice: **0.715**  
- Median Dice: 0.798  
- Max Dice: 0.955  
The model is stable and reliable across the dataset.

### ❌ Tumor segmentation — **failed**
- Mean Dice: **0.0029**  
- Detection rate: ~34%  

Reasons:
- very small tumors  
- severe class imbalance  
- only ~50 volumes used instead of 130  
- limited GPU capacity on my laptop  
- shallow 2D architecture  
- ROI cropping edge cases  

---

## What Worked Well
- Strong liver segmentation  
- Complete 8-phase cascade  
- Clean NIfTI workflows  
- Detailed evaluation outputs  
- Best/worst slice analysis  
- Reproducible and well-structured codebase

---

## Future Work
With access to a stronger GPU, I plan to:
- use the **full 130 LiTS volumes**  
- train deeper or 3D U-Nets  
- increase input resolution  
- adopt better loss functions (Dice + Focal)  
- improve ROI strategies  
- add attention and multi-scale features  

These upgrades should significantly improve tumor segmentation.

---

## Conclusion
The pipeline is complete and functional.  
Liver segmentation performs well.  
Tumor segmentation failed due to dataset size, hardware limitations, and the difficulty of the task.  
I documented the results honestly and prepared a clear plan for future improvement once stronger hardware becomes available.
