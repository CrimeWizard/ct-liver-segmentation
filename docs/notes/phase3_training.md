## Phase 3 – Baseline U-Net (Training Summary)

**Date:** Nov 2025  
**Environment:** fraunhofer (Python 3.10, PyTorch 2.7.1 + cu118)  
**GPU:** NVIDIA GeForce MX450 (2 GB VRAM)

| Setting | Value |
|:--|:--|
| Image size | 256 × 256 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Epochs | 33 |
| Loss | BCE + Dice (0.5 : 0.5) |
| Augmentation | Flip, rotation |
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Dropout | 0.5 |
| Mixed precision | Yes |
| Best validation Dice | **0.9427** |
| IoU | 0.9010 |
| Pixel accuracy | 0.9948 |
| Checkpoint | `checkpoints/unet_baseline_best.pth` |


### Combined vs. Class-wise Dice

In this baseline, a **single binary segmentation model** was trained to detect
the entire liver region (including any tumors) as one foreground class
(`mask > 0`).  
Consequently, the reported Dice of **0.93–0.94** reflects the **overall
foreground overlap** rather than separate liver and tumor Dice scores.

This design choice was intentional for Phase 3:
- It reproduces the *simplified baseline* used in several LiTS reimplementations
  (e.g., Chlebus et al., 2018 pre-liver cascade stage).
- It reduces GPU memory usage and simplifies the architecture
  for reproducibility on limited hardware (NVIDIA MX450 2 GB).
- It establishes a clean, high-quality baseline before extending
  to **multi-class segmentation (background / liver / tumor)** in Phase 4.

Later phases will compute **separate Dice scores** for liver and tumor
after reintroducing class-specific masks and uncertainty modeling.



### Comparison with Published Baseline — Cangalović et al. (2024)

| Parameter            | Cangalović et al. (2024)              | MEVIS Baseline (Youssef, 2025)            | Difference / Comment                                                      |
|:---------------------|:--------------------------------------|:------------------------------------------|:---------------------------------------------------------------------------|
| Model                | Bayesian U-Net (MC-Dropout 0.2–0.5)  | Deterministic U-Net (Dropout 0.5, OFF at inference) | They sample dropout at test for uncertainty; you use the non-Bayesian baseline. |
| Resolution           | 256 × 256                             | 256 × 256                                 | identical                                                                  |
| Loss                 | BCE + Dice                            | BCE + Dice (0.5 : 0.5)                    | identical formulation                                                      |
| Optimizer / LR       | Adam (1e-4)                           | Adam (1e-4)                               | identical                                                                  |
| Batch size           | 8                                     | 4                                         | halved due to 2 GB MX450 VRAM                                             |
| Training epochs      | 60–80                                 | 33                                        | fewer, but reached same convergence (Dice ≈ 0.93)                          |
| Augmentation         | Flip, rotation, brightness, elastic   | Flip, rotation                            | lighter aug for speed/VRAM                                                |
| Evaluation metrics   | Dice, IoU, **uncertainty**            | Dice, IoU, Accuracy                       | uncertainty planned in Phase 4                                            |
| Reported Dice (mean) | 0.94 (liver) / 0.86 (tumor)           | **0.93 (combined)**                       | very close overall despite smaller model                                   |
