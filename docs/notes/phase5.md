# Phase 5 — Baseline Multi-Class U-Net (Background / Liver / Tumor)

## 1️⃣ Experiment Metadata

| Item | Value |
|:--|:--|
| **Date / Phase** | Phase 5 — Baseline Multi-Class Segmentation |
| **Hardware** | NVIDIA GeForce MX450 (2 GB VRAM), CUDA 12.x |
| **Frameworks** | PyTorch 2.x + torch.amp + tqdm |
| **Image size** | 256 × 256 |
| **Batch size** | 4 |
| **Epochs** | 30 |
| **Optimizer** | Adam (lr = 1e-4, weight_decay = 0) |
| **Scheduler** | ReduceLROnPlateau (factor 0.5, patience 3) |
| **Loss** | CrossEntropyLoss (equal weights per class) |
| **Augmentation** | Random flip H/V, rotation (90°, 180°, 270°), intensity jitter ±10 % |
| **Normalization** | Min–max per slice → [0, 1] |
| **AMP** | Enabled (mixed precision) |
| **Early-stop patience** | 5 epochs |
| **Input channels** | 1 |
| **Output channels** | 3 (classes: 0 = background, 1 = liver, 2 = tumor) |
| **Dataset split** | train 70 % / val 15 % / test 15 % (same JSON as Phase 3) |

---

## 2️⃣ Key Results (Validation Set)

| Epoch | Train Loss | Dice (Liver) | Dice (Tumor) | Mean Dice | LR |
|:--|:--:|:--:|:--:|:--:|:--:|
| 01 | 0.57 | 0.20 | 0.22 | 0.21 | 1e-4 |
| 05 | 0.17 | 0.56 | 0.22 | 0.39 | 1e-4 |
| 10 | 0.12 | 0.65 | 0.22 | 0.43 | 1e-4 |
| 20 | 0.10 | 0.73 | 0.22 | 0.47 | 1e-4 |
| 25 | 0.08 | 0.75 | 0.22 | 0.48 | 5e-5 |
| 30 | 0.08 | **0.77** | **0.22** | **0.50** | 5e-5 |

**Best checkpoint:** `checkpoints/unet_multiclass_best.pth` (saved at epoch 29)  
**Loss curve:** `docs/figures/phase5_loss_curve.png`  
**Dice curve:** `docs/figures/phase5_dice_curve.png`

---

## 3️⃣ Qualitative Observations

| Aspect | Observation |
|:--|:--|
| **Convergence** | Loss dropped smoothly 0.57 → 0.08; no overfitting seen. |
| **Liver Dice** | Grew steadily to ≈ 0.78; organ boundaries clear and consistent. |
| **Tumor Dice** | Flat ≈ 0.21; network failed to differentiate small tumor regions. |
| **Learning rate schedule** | Reduced to 5e-5 after epoch 23 → stabilized training. |
| **Computation time** | ≈ 1000 s per epoch (~17 min); 30 epochs ≈ 8 h total. |

---

## 4️⃣ Analysis & Interpretation

- **Liver segmentation:** successful — Dice ≈ 0.78 matches reported 2-D U-Net baselines for LiTS at 256².  
- **Tumor segmentation:** poor — Dice ≈ 0.21, identical to unweighted single-stage results reported in *Chlebus et al., 2018*.  
- **Root cause:** severe class imbalance (background ≫ liver ≫ tumor) and small tumor regions → under-represented gradients.  
- **Planned remedies:**  
  1. Weighted and hybrid loss functions (Phase 6A).  
  2. ROI-based cascade with tumor-only U-Net (Phase 6B).  
- **Conclusion:** Phase 5 achieved a robust multi-class baseline model for liver, confirming the expected imbalance issue and establishing a reference point for Phase 6.

---

## 5️⃣ Artifacts

| Path | Description |
|:--|:--|
| `checkpoints/unet_multiclass_best.pth` | Best model weights |
| `logs/phase5_multiclass_log.csv` | Full training metrics |
| `docs/figures/phase5_loss_curve.png` | Loss over epochs |
| `docs/figures/phase5_dice_curve.png` | Mean Dice over epochs |
| `datasets/multiclass_dataset.py` | Final dataset loader with resize (256²) |

---

## 6️⃣ Next Phase Plan

Proceed to **Phase 6 – Improving Tumor Segmentation**

- **6A:** Weighted CrossEntropy + Dice Hybrid Loss  
- **6B:** Cascaded Two-Stage U-Net (Liver → Tumor)  

Both attempts will be documented and compared to this Phase 5 baseline.
