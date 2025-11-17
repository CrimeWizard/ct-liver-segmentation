#!/usr/bin/env python3
"""
Evaluate Phase 8: compute per-slice liver/tumor Dice from NIfTI predictions.

Expect:
  cascade_nifti/
    pred_multiclass_<stem>.nii.gz
    gt_multiclass_<stem>.nii.gz
    ct_<stem>.nii.gz  (optional)

Outputs (written to results/phase8_eval/):
  - phase8_metrics.csv        (per-slice)
  - summary.txt               (summary statistics)
  - liver_dice_hist.png
  - tumor_dice_hist.png
  - liver_dice_box.png
  - tumor_dice_box.png
  - best_slices.txt
  - worst_slices.txt
"""
import os
from pathlib import Path
import re
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path('.')
NIFTI_DIR = ROOT / 'code' / 'cascade_nifti'
OUT_DIR = ROOT / 'results' / 'phase8_eval'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# helper dice (works on integer/boolean arrays)
def dice_np(pred, gt, eps=1e-6):
    pred = (pred > 0).astype(np.uint64)
    gt = (gt > 0).astype(np.uint64)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum() + eps
    return float((2 * inter) / denom)

# find matching files
pred_files = sorted(NIFTI_DIR.glob('pred_multiclass_*.nii*'))
gt_files   = {p.stem.replace('gt_multiclass_', ''): p for p in NIFTI_DIR.glob('gt_multiclass_*.nii*')}

if len(pred_files) == 0:
    raise SystemExit(f"No prediction files found in {NIFTI_DIR}. Run cascade_inference_nifti.py first.")

rows = []
for p in pred_files:
    stem = p.stem.replace('pred_multiclass_', '')
    gt_path = gt_files.get(stem)
    if gt_path is None:
        # try alternative: maybe filenames don't have prefix 'pred_multiclass_...'
        possible = NIFTI_DIR / f'gt_multiclass_{stem}.nii.gz'
        if possible.exists():
            gt_path = possible
        else:
            print(f"[WARN] No matching GT for {p.name} (stem={stem}), skipping.")
            continue

    # load arrays (assume 2D single-slice or small 2D saved as 3D)
    pred_img = nib.load(str(p))
    pred_arr = pred_img.get_fdata().astype(np.int32)
    # squeeze to 2D if needed
    if pred_arr.ndim == 3 and (pred_arr.shape[2] == 1 or pred_arr.shape[0] == 1):
        pred_arr = pred_arr.squeeze()
    gt_img = nib.load(str(gt_path))
    gt_arr = gt_img.get_fdata().astype(np.int32)
    if gt_arr.ndim == 3 and (gt_arr.shape[2] == 1 or gt_arr.shape[0] == 1):
        gt_arr = gt_arr.squeeze()

    # ensure shapes match
    if pred_arr.shape != gt_arr.shape:
        print(f"[WARN] Shape mismatch for {stem}: pred {pred_arr.shape}, gt {gt_arr.shape} -> try resize by nearest")
        # as fallback, attempt a fast nearest-neighbor resize using numpy (only if dims compatible)
        try:
            import cv2
            # choose orientation: if shapes differ but both 2D, resize pred to gt
            if pred_arr.ndim == 2 and gt_arr.ndim == 2:
                pred_arr = cv2.resize(pred_arr.astype(np.uint8), (gt_arr.shape[1], gt_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                raise RuntimeError("Cannot auto-resize 3D mismatch")
        except Exception:
            print(f"[ERROR] Unable to align shapes for {stem}. Skipping.")
            continue

    # multiclass coding: 0 background, 1 liver, 2 tumor
    pred_liver = (pred_arr == 1).astype(np.uint8)
    pred_tumor = (pred_arr == 2).astype(np.uint8)
    gt_liver = (gt_arr == 1).astype(np.uint8)
    gt_tumor = (gt_arr == 2).astype(np.uint8)

    liver_d = dice_np(pred_liver, gt_liver)
    tumor_d = dice_np(pred_tumor, gt_tumor)

    rows.append({
        'id': stem,
        'liver_dice': liver_d,
        'tumor_dice': tumor_d,
        'liver_gt_sum': int(gt_liver.sum()),
        'tumor_gt_sum': int(gt_tumor.sum()),
        'liver_pred_sum': int(pred_liver.sum()),
        'tumor_pred_sum': int(pred_tumor.sum()),
    })

# Save CSV
df = pd.DataFrame(rows)
df = df.sort_values('id').reset_index(drop=True)
csv_path = OUT_DIR / 'phase8_metrics.csv'
df.to_csv(csv_path, index=False)
print(f"Wrote metrics CSV -> {csv_path}")

# Summary stats
def summarize(series):
    return {
        'count': int(series.count()),
        'mean': float(series.mean()),
        'median': float(series.median()),
        'std': float(series.std()),
        'min': float(series.min()),
        'max': float(series.max())
    }

summary = {
    'liver': summarize(df['liver_dice']),
    'tumor': summarize(df['tumor_dice'])
}

with open(OUT_DIR / 'summary.txt', 'w') as f:
    f.write("Phase 8 evaluation summary\n\n")
    f.write("Liver Dice:\n")
    for k, v in summary['liver'].items():
        f.write(f"  {k}: {v}\n")
    f.write("\nTumor Dice:\n")
    for k, v in summary['tumor'].items():
        f.write(f"  {k}: {v}\n")
print("Wrote summary.txt")

# Plots
plt.figure(figsize=(6,4))
plt.hist(df['liver_dice'].dropna(), bins=40)
plt.title('Liver Dice histogram')
plt.xlabel('Dice')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUT_DIR / 'liver_dice_hist.png')
plt.close()

plt.figure(figsize=(6,4))
plt.hist(df['tumor_dice'].dropna(), bins=40)
plt.title('Tumor Dice histogram')
plt.xlabel('Dice')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(OUT_DIR / 'tumor_dice_hist.png')
plt.close()

plt.figure(figsize=(4,6))
plt.boxplot([df['liver_dice'].dropna()], vert=True, labels=['liver'])
plt.title('Liver Dice boxplot')
plt.tight_layout()
plt.savefig(OUT_DIR / 'liver_dice_box.png')
plt.close()

plt.figure(figsize=(4,6))
plt.boxplot([df['tumor_dice'].dropna()], vert=True, labels=['tumor'])
plt.title('Tumor Dice boxplot')
plt.tight_layout()
plt.savefig(OUT_DIR / 'tumor_dice_box.png')
plt.close()

# Best/Worst lists
df_sorted_tumor = df.sort_values('tumor_dice', ascending=False)
best = df_sorted_tumor.head(10)
worst = df_sorted_tumor.tail(10)

(best_ids := list(best['id'])).sort()
(worst_ids := list(worst['id'])).sort()

with open(OUT_DIR / 'best_slices.txt', 'w') as f:
    for r in best.itertuples():
        f.write(f"{r.id}\tLiverDice={r.liver_dice:.4f}\tTumorDice={r.tumor_dice:.4f}\n")

with open(OUT_DIR / 'worst_slices.txt', 'w') as f:
    for r in worst.itertuples():
        f.write(f"{r.id}\tLiverDice={r.liver_dice:.4f}\tTumorDice={r.tumor_dice:.4f}\n")

print("Saved best/worst lists and plots in", OUT_DIR)
