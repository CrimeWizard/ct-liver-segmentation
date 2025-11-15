# code/preprocess_multiclass_masks.py
# Phase 5A — Step 0: Build 3-class masks (0 = background, 1 = liver, 2 = tumor)
# ------------------------------------------------------------

import os
import numpy as np
from tqdm import tqdm

# ---------------- CONFIG ----------------
# root folder containing subfolders (volume_pt1, volume_pt2, ...)
SRC_ROOT = "../data/segmentations"
LIVER_DIR = "../data/processed/masks/liver"
TUMOR_DIR = "../data/processed/masks/tumor"
OUT_DIR   = "../data/processed/masks_multiclass"
# ----------------------------------------

os.makedirs(LIVER_DIR, exist_ok=True)
os.makedirs(TUMOR_DIR, exist_ok=True)
os.makedirs(OUT_DIR,   exist_ok=True)

# find all .npy mask files inside any subfolder of SRC_ROOT
mask_files = []
for root, _, files in os.walk(SRC_ROOT):
    for f in files:
        if f.endswith(".npy"):
            mask_files.append(os.path.join(root, f))
mask_files = sorted(mask_files)

print(f"Found {len(mask_files)} mask files under {SRC_ROOT}")

for src_path in tqdm(mask_files, desc="Processing masks"):
    fname = os.path.basename(src_path)
    mask = np.load(src_path)

    # ----- Step 1: extract binary liver/tumor -----
    liver_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    tumor_mask = (mask == 2).astype(np.uint8)

    np.save(os.path.join(LIVER_DIR, fname), liver_mask)
    np.save(os.path.join(TUMOR_DIR, fname), tumor_mask)

    # ----- Step 2: merge into 3-class -----
    multiclass = np.zeros_like(mask, dtype=np.uint8)
    multiclass[liver_mask == 1] = 1
    multiclass[tumor_mask == 1] = 2
    np.save(os.path.join(OUT_DIR, fname), multiclass)

print(f"✅ Done! Saved {len(mask_files)} masks:")
print(f"   Liver → {LIVER_DIR}")
print(f"   Tumor → {TUMOR_DIR}")
print(f"   3-class → {OUT_DIR}")
