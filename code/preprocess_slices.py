#!/usr/bin/env python3
"""
Phase 2 – Preprocessing & Slice Extraction
Fraunhofer MEVIS-style reproducible data pipeline
Author: Youssef (MEVIS Project)
"""

import os, json, random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

# ---------------- CONFIG ----------------
RAW_PATHS = [
    "../data/volume_pt1",
    "../data/volume_pt2",
    "../data/volume_pt3",
    "../data/volume_pt4",
    "../data/volume_pt5",
]
SEG_PATH = "../data/segmentations"

OUT_PATH = "../data/processed"
IMG_SIZE = (256, 256)
CLIP_RANGE = (-100, 400)
SPLIT_RATIOS = (0.7, 0.15, 0.15)
random.seed(42)
# ----------------------------------------

def ensure_dirs():
    os.makedirs(f"{OUT_PATH}/images", exist_ok=True)
    os.makedirs(f"{OUT_PATH}/masks", exist_ok=True)
    os.makedirs(f"{OUT_PATH}/splits", exist_ok=True)

def load_nifti(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def normalize(volume):
    v = np.clip(volume, *CLIP_RANGE)
    return (v - np.mean(v)) / (np.std(v) + 1e-8)

def preprocess_case(vol_path, seg_path, case_id):
    vol = normalize(load_nifti(vol_path))
    seg = load_nifti(seg_path)
    slices = []
    for z in range(vol.shape[0]):
        if seg[z].sum() == 0:
            continue  # skip empty slices
        img = resize(vol[z], IMG_SIZE, preserve_range=True, anti_aliasing=True)
        mask = resize(seg[z], IMG_SIZE, order=0, preserve_range=True, anti_aliasing=False)
        img = img.astype(np.float32)
        mask = mask.astype(np.uint8)
        out_name = f"volume_{case_id}_{z:03d}.npy"
        np.save(f"{OUT_PATH}/images/{out_name}", img)
        np.save(f"{OUT_PATH}/masks/{out_name}", mask)
        slices.append(out_name)
    return slices

def split_ids(ids):
    random.shuffle(ids)
    n = len(ids)
    n_train = int(SPLIT_RATIOS[0] * n)
    n_val = int(SPLIT_RATIOS[1] * n)
    train, val, test = ids[:n_train], ids[n_train:n_train+n_val], ids[n_train+n_val:]
    with open(f"{OUT_PATH}/splits/train_ids.json", "w") as f: json.dump(train, f)
    with open(f"{OUT_PATH}/splits/val_ids.json", "w") as f: json.dump(val, f)
    with open(f"{OUT_PATH}/splits/test_ids.json", "w") as f: json.dump(test, f)

def preview_random_slice():
    import random
    imgs = os.listdir(f"{OUT_PATH}/images")
    if not imgs: return
    fn = random.choice(imgs)
    img = np.load(f"{OUT_PATH}/images/{fn}")
    mask = np.load(f"{OUT_PATH}/masks/{fn}")
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1); plt.imshow(img, cmap="gray"); plt.title("CT Slice")
    plt.subplot(1,2,2); plt.imshow(img, cmap="gray"); plt.imshow(mask, alpha=0.4); plt.title("Overlay")
    plt.tight_layout()
    plt.savefig("../docs/figures/preprocessing_preview.png", dpi=200)
    plt.close()

def main():
    ensure_dirs()
    pairs = []
    for path in RAW_PATHS:
        if not os.path.exists(path):
            continue
        for f in os.listdir(path):
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                idx = ''.join(ch for ch in f if ch.isdigit())
                seg_candidates = [
                    f"segmentation-{idx}.nii",
                    f"segmentation_{idx}.nii",
                    f"segmentation{idx}.nii",
                ]
                seg_path = None
                for s in seg_candidates:
                    s_full = os.path.join(SEG_PATH, s)
                    if os.path.exists(s_full):
                        seg_path = s_full
                        break
                if seg_path:
                    pairs.append((os.path.join(path, f), seg_path, idx))

    print(f"Found {len(pairs)} paired volumes.")
    all_ids = []
    for vpath, spath, cid in tqdm(pairs, desc="Processing volumes"):
        ids = preprocess_case(vpath, spath, cid)
        all_ids.extend(ids)
    split_ids(all_ids)
    preview_random_slice()
    print("✅ Phase 2 complete.")

if __name__ == "__main__":
    main()
