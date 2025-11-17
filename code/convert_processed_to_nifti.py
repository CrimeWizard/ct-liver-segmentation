#!/usr/bin/env python3
"""
Convert processed .npy images and multiclass masks into MEVISdraw-ready NIfTI files.

- Expects folder structure like:
  data/processed/images/*.npy
  data/processed/masks_multiclass/*.npy

Outputs:
  data/processed/nifti/images/ct_<name>.nii.gz
  data/processed/nifti/masks_multiclass/mask_multiclass_<name>.nii.gz
  data/processed/nifti/masks_binary/mask_liver_<name>.nii.gz
  data/processed/nifti/masks_binary/mask_tumor_<name>.nii.gz

Run:
  python code/convert_processed_to_nifti.py
"""
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm

ROOT = Path("data/processed")
IMG_DIR = ROOT / "images"
MCL_DIR = ROOT / "masks_multiclass"

OUT_ROOT = ROOT / "nifti"
OUT_IMG = OUT_ROOT / "images"
OUT_MCL = OUT_ROOT / "masks_multiclass"
OUT_BIN = OUT_ROOT / "masks_binary"

AFFINE = np.eye(4, dtype=np.float32)  # simple identity affine

def ensure(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def save_nifti_array(arr, out_path):
    nii = nib.Nifti1Image(arr.astype(np.uint8), AFFINE)
    nib.save(nii, str(out_path))

def convert_images():
    if not IMG_DIR.exists():
        print("No images folder:", IMG_DIR); return
    ensure(OUT_IMG)
    files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".npy")])
    if not files:
        print("No .npy images found in", IMG_DIR); return
    for fn in tqdm(files, desc="Images"):
        arr = np.load(IMG_DIR / fn)
        # ensure 2D shape (H,W)
        if arr.ndim == 3 and arr.shape[0] in (1,):
            arr = arr.squeeze(0)
        out_name = f"ct_{Path(fn).stem}.nii.gz"
        save_nifti_array(arr, OUT_IMG / out_name)

def convert_multiclass_masks():
    if not MCL_DIR.exists():
        print("No masks_multiclass folder:", MCL_DIR); return
    ensure(OUT_MCL); ensure(OUT_BIN)
    files = sorted([f for f in os.listdir(MCL_DIR) if f.endswith(".npy")])
    if not files:
        print("No .npy masks found in", MCL_DIR); return
    for fn in tqdm(files, desc="Masks"):
        arr = np.load(MCL_DIR / fn)  # expected values 0/1/2
        # ensure 2D
        if arr.ndim == 3 and arr.shape[0] in (1,):
            arr = arr.squeeze(0)
        stem = Path(fn).stem
        # Save multiclass raw
        save_nifti_array(arr, OUT_MCL / f"mask_multiclass_{stem}.nii.gz")
        # Save binary liver
        liver = (arr == 1).astype(np.uint8)
        save_nifti_array(liver, OUT_BIN / f"mask_liver_{stem}.nii.gz")
        # Save binary tumor
        tumor = (arr == 2).astype(np.uint8)
        save_nifti_array(tumor, OUT_BIN / f"mask_tumor_{stem}.nii.gz")

def main():
    ensure(OUT_ROOT)
    convert_images()
    convert_multiclass_masks()
    print("Done. NIfTI outputs written to:", OUT_ROOT)

if __name__ == "__main__":
    main()
