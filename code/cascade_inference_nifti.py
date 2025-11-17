import os
import numpy as np
import torch
import cv2
from pathlib import Path
import nibabel as nib

from models.unet_binary_roi import UNetBinaryROI
from models.unet_multiclass import UNetMultiClass

# ============================================================
# CONFIG  (USE SAME PATHS AS OLD SCRIPT)
# ============================================================
BASE_DIR = "../data/processed"
IMG_DIR  = f"{BASE_DIR}/images"
MASK_DIR = f"{BASE_DIR}/masks_multiclass"

LIVER_CKPT = "checkpoints/unet_multiclass_best.pth"
TUMOR_CKPT = "checkpoints/phase6b_tumor_roi_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Where to save NIfTI predictions
NIFTI_DIR = "cascade_nifti"
os.makedirs(NIFTI_DIR, exist_ok=True)

AFFINE = np.eye(4, dtype=np.float32)

# ============================================================
# LOAD MODELS
# ============================================================
print("Loading liver model...")
liver_model = UNetMultiClass(
    in_channels=1,
    n_classes=3,
    base_ch=32,
    dropout=0.5,
    use_transpose=False
)
liver_model.load_state_dict(torch.load(LIVER_CKPT, map_location=DEVICE))
liver_model.to(DEVICE).eval()

print("Loading tumor model...")
tumor_model = UNetBinaryROI(in_channels=1, base_ch=32)
tumor_model.load_state_dict(torch.load(TUMOR_CKPT, map_location=DEVICE))
tumor_model.to(DEVICE).eval()


# ============================================================
# SAVE NIFTI
# ============================================================
def save_nii(arr, out_path):
    nii = nib.Nifti1Image(arr.astype(np.uint8), AFFINE)
    nib.save(nii, out_path)


# ============================================================
# MAIN LOOP
# ============================================================
test_ids = sorted(os.listdir(IMG_DIR))

for sid in test_ids:

    # Load CT slice
    img = np.load(os.path.join(IMG_DIR, sid))
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Load GT (512→256)
    gt_full = np.load(os.path.join(MASK_DIR, sid))
    gt = cv2.resize(gt_full, (256, 256), interpolation=cv2.INTER_NEAREST)

    # GT binary masks
    gt_liver = (gt == 1).astype(np.uint8)
    gt_tumor = (gt == 2).astype(np.uint8)

    # ========== LIVER PREDICTION ==========
    x = torch.from_numpy(img_norm).float()[None, None].to(DEVICE)
    with torch.no_grad():
        logits = liver_model(x)
    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    liver_mask = (pred == 1).astype(np.uint8)

    # ========== ROI ==========
    ys, xs = np.where(liver_mask == 1)
    if len(xs) == 0:
        print(f"{sid}: NO LIVER FOUND")
        continue

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    if (y2 - y1 < 5) or (x2 - x1 < 5):
        print(f"{sid}: ROI TOO SMALL")
        continue

    roi = img_norm[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (256, 256))

    # ========== TUMOR ROI PRED ==========
    x_roi = torch.from_numpy(roi_resized)[None, None].float().to(DEVICE)
    with torch.no_grad():
        tlog = tumor_model(x_roi)
    tprob = torch.sigmoid(tlog).squeeze().cpu().numpy()
    tbin = (tprob > 0.5).astype(np.uint8)

    # Back-resize tumor
    tbin_back = cv2.resize(
        tbin.astype(np.uint8),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST
    )

    # ========== FULL TUMOR MASK (256x256) ==========
    full_tumor_mask = np.zeros_like(img_norm, dtype=np.uint8)
    full_tumor_mask[y1:y2, x1:x2] = tbin_back

    # ========== FINAL MULTICLASS (0/1/2) ==========
    final_pred = np.zeros_like(img_norm, dtype=np.uint8)
    final_pred[liver_mask == 1] = 1
    final_pred[full_tumor_mask == 1] = 2

    # ========== SAVE NIFTI ONLY ==========
    stem = Path(sid).stem

    save_nii((img_norm * 255).astype(np.uint8),
             os.path.join(NIFTI_DIR, f"ct_{stem}.nii.gz"))

    save_nii(liver_mask,
             os.path.join(NIFTI_DIR, f"pred_liver_{stem}.nii.gz"))

    save_nii(full_tumor_mask,
             os.path.join(NIFTI_DIR, f"pred_tumor_{stem}.nii.gz"))

    save_nii(final_pred,
             os.path.join(NIFTI_DIR, f"pred_multiclass_{stem}.nii.gz"))

    save_nii(gt,
             os.path.join(NIFTI_DIR, f"gt_multiclass_{stem}.nii.gz"))

    print(f"[OK] Saved NIfTI for {sid}")
