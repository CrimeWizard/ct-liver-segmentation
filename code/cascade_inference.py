import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from models.unet_binary_roi import UNetBinaryROI
from models.unet_multiclass import UNetMultiClass

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = "../data/processed"
IMG_DIR  = f"{BASE_DIR}/images"
MASK_DIR = f"{BASE_DIR}/masks_multiclass"

LIVER_CKPT = "checkpoints/unet_multiclass_best.pth"
TUMOR_CKPT = "checkpoints/phase6b_tumor_roi_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
# DICE FUNCTION
# ============================================================
def dice(pred, gt, eps=1e-6):
    inter = (pred * gt).sum()
    denom = pred.sum() + gt.sum() + eps
    return (2 * inter) / denom


# ============================================================
# OUTPUT DIR
# ============================================================
SAVE_DIR = "cascade_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# MAIN LOOP
# ============================================================
test_ids = sorted(os.listdir(IMG_DIR))

for sid in test_ids:

    # ----------------------------------------------------------
    # LOAD IMAGE + NORMALIZE (MUST MATCH TRAINING)
    # ----------------------------------------------------------
    img = np.load(os.path.join(IMG_DIR, sid))  # 256×256

    # Normalization (same as Phase 5 training)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Load GT (512×512 originally)
    gt_full = np.load(os.path.join(MASK_DIR, sid))

    # Resize GT to 256×256
    gt = cv2.resize(gt_full, (256, 256), interpolation=cv2.INTER_NEAREST)

    gt_liver = (gt == 1).astype(np.uint8)
    gt_tumor = (gt == 2).astype(np.uint8)


    # ----------------------------------------------------------
    # 1) LIVER PREDICTION
    # ----------------------------------------------------------
    x = torch.from_numpy(img).float()[None, None].to(DEVICE)
    with torch.no_grad():
        logits = liver_model(x)

    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    liver_mask = (pred == 1).astype(np.uint8)


    # ----------------------------------------------------------
    # 2) COMPUTE LIVER ROI
    # ----------------------------------------------------------
    ys, xs = np.where(liver_mask == 1)

    if len(xs) == 0:
        print(f"{sid}: no liver found")
        continue

    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    # ROI must have non-zero size
    if y2 - y1 < 5 or x2 - x1 < 5:
        print(f"{sid}: ROI too small — skipping")
        continue

    roi = img[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (256, 256))


    # ----------------------------------------------------------
    # 3) TUMOR ROI PREDICTION
    # ----------------------------------------------------------
    x_roi = torch.from_numpy(roi_resized)[None, None].float().to(DEVICE)

    with torch.no_grad():
        tlog = tumor_model(x_roi)

    tprob = torch.sigmoid(tlog).squeeze().cpu().numpy()
    tbin = (tprob > 0.5).astype(np.uint8)

    # Resize tumor mask back into ROI
    tbin_back = cv2.resize(
        tbin.astype(np.uint8),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_NEAREST
    )


    # ----------------------------------------------------------
    # 4) RECONSTRUCT FULL TUMOR MASK
    # ----------------------------------------------------------
    full_tumor_mask = np.zeros_like(img, dtype=np.uint8)
    full_tumor_mask[y1:y2, x1:x2] = tbin_back


    # ----------------------------------------------------------
    # 5) FINAL COMPOSITE MASK
    # ----------------------------------------------------------
    final_pred = np.zeros_like(img, dtype=np.uint8)
    final_pred[liver_mask == 1] = 1
    final_pred[full_tumor_mask == 1] = 2


    # ----------------------------------------------------------
    # 6) DICE SCORES
    # ----------------------------------------------------------
    d_liver = float(dice(liver_mask, gt_liver))
    d_tumor = float(dice(full_tumor_mask, gt_tumor))

    print(f"{sid} | Liver Dice={d_liver:.3f} | Tumor Dice={d_tumor:.3f}")


    # ----------------------------------------------------------
    # 7) VISUALIZATION
    # ----------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(img, cmap='gray'); ax[0].set_title("Input")
    ax[1].imshow(gt, cmap='gray');  ax[1].set_title("GT (256x256)")
    ax[2].imshow(final_pred, cmap='gray'); ax[2].set_title("Prediction")
    ax[3].imshow(full_tumor_mask, cmap='gray'); ax[3].set_title("Tumor Only")

    for a in ax: a.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, sid.replace(".npy", ".png")))
    plt.close()
