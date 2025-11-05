# code/eval_unet_baseline.py
# Phase 3 — Evaluation & Visualization on test set

import os, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.liver_dataset import get_loaders
from models.unet import UNet
from utils.metrics import dice_coeff, iou_score, pixel_accuracy

# -------------------------------
# Config
# -------------------------------
CFG = {
    "base_dir": "../data/processed",
    "checkpoint_path": "checkpoints/unet_baseline_best.pth",
    "save_dir": "docs/figures/phase3_predictions",
    "num_visuals": 6,     # number of slices to save overlays
    "threshold": 0.5,
}

os.makedirs(CFG["save_dir"], exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running evaluation on {device}")

# -------------------------------
# Load data & model
# -------------------------------
_, _, test_loader = get_loaders(CFG["base_dir"], batch_size=1, num_workers=2, augment_train=False)

model = UNet(in_channels=1, out_channels=1, base_ch=32, dropout=0.5, use_transpose=False).to(device)
model.load_state_dict(torch.load(CFG["checkpoint_path"], map_location=device))
model.eval()

# -------------------------------
# Evaluate metrics
# -------------------------------
dice_scores, iou_scores, acc_scores = [], [], []

with torch.no_grad():
    for imgs, masks, ids in tqdm(test_loader, desc="Testing"):
        imgs, masks = imgs.to(device), masks.to(device)

        # ---- forward pass ----
        logits = model(imgs)  # use raw logits (no sigmoid)

        # ---- metrics (same as during training/validation) ----
        dice_scores.append(dice_coeff(logits, masks))
        iou_scores.append(iou_score(logits, masks))
        acc_scores.append(pixel_accuracy(logits, masks))


print("\n=== Test Metrics ===")
print(f"Dice: {np.mean(dice_scores):.4f}")
print(f"IoU:  {np.mean(iou_scores):.4f}")
print(f"Acc:  {np.mean(acc_scores):.4f}")

# -------------------------------
# Save a few overlay visualizations
# -------------------------------
print(f"\nSaving {CFG['num_visuals']} overlay examples to {CFG['save_dir']} ...")
count = 0
for imgs, masks, ids in test_loader:
    imgs, masks = imgs.to(device), masks.to(device)
    preds = torch.sigmoid(model(imgs))
    preds = (preds > CFG["threshold"]).float()

    img_np  = imgs[0,0].cpu().numpy()
    mask_np = masks[0,0].cpu().numpy()
    pred_np = preds[0,0].cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(img_np, cmap="gray"); ax[0].set_title("Original CT"); ax[0].axis("off")
    ax[1].imshow(mask_np, cmap="gray"); ax[1].set_title("Ground Truth"); ax[1].axis("off")
    ax[2].imshow(img_np, cmap="gray")
    ax[2].imshow(pred_np, alpha=0.4, cmap="Reds"); ax[2].set_title("Prediction"); ax[2].axis("off")

    plt.tight_layout()
    base_id = os.path.splitext(os.path.basename(ids[0]))[0]
    save_path = os.path.join(CFG["save_dir"], f"{base_id}_overlay.png")

    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    count += 1
    if count >= CFG["num_visuals"]:
        break

print("✅ Evaluation complete.")
print("Overlay images saved in:", CFG["save_dir"])
