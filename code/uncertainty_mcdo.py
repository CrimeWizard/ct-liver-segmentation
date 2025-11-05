# code/uncertainty_mcdo.py
# Phase 4 — Step 1: MC-Dropout Uncertainty Estimation (Final Version)
# -------------------------------------------------------------------
# Loads the trained U-Net, performs MC-Dropout inference, and saves:
#   • Predictive mean probability maps
#   • Aleatoric and epistemic uncertainty maps
#   • Overlay PNGs for visualization
#   • CSV summary of per-slice uncertainty statistics

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from datasets.liver_dataset import get_loaders
from models.unet import UNet

# ---------------- CONFIG ----------------
CKPT = "checkpoints/unet_baseline_best.pth"
OUT_DIR = "docs/figures/phase4_uncertainty"  # outputs (PNGs, npy, csv)
N_SAMPLES = 20        # number of MC-Dropout runs
BATCH_SIZE = 1        # per-slice inference
THRESH = 0.5          # binarization threshold
MAX_SLICES = None     # process all slices
# ----------------------------------------


def enable_mc_dropout(model):
    """Enable dropout layers during inference."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def forward_samples(model, x, n):
    """Run n stochastic forward passes with dropout enabled."""
    preds = []
    for _ in range(n):
        y = torch.sigmoid(model(x))
        preds.append(y.cpu().numpy())
    return np.stack(preds, axis=0)  # [n, b, 1, H, W]


def entropy(p, eps=1e-6):
    """Binary entropy per pixel."""
    p = np.clip(p, eps, 1 - eps)
    out = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    out[np.isnan(out)] = 0
    return out


def overlay_preview(ct, pred_bin, aleatoric, epistemic, save_path):
    """Save a 4-panel overlay (CT, prediction, aleatoric, epistemic)."""
    plt.figure(figsize=(12, 4))
    titles = ["CT", "Prediction", "Aleatoric", "Epistemic"]
    images = [ct, pred_bin, aleatoric, epistemic]
    cmaps = ["gray", "gray", "hot", "hot"]

    for i, (img, title, cmap) in enumerate(zip(images, titles, cmaps), 1):
        ax = plt.subplot(1, 4, i)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare CSV summary
    summary_csv = os.path.join(OUT_DIR, "uncertainty_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slice_id",
            "mean_aleatoric", "std_aleatoric",
            "mean_epistemic", "std_epistemic",
            "path_pred", "path_aleatoric", "path_epistemic", "path_overlay"
        ])

    # Load data
    _, _, test_loader = get_loaders(
        "../data/processed",
        batch_size=BATCH_SIZE,
        num_workers=2,
        augment_train=False
    )

    # Load model
    model = UNet(
        in_channels=1,
        out_channels=1,
        base_ch=32,
        dropout=0.5,
        use_transpose=False
    ).to(device)

    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()
    enable_mc_dropout(model)

    print(f"Running MC-Dropout inference with N={N_SAMPLES} on {device}...")

    for i, (img, mask, meta) in enumerate(tqdm(test_loader, desc="MC-Dropout", ncols=80)):
        if MAX_SLICES is not None and i >= MAX_SLICES:
            break

        img = img.to(device, non_blocking=True)
        slice_id = f"case_{i:04d}"

        # ---- stochastic predictions ----
        samples = forward_samples(model, img, N_SAMPLES)  # [N, B, 1, H, W]
        p_mean = samples.mean(axis=0)[0, 0]

        # ---- entropy-based uncertainties ----
        ent_pred = entropy(p_mean)                         # predictive entropy
        aleatoric = entropy(samples[:, 0, 0]).mean(axis=0)  # expected entropy
        epistemic = ent_pred - aleatoric                    # mutual information

        # ---- prepare visual data ----
        ct = img[0, 0].cpu().numpy()
        pred_bin = (p_mean >= THRESH).astype(np.uint8)

        # ---- paths ----
        overlay_path = os.path.join(OUT_DIR, f"uncer_{slice_id}.png")
        pred_path = os.path.join(OUT_DIR, f"{slice_id}_pred.npy")
        aleatoric_path = os.path.join(OUT_DIR, f"{slice_id}_aleatoric.npy")
        epistemic_path = os.path.join(OUT_DIR, f"{slice_id}_epistemic.npy")

        # ---- save ----
        np.save(pred_path, p_mean.astype(np.float32))
        np.save(aleatoric_path, aleatoric.astype(np.float32))
        np.save(epistemic_path, epistemic.astype(np.float32))
        overlay_preview(ct, pred_bin, aleatoric, epistemic, overlay_path)

        # ---- write CSV entry ----
        with open(summary_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                slice_id,
                float(np.mean(aleatoric)), float(np.std(aleatoric)),
                float(np.mean(epistemic)), float(np.std(epistemic)),
                pred_path, aleatoric_path, epistemic_path, overlay_path
            ])

    print(f"✅ Done! All outputs saved to: {OUT_DIR}")
    print(f"   ↳ Summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
