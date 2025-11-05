# code/reliability_calibration.py
# Phase 4 — Step 2: Reliability Calibration for U-Net Predictions
# ---------------------------------------------------------------
# Uses predicted probability maps (.npy) and ground-truth masks
# to compute reliability diagrams, Expected Calibration Error (ECE),
# and other summary metrics.

import os, numpy as np, torch, matplotlib.pyplot as plt, csv
from tqdm import tqdm
from datasets.liver_dataset import get_loaders
from sklearn.isotonic import IsotonicRegression

# ---------------- CONFIG ----------------
PRED_DIR = "docs/figures/phase4_uncertainty"
OUT_FIG = "docs/figures/phase4_reliability_diagram.png"
OUT_CSV = "docs/figures/phase4_calibration_metrics.csv"
N_BINS = 15
# ----------------------------------------

def compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers, bin_uppers = bins[:-1], bins[1:]
    ece, accs, confs = 0.0, [], []
    for bl, bu in zip(bin_lowers, bin_uppers):
        mask = (probs > bl) & (probs <= bu)
        if np.any(mask):
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()
            accs.append(acc); confs.append(conf)
    return ece, np.array(accs), np.array(confs), (bin_lowers + bin_uppers) / 2

def main():
    # load data
    _, _, test_loader = get_loaders("../data/processed", batch_size=1, num_workers=2, augment_train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probs_all, labels_all = [], []

    for i, (img, mask, meta) in enumerate(tqdm(test_loader, desc="Collecting predictions", ncols=80)):
        pred_path = os.path.join(PRED_DIR, f"case_{i:04d}_pred.npy")
        if not os.path.exists(pred_path):
            continue
        pred = np.load(pred_path)
        gt = mask[0, 0].numpy()

        probs_all.append(pred.flatten())
        labels_all.append(gt.flatten())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)

    # --- Uncalibrated ---
    ece_raw, accs, confs, centers = compute_ece(probs_all, labels_all, N_BINS)
    brier_raw = np.mean((probs_all - labels_all) ** 2)

    # --- Temperature / Isotonic calibration ---
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs_all, labels_all)
    probs_cal = ir.predict(probs_all)
    ece_cal, accs_cal, confs_cal, centers_cal = compute_ece(probs_cal, labels_all, N_BINS)
    brier_cal = np.mean((probs_cal - labels_all) ** 2)

    # --- Save metrics ---
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Uncalibrated", "Calibrated"])
        w.writerow(["ECE", ece_raw, ece_cal])
        w.writerow(["Brier", brier_raw, brier_cal])

    # --- Plot reliability diagram ---
    plt.figure(figsize=(6,6))
    plt.plot(centers, accs, marker="o", label=f"Uncalibrated (ECE = {ece_raw:.3f})")
    plt.plot(centers_cal, accs_cal, marker="s", label=f"Calibrated (ECE = {ece_cal:.3f})")
    plt.plot([0,1],[0,1],"k--",label="Perfect Calibration")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    plt.close()

    print("✅ Calibration complete.")
    print(f"   ↳ Saved metrics to: {OUT_CSV}")
    print(f"   ↳ Saved figure to: {OUT_FIG}")

if __name__ == "__main__":
    main()
