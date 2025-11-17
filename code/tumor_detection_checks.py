#!/usr/bin/env python3
"""
Compute tumor detection stats without overwriting main CSV.

Reads:
  results/phase8_eval/phase8_metrics.csv

Outputs:
  results/phase8_eval/tumor_detection_stats.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(".")
METRICS_CSV = ROOT / "results" / "phase8_eval" / "phase8_metrics.csv"
OUT_CSV = ROOT / "results" / "phase8_eval" / "tumor_detection_stats.csv"

if not METRICS_CSV.exists():
    raise SystemExit(f"Metrics file not found: {METRICS_CSV}")

df = pd.read_csv(METRICS_CSV)

total = len(df)
gt_slices = (df['tumor_gt_sum'] > 0).sum()
pred_slices = (df['tumor_pred_sum'] > 0).sum()

missed = ((df['tumor_gt_sum'] > 0) & (df['tumor_pred_sum'] == 0)).sum()
detected_when_gt = ((df['tumor_gt_sum'] > 0) & (df['tumor_pred_sum'] > 0)).sum()

det_rate = 100 * detected_when_gt / max(gt_slices, 1)

print("==== TUMOR DETECTION CHECKS ====")
print("Total slices:", total)
print("Slices with GT tumor:", gt_slices)
print("Slices with any predicted tumor:", pred_slices)
print("GT tumor slices missed by model:", missed)
print("GT tumor slices detected:", detected_when_gt)
print("Detection rate on tumor-containing slices: {:.2f}%".format(det_rate))

# Save to CSV
out_df = pd.DataFrame([{
    'total_slices': total,
    'gt_tumor_slices': gt_slices,
    'predicted_tumor_slices': pred_slices,
    'missed_tumor_slices': missed,
    'detected_gt_tumor_slices': detected_when_gt,
    'detection_rate_percent': det_rate
}])

out_df.to_csv(OUT_CSV, index=False)
print(f"\nSaved tumor detection stats → {OUT_CSV}")
