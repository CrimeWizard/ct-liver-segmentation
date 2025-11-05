🧠 MEVIS Project — Phase 4: Uncertainty & Reliability Calibration

Objective:
Enhance the baseline U-Net with uncertainty estimation and probabilistic calibration, enabling confidence-aware predictions and visual interpretability.

Step 1 — MC-Dropout Uncertainty Estimation

Method:
Monte Carlo Dropout (N = 20 samples per slice).
Dropout layers were re-enabled during inference to simulate posterior sampling.

Outputs generated:

Predictive mean probability map

Aleatoric uncertainty map (data noise)

Epistemic uncertainty map (model uncertainty)

Overlay visualizations (uncer_case_XXXX.png)

Raw .npy arrays for every slice

uncertainty_summary.csv — per-slice mean ± std uncertainty metrics

Runtime: ≈ 15 min (GPU MX450)
Dataset: 1021 test slices
Model: Baseline U-Net (7.24 M params)
Samples per slice: 20

Findings:

Aleatoric uncertainty concentrated along liver/tumor boundaries → data ambiguity.

Epistemic uncertainty highlighted rare or difficult regions → model doubt.

Histogram confirmed majority of slices had low uncertainty; only a small tail showed high risk.

Step 2 — Reliability Calibration

Goal:
Check whether model confidence values correspond to actual correctness.

Method:

Isotonic regression calibration

Reliability diagram + Expected Calibration Error (ECE) + Brier Score

Results:

Metric	Uncalibrated	Calibrated	Δ (Improvement)
ECE	0.0036	0.00005	↓ 99 %
Brier Score	0.00427	0.00376	↓ 12 %

Interpretation:
The baseline U-Net was already well calibrated (ECE ≈ 0.004), slightly over-confident, and improved to nearly perfect calibration after isotonic regression.
The reliability diagram shows the calibrated curve tightly follows the diagonal, confirming confidence ≈ accuracy.

Deliverables:

File	Description
uncertainty_summary.csv	Slice-wise mean ± std uncertainty values
phase4_uncertainty_distribution.png	Uncertainty histograms
phase4_reliability_diagram.png	Calibration diagram
phase4_calibration_metrics.csv	ECE & Brier metrics
uncer_case_XXXX.png / .npy	Full uncertainty visualizations

Conclusion:
Monte Carlo Dropout and isotonic calibration successfully quantified and corrected prediction confidence.
The model’s probabilities are now statistically reliable, enabling robust uncertainty-aware segmentation and future active-learning experiments.

✅ Phase 4 status: Completed successfully.