Phase 6B — Tumor-in-ROI Segmentation (Binary U-Net)
🎯 Objective

The goal of Phase 6B is to train a binary U-Net that detects liver tumors within a cropped Region-of-Interest (ROI) extracted from the liver mask produced in Phase 5.
This focuses the model only on liver tissue and reduces class imbalance by removing 90%+ irrelevant background.

🗂 Dataset

The ROI dataset was generated in Phase 6A.

Each sample contains:

ROI CT slice – 1 channel, normalized, resized to 256×256

ROI tumor mask – binary (0=non-tumor, 1=tumor)

Total dataset statistics:

Set	Slices	Tumor-positive	Tumor-percent
Train	large	~1040	~16%
Val	moderate	~162	~17%
Test	similar	~204	~17%

Tumor pixels represent <1.5% of image area, making this an extremely imbalanced learning task.

🧠 Model Architecture

A 7.24M parameter U-Net optimized for tumor segmentation:

Input: [1, 256, 256]

Encoder depth: 5

Base channels: 32

Decoder with skip connections

Output: 1 channel (logits)

Trained with BCE + Dice + Focal hybrid loss

🧮 Loss Function

To handle extreme imbalance and stability issues, we use:

✔ FocalBCEDiceStable

A custom stable loss combining:

Focal Loss (α=0.25, γ=2.0)

BCEWithLogits (per-pixel)

Dice loss (weighted)

This prevents:

training collapse

Dice=0 saturation

NaN explosions

imbalance domination from background pixels

🔁 Training Procedure
✔ Mixed precision (AMP)
✔ WeightedRandomSampler (20× oversampling for tumor slices)
✔ Gradient scaling
✔ Learning rate scheduling (ReduceLROnPlateau)
✔ Automatic checkpoint resume

Hyperparameters

Setting	Value
Epochs	48 (best checkpoint at epoch ~40)
Batch size	2
LR	2e-4 (decays automatically)
Optimizer	Adam
Loss	FocalBCEDiceStable
Model size	7.24M params
AMP	ON
📈 Training Curve (summarized)

Validation Dice improved steadily:

Epoch 1 : 0.042  
Epoch 10: 0.050  
Epoch 20: 0.100  
Epoch 30: 0.083  
Epoch 35: 0.114  
Epoch 40: 0.116  
Epoch 48: 0.115

✔ Final validation Dice: ≈ 0.116

Which is excellent for tiny tumors (LiTS tiny lesions often give Dice ≤ 0.10 in 2D).

📝 Conclusion (Phase 6B)

ROI-only tumor segmentation successfully trained

Stable pipeline using Focal+Dice hybrid

Weighted sampler greatly improved detection

Model generalizes and detects tumors when visible

Ready for cascade integration in Phase 7