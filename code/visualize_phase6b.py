import os, json, numpy as np, torch
import matplotlib.pyplot as plt
from models.unet_binary_roi import UNetBinaryROI

# -----------------------------
# 1) Load val IDs
# -----------------------------
base = "../data/processed/roi_tumor"
with open(f"{base}/splits/val_ids.json","r") as f:
    val_ids = json.load(f)

# -----------------------------
# 2) Filter ONLY tumor-positive slices
# -----------------------------
tumor_sids = []
for sid in val_ids:
    mask = np.load(f"{base}/masks/{sid}")
    if mask.sum() > 0:
        tumor_sids.append(sid)

print(f"Tumor-positive val slices: {len(tumor_sids)}")

if len(tumor_sids) == 0:
    raise RuntimeError("No tumor-positive slices in val set!")

# Pick a random tumor slice
import random
sid = random.choice(tumor_sids)
print("Visualizing:", sid)

# -----------------------------
# 3) Load image + mask
# -----------------------------
img = np.load(f"{base}/images/{sid}")
gt = np.load(f"{base}/masks/{sid}")

# -----------------------------
# 4) Load model
# -----------------------------
model = UNetBinaryROI(in_channels=1, base_ch=32)
model.load_state_dict(torch.load("checkpoints/phase6b_tumor_roi_best.pth", map_location="cpu"))
model.eval()

x = torch.from_numpy(img[None,None]).float()
with torch.no_grad():
    pred = torch.sigmoid(model(x)).squeeze().numpy()
    pred_bin = (pred > 0.5).astype(np.uint8)

# -----------------------------
# 5) Visualize
# -----------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("ROI Input")
plt.subplot(1,3,2); plt.imshow(gt, cmap="gray"); plt.title("Ground Truth")
plt.subplot(1,3,3); plt.imshow(pred_bin, cmap="gray"); plt.title("Prediction")
plt.show()
