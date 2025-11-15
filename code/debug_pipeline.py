import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Import all our components
from datasets.tumor_roi_dataset import get_loaders
from models.unet_binary_roi import UNetBinaryROI, count_parameters
# This time, we import ALL the losses defined in your file
from utils.losses_phase6b import DiceLossBinary, FocalBCEDice 
from utils.losses_phase6b import BCEPlusDice # This is the class you just added

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "../data/processed/roi_tumor"
BATCH_SIZE = 2
BASE_CH = 32
print(f"--- STARTING DEBUG PIPELINE ON {device} ---")
print("This will test Data, Model, and Loss functions in isolation.")

# ---
# 1. DATA VERIFICATION
# ---
print("\n--- 1. Verifying Data (using NEW 20:1 WeightedRandomSampler) ---")
try:
    train_loader, _, _ = get_loaders(
        base_dir=BASE_DIR, 
        batch_size=BATCH_SIZE, 
        num_workers=2, 
        only_tumor=False, 
        use_sampler=True
    )
    
    # Get ONE batch from the loader
    images, masks, ids = next(iter(train_loader))
    images, masks = images.to(device), masks.to(device)

    print(f"Batch loaded successfully.")
    print(f"Image batch shape: {images.shape}, dtype: {images.dtype}")
    print(f"Mask batch shape:  {masks.shape}, dtype: {masks.dtype}")
    print(f"Image stats (min/max/mean): {images.min():.2f} / {images.max():.2f} / {images.mean():.2f}")
    print(f"Mask stats (min/max/mean):  {masks.min():.2f} / {masks.max():.2f} / {masks.mean():.2f}")

    if masks.mean() == 0.0:
        print("ERROR: Sampler *still* gave a bad batch. This is unlucky. Try running debug again.")
    else:
        print("SUCCESS: Sampler provided a batch with tumor pixels!")

    # --- Visual Check ---
    print("\nSaving visual debug plot to debug_batch_overlay.png...")
    fig, axs = plt.subplots(BATCH_SIZE, 2, figsize=(10, 5 * BATCH_SIZE))
    for i in range(BATCH_SIZE):
        img_np = images[i, 0].cpu().numpy()
        msk_np = masks[i].cpu().numpy()
        
        axs[i, 0].imshow(img_np, cmap='gray')
        axs[i, 0].set_title(f"Image: {ids[i]}")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(img_np, cmap='gray')
        axs[i, 1].imshow(msk_np, cmap='Reds', alpha=0.5)
        axs[i, 1].set_title(f"Mask (Mean: {msk_np.mean():.4f})")
        axs[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig("debug_batch_overlay.png")
    print(f"Visual check saved. Please open debug_batch_overlay.png")

except Exception as e:
    print(f"\n--- !!! DATA VERIFICATION FAILED !!! ---")
    print(f"Error: {e}")
    exit()


# ---
# 2. MODEL VERIFICATION
# ---
print("\n--- 2. Verifying Model ---")
try:
    model = UNetBinaryROI(in_channels=1, base_ch=BASE_CH).to(device)
    model.eval() 
    with torch.no_grad():
        logits = model(images) 
    
    print(f"Model forward pass successful.")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits stats (min/max/mean): {logits.min():.2f} / {logits.max():.2f} / {logits.mean():.2f}")
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("ERROR: Model produced NaN or Inf logits.")
    else:
        print("SUCCESS: Model produced stable, finite logits.")

except Exception as e:
    print(f"\n--- !!! MODEL VERIFICATION FAILED !!! ---")
    print(f"Error: {e}")
    exit()


# ---
# 3. LOSS FUNCTION VERIFICATION
# ---
print("\n--- 3. Verifying Loss Functions ---")
try:
    masks_unsqueezed = masks.unsqueeze(1) 
    perfect_logits = (masks_unsqueezed.float() - 0.5) * 20 
    all_black_logits = torch.zeros_like(logits)

    print("\nTesting (Simpler) BCEPlusDice:")
    criterion_simple = BCEPlusDice(bce_w=0.5)
    loss_simple_real = criterion_simple(logits, masks_unsqueezed)
    loss_simple_perfect = criterion_simple(perfect_logits, masks_unsqueezed)
    loss_simple_all_black = criterion_simple(all_black_logits, masks_unsqueezed)

    print(f"Loss (real data): {loss_simple_real.item():.4f}")
    print(f"Loss (perfect prediction): {loss_simple_perfect.item():.4f}")
    print(f"Loss (all-black prediction): {loss_simple_all_black.item():.4f}")
    
    if loss_simple_perfect < loss_simple_real and loss_simple_real < loss_simple_all_black:
        print("SUCCESS: Simple loss is logical.")
    else:
        print("ERROR: Simple loss is NOT logical.")

    print("\nTesting (Focal) FocalBCEDice:")
    # This is the line I fixed. Your file has 'dice_weight', not 'dice_w'.
    criterion_focal = FocalBCEDice(alpha=0.25, gamma=2.0, dice_weight=0.8) 
    loss_focal_real = criterion_focal(logits, masks_unsqueezed)
    loss_focal_perfect = criterion_focal(perfect_logits, masks_unsqueezed)
    loss_focal_all_black = criterion_focal(all_black_logits, masks_unsqueezed)

    print(f"Loss (real data): {loss_focal_real.item():.4f}")
    print(f"Loss (perfect prediction): {loss_focal_perfect.item():.4f}")
    print(f"Loss (all-black prediction): {loss_focal_all_black.item():.4f}")

    if loss_focal_perfect < loss_focal_real and loss_focal_real < loss_focal_all_black:
        print("SUCCESS: Focal loss is logical.")
    else:
        print("ERROR: Focal loss is NOT logical.")

except Exception as e:
    print(f"\n--- !!! LOSS VERIFICATION FAILED !!! ---")
    print(f"Error: {e}")

print("\n--- DEBUG PIPELINE COMPLETE ---")