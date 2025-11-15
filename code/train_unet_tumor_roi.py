#!/usr/bin/env python3
# Phase 6B — Train binary U-Net on liver ROI for tumor segmentation
# Final version: WeightedSampler, Stable Focal+Dice, AMP, Resumable.

import os, csv, time, numpy as np, torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.tumor_roi_dataset import get_loaders
from models.unet_binary_roi import UNetBinaryROI, count_parameters
# Import our new STABLE loss classes
from utils.losses_phase6b import FocalBCEDiceStable, BCEPlusDice

# -----------------------------
# Training configuration (Final)
# -----------------------------
CFG = {
    "base_dir": "../data/processed/roi_tumor",
    "epochs": 60,             # <-- CHANGED: Total epochs to run (was 20)
    "batch_size": 2,          # Kept low for 7.24M param model on 2GB VRAM
    "num_workers": 2,
    "lr": 2e-4,               # Stable LR
    "weight_decay": 0.0,
    "amp": True,              # Enabled for speed and cooler temps
    "checkpoint_path": "checkpoints/phase6b_tumor_roi_best.pth",
    "log_csv": "logs/phase6b_tumor_roi_FOCAL_STABLE.csv", # Will append to this log
    "seed": 42,
    "early_stop_patience": 5, 
    "base_ch": 32,            # 7.24M parameter model
}

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dice_score_bin(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float().unsqueeze(1) # Add channel dim for comparison
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3]) + 1e-6
    dice = (2*inter) / denom
    return dice.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train(); total=0.0; n=0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        # Masks come from loader as [B, H, W], loss functions handle unsqueezing
        masks = masks.to(device, non_blocking=True) 
        optimizer.zero_grad(set_to_none=True)
        
        if CFG["amp"]:
            with autocast(device_type="cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ NaN/Inf loss encountered in AMP. Skipping step.")
                continue
                
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else: # Fallback for CPU or if AMP is off
            logits = model(imgs)
            loss = criterion(logits, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ NaN/Inf loss encountered. Skipping step.")
                continue
            loss.backward(); optimizer.step()
            
        total += float(loss.detach().cpu().item()) * imgs.size(0); n += imgs.size(0)
    
    return (total / n) if n > 0 else float('nan')

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); dices=[]
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True) # [B, H, W]
        logits = model(imgs)
        dices.append(dice_score_bin(logits, masks))
    return float(np.mean(dices)) if dices else 0.0

def main():
    set_seed(CFG["seed"])
    os.makedirs(os.path.dirname(CFG["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CFG["log_csv"]), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # --- Loaders with Weighted Sampler ---
    print("Starting training with WeightedRandomSampler (oversampling tumor slices)...")
    train_loader, val_loader, test_loader = get_loaders(
        CFG["base_dir"], 
        CFG["batch_size"], 
        CFG["num_workers"], 
        only_tumor=False,    
        use_sampler=True     
    )
    # -------------------------------------

    model = UNetBinaryROI(in_channels=1, base_ch=CFG["base_ch"], dropout=0.1, use_transpose=False).to(device)
    print(f"Model params: {count_parameters(model)/1e6:.2f}M on {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = FocalBCEDiceStable(alpha=0.25, gamma=2.0, dice_weight=0.8)
    scaler = GradScaler(enabled=CFG["amp"]) 

    # --- NEW: Resume from checkpoint logic ---
    start_epoch = 1
    best_dice = -1.0
    
    if os.path.exists(CFG["checkpoint_path"]):
        print(f"Resuming training from checkpoint: {CFG['checkpoint_path']}")
        model.load_state_dict(torch.load(CFG["checkpoint_path"]))
        
        # Try to read the log file to find the last epoch and best dice
        try:
            with open(CFG["log_csv"], 'r') as f:
                reader = csv.DictReader(f)
                log_data = list(reader)
                if len(log_data) > 0:
                    last_epoch_data = log_data[-1]
                    start_epoch = int(last_epoch_data['epoch']) + 1
                    best_dice = float(last_epoch_data['val_dice'])
                    # Find the actual best dice from the log
                    for row in log_data:
                        if float(row['val_dice']) > best_dice:
                            best_dice = float(row['val_dice'])
                    
                    print(f"Resuming from Epoch {start_epoch}, previous best Dice: {best_dice:.4f}")
        except Exception as e:
            print(f"Could not read log file ({e}), starting from epoch 1 with loaded weights.")
            start_epoch = 1
            best_dice = -1.0 # Will be updated on first eval
    else:
        # If no checkpoint, create a clean log file
        print("Starting a new training run.")
        with open(CFG["log_csv"], "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_dice","lr"])
    # ----------------------------------------
            
    epochs_no_improve = 0
    # --- UPDATED: Training loop now starts from start_epoch ---
    for epoch in range(start_epoch, CFG["epochs"] + 1):
        t0=time.time()
        
        tr_loss = train_one_epoch(model, loader=train_loader, criterion=criterion, optimizer=optimizer, scaler=scaler, device=device)
        val_dice = evaluate(model, loader=val_loader, device=device)
        scheduler.step(val_dice)

        with open(CFG["log_csv"], "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{val_dice:.6f}", f"{optimizer.param_groups[0]['lr']:.2e}"])

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            torch.save(model.state_dict(), CFG["checkpoint_path"])
        else:
            epochs_no_improve += 1

        dt=time.time()-t0
        print(f"Epoch {epoch:02d}/{CFG['epochs']}  loss={tr_loss:.4f}  val_dice={val_dice:.4f}  best={best_dice:.4f}  time={dt:.1f}s")

        if epochs_no_improve >= CFG["early_stop_patience"]:
            print(f"Early stopping triggered after {epoch} epochs (no improvement)."); break

    print(f"✅ Phase 6B tumor-ROI training done. Best val Dice: {best_dice:.4f}")

if __name__ == "__main__":
    main()