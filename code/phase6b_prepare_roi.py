#!/usr/bin/env python3
# Phase 6B — Train binary U-Net on liver ROI for tumor segmentation

import os, csv, time, numpy as np, torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.tumor_roi_dataset import get_loaders
from models.unet_binary_roi import UNetBinaryROI, count_parameters
from utils.losses_phase6b import BCEDiceHybrid

CFG = {
    "base_dir": "../data/processed/roi_tumor",
    "epochs": 15,
    "batch_size": 6,
    "num_workers": 2,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "amp": True,
    "checkpoint_path": "checkpoints/phase6b_tumor_roi_best.pth",
    "log_csv": "logs/phase6b_tumor_roi.csv",
    "seed": 42,
    "early_stop_patience": 5,
}

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dice_score_bin(logits, targets):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    inter = (preds * targets).sum(dim=[1,2,3])
    denom = preds.sum(dim=[1,2,3]) + targets.sum(dim=[1,2,3]) + 1e-6
    dice = (2*inter) / denom
    return dice.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train(); total=0.0; n=0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)  # [B,1,H,W]
        optimizer.zero_grad(set_to_none=True)
        if CFG["amp"]:
            with autocast(device_type="cuda"):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            logits = model(imgs); loss = criterion(logits, masks); loss.backward(); optimizer.step()
        total += loss.item() * imgs.size(0); n += imgs.size(0)
    return total / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); dices=[]
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)
        logits = model(imgs)
        dices.append(dice_score_bin(logits, masks))
    return float(np.mean(dices)) if dices else 0.0

def main():
    set_seed(CFG["seed"])
    os.makedirs(os.path.dirname(CFG["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CFG["log_csv"]), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, _ = get_loaders(CFG["base_dir"], CFG["batch_size"], CFG["num_workers"])

    model = UNetBinaryROI(in_channels=1, base_ch=32, dropout=0.1, use_transpose=False).to(device)
    print(f"Model params: {count_parameters(model)/1e6:.2f}M on {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    criterion = BCEDiceHybrid(bce_weight=0.5)
    scaler = GradScaler("cuda", enabled=CFG["amp"])

    # logging header
    if not os.path.exists(CFG["log_csv"]):
        with open(CFG["log_csv"], "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_dice","lr"])

    best = -1.0; noimp=0
    for epoch in range(1, CFG["epochs"]+1):
        t0=time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_dice = evaluate(model, val_loader, device)
        scheduler.step(val_dice)

        with open(CFG["log_csv"], "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{val_dice:.6f}", f"{optimizer.param_groups[0]['lr']:.2e}"])

        if val_dice > best:
            best = val_dice; noimp=0
            torch.save(model.state_dict(), CFG["checkpoint_path"])
        else:
            noimp += 1

        dt=time.time()-t0
        print(f"Epoch {epoch:02d}/{CFG['epochs']}  loss={tr_loss:.4f}  val_dice={val_dice:.4f}  best={best:.4f}  time={dt:.1f}s")

        if noimp >= CFG["early_stop_patience"]:
            print("Early stopping (no improvement)."); break

    print("✅ Phase 6B tumor-ROI training done. Best val Dice:", f"{best:.4f}")

if __name__ == "__main__":
    main()
