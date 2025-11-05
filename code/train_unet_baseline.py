# code/train_unet_baseline.py
# Phase 3 — Baseline 2D U-Net Training (GPU + AMP + checkpoints + curves)
# watch -n 1 nvidia-smi    (GPU tracking command)

import os, csv, time
import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler     # ✅ new AMP syntax
from tqdm import tqdm

from datasets.liver_dataset import get_loaders
from models.unet import UNet, count_parameters
from utils.losses import BCEDiceLoss
from utils.metrics import dice_coeff, iou_score, pixel_accuracy

# ---------------------------------------------------
# Config (tuned for NVIDIA MX450, 2 GB VRAM)
# ---------------------------------------------------
CFG = {
    "base_dir": "../data/processed",
    "epochs": 30,
    "batch_size": 4,
    "num_workers": 2,
    "lr": 1e-4,
    "weight_decay": 0.0,
    "augment_train": True,
    "amp": True,
    "checkpoint_path": "checkpoints/unet_baseline_best.pth",
    "log_csv": "logs/phase3_training_log.csv",
    "loss_curve_png": "docs/figures/phase3_loss_curve.png",
    "dice_curve_png": "docs/figures/phase3_dice_curve.png",
    "seed": 42,
    "early_stop_patience": 5,   # stop if no Dice improvement for 5 epochs
}


# ---------------------------------------------------
# Utility functions
# ---------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True):
    model.train()
    running_loss = 0.0

    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    dices, ious, accs = [], [], []

    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        outputs = model(imgs)
        dices.append(dice_coeff(outputs, masks))
        ious.append(iou_score(outputs, masks))
        accs.append(pixel_accuracy(outputs, masks))

    return float(np.mean(dices)), float(np.mean(ious)), float(np.mean(accs))


# ---------------------------------------------------
# Main training pipeline
# ---------------------------------------------------
def main():
    set_seed(CFG["seed"])
    os.makedirs(os.path.dirname(CFG["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CFG["log_csv"]), exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # speed up on constant input size

    # -------------------------------
    # Data
    # -------------------------------
    train_loader, val_loader, _ = get_loaders(
        CFG["base_dir"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        augment_train=CFG["augment_train"],
    )

    # -------------------------------
    # Model, optimizer, scheduler
    # -------------------------------
    model = UNet(
        in_channels=1, out_channels=1, base_ch=32, dropout=0.5, use_transpose=False
    ).to(device)
    print(f"Model params: {count_parameters(model)/1e6:.2f}M on {device}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    scaler = GradScaler("cuda", enabled=CFG["amp"])

    # -------------------------------
    # Resume if checkpoint exists
    # -------------------------------
    best_dice = -1.0
    if os.path.exists(CFG["checkpoint_path"]):
        print(f"Resuming from {CFG['checkpoint_path']}")
        model.load_state_dict(torch.load(CFG["checkpoint_path"]))
        best_dice = 0.0  # will be updated after next eval

    # -------------------------------
    # Logging setup
    # -------------------------------
    if not os.path.exists(CFG["log_csv"]):
        with open(CFG["log_csv"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_dice", "val_iou", "val_acc", "lr"])

    train_losses, val_dices = [], []
    epochs_no_improve = 0

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp=CFG["amp"]
        )
        val_dice, val_iou, val_acc = evaluate(model, val_loader, device)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(1.0 - val_dice)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < lr_now:
            print(f"Learning rate reduced to {new_lr:.2e}")

        # Log
        train_losses.append(train_loss)
        val_dices.append(val_dice)
        with open(CFG["log_csv"], "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_dice:.6f}",
                 f"{val_iou:.6f}", f"{val_acc:.6f}", f"{new_lr:.6e}"]
            )

        # Checkpoint
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CFG["checkpoint_path"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  val_dice={val_dice:.4f}  "
            f"val_iou={val_iou:.4f}  val_acc={val_acc:.4f}  "
            f"best_dice={best_dice:.4f}  time={dt:.1f}s"
        )

        # Early stopping
        if epochs_no_improve >= CFG["early_stop_patience"]:
            print(f"Early stopping triggered after {epoch} epochs (no improvement).")
            break

    # -------------------------------
    # Plot curves
    # -------------------------------
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_losses, label="train loss")
        plt.xlabel("epoch"); plt.ylabel("loss")
        plt.legend(); plt.tight_layout()
        plt.savefig(CFG["loss_curve_png"], dpi=150)
        plt.close()

        plt.figure()
        plt.plot(val_dices, label="val Dice")
        plt.xlabel("epoch"); plt.ylabel("Dice")
        plt.legend(); plt.tight_layout()
        plt.savefig(CFG["dice_curve_png"], dpi=150)
        plt.close()
        print("Saved curves to:", CFG["loss_curve_png"], "and", CFG["dice_curve_png"])
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
