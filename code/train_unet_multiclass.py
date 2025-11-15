# Phase 5 — Multi-Class U-Net Training (background, liver, tumor)

import os, csv, time
import numpy as np
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from datasets.multiclass_dataset import get_loaders   # you'll create/point this to masks_multiclass
from models.unet_multiclass import UNetMultiClass, count_parameters


# ---------------------------------------------------
# Config
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
    "checkpoint_path": "checkpoints/unet_multiclass_best.pth",
    "log_csv": "logs/phase5_multiclass_log.csv",
    "loss_curve_png": "docs/figures/phase5_loss_curve.png",
    "dice_curve_png": "docs/figures/phase5_dice_curve.png",
    "seed": 42,
    "early_stop_patience": 5,
}


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_per_class(preds, targets, cls):
    """Compute Dice for a specific class index."""
    preds_c = (preds == cls)
    targets_c = (targets == cls)
    inter = (preds_c & targets_c).sum().float()
    denom = preds_c.sum() + targets_c.sum()
    return (2 * inter / denom).item() if denom > 0 else 1.0


# ---------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp=True):
    model.train()
    running_loss = 0.0
    for imgs, masks, _ in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()   # (B,H,W)
        optimizer.zero_grad(set_to_none=True)

        if amp:
            with autocast(device_type="cuda"):
                outputs = model(imgs)  # (B,3,H,W)
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
    dice_liver, dice_tumor = [], []
    for imgs, masks, _ in tqdm(loader, desc="Valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()

        outputs = model(imgs)
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        dice_liver.append(dice_per_class(preds, masks, 1))
        dice_tumor.append(dice_per_class(preds, masks, 2))

    return float(np.mean(dice_liver)), float(np.mean(dice_tumor))


# ---------------------------------------------------
# Main training
# ---------------------------------------------------
def main():
    set_seed(CFG["seed"])
    os.makedirs(os.path.dirname(CFG["checkpoint_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(CFG["log_csv"]), exist_ok=True)
    os.makedirs("docs/figures", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # -------------------------------
    # Data
    # -------------------------------
    train_loader, val_loader, _ = get_loaders(
        CFG["base_dir"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        augment_train=CFG["augment_train"],
        multiclass=True,  # your dataset can use this flag
    )

    # -------------------------------
    # Model, optimizer, loss
    # -------------------------------
    model = UNetMultiClass(
        in_channels=1, n_classes=3, base_ch=32, dropout=0.5, use_transpose=False
    ).to(device)
    print(f"Model params: {count_parameters(model)/1e6:.2f}M on {device}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda", enabled=CFG["amp"])

    # -------------------------------
    # Resume
    # -------------------------------
    best_mean_dice = -1.0
    if os.path.exists(CFG["checkpoint_path"]):
        print(f"Resuming from {CFG['checkpoint_path']}")
        model.load_state_dict(torch.load(CFG["checkpoint_path"]))
        best_mean_dice = 0.0

    # -------------------------------
    # Logging
    # -------------------------------
    if not os.path.exists(CFG["log_csv"]):
        with open(CFG["log_csv"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "dice_liver", "dice_tumor", "mean_dice", "lr"])

    train_losses, val_mean_dices = [], []
    epochs_no_improve = 0

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, amp=CFG["amp"])
        dice_liver, dice_tumor = evaluate(model, val_loader, device)
        mean_dice = (dice_liver + dice_tumor) / 2.0

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(1.0 - mean_dice)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < lr_now:
            print(f"Learning rate reduced to {new_lr:.2e}")

        # log
        train_losses.append(train_loss)
        val_mean_dices.append(mean_dice)
        with open(CFG["log_csv"], "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{dice_liver:.6f}",
                 f"{dice_tumor:.6f}", f"{mean_dice:.6f}", f"{new_lr:.6e}"]
            )

        # checkpoint
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(model.state_dict(), CFG["checkpoint_path"])
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{CFG['epochs']}  "
            f"train_loss={train_loss:.4f}  "
            f"dice_liver={dice_liver:.4f}  dice_tumor={dice_tumor:.4f}  "
            f"mean_dice={mean_dice:.4f}  best={best_mean_dice:.4f}  "
            f"time={dt:.1f}s"
        )

        if epochs_no_improve >= CFG["early_stop_patience"]:
            print(f"Early stopping after {epoch} epochs (no improvement).")
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
        plt.plot(val_mean_dices, label="mean Dice")
        plt.xlabel("epoch"); plt.ylabel("Dice")
        plt.legend(); plt.tight_layout()
        plt.savefig(CFG["dice_curve_png"], dpi=150)
        plt.close()
        print("Saved curves to:", CFG["loss_curve_png"], "and", CFG["dice_curve_png"])
    except Exception as e:
        print("Plotting skipped:", e)


if __name__ == "__main__":
    main()
