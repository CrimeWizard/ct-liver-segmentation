# utils/losses_weighted.py
# Phase 6A – Weighted + Hybrid Losses for Multi-Class Segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassDiceLoss(nn.Module):
    """Multi-class soft Dice loss used for small-region sensitivity."""
    def __init__(self, num_classes=3, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, preds, targets):
        # preds: (B,C,H,W), targets: (B,H,W)
        preds = F.softmax(preds, dim=1)
        total = 0.0
        for c in range(self.num_classes):
            p = preds[:, c]
            t = (targets == c).float()
            inter = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice = (2 * inter + self.eps) / (union + self.eps)
            total += 1 - dice
        return total.mean()


class WeightedHybridLoss(nn.Module):
    """
    Weighted CrossEntropy + Multi-class Dice hybrid.
    Balances gradients between large (liver) and small (tumor) regions.
    """
    def __init__(self, class_weights=(0.2, 1.0, 4.0), num_classes=3, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        self.dice = MulticlassDiceLoss(num_classes=num_classes)
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        ce_loss = self.ce(outputs, targets)
        dice_loss = self.dice(outputs, targets)
        return (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
