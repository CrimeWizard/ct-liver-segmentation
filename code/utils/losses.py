import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Dice Loss
# ------------------------------------------------------------
def dice_loss(pred, target, epsilon=1e-6):
    """
    Computes soft Dice loss (1 - Dice coefficient).
    pred: raw logits or probabilities [B,1,H,W]
    target: binary mask [B,1,H,W]
    """
    # Apply sigmoid if logits
    if pred.shape != target.shape:
        raise ValueError("Pred and target shapes must match")
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    loss = 1 - dice
    return loss.mean()

# ------------------------------------------------------------
# Combined BCE + Dice Loss
# ------------------------------------------------------------
class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy (with logits) + Dice loss.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dsc = dice_loss(pred, target)
        return self.bce_w * bce + self.dice_w * dsc
