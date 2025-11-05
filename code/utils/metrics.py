import torch

# ------------------------------------------------------------
# Dice coefficient
# ------------------------------------------------------------
def dice_coeff(pred, target, threshold=0.5, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.mean().item()

# ------------------------------------------------------------
# IoU (Jaccard Index)
# ------------------------------------------------------------
def iou_score(pred, target, threshold=0.5, epsilon=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=(1,2,3))
    union = (pred + target).clamp(0,1).sum(dim=(1,2,3))
    iou = (intersection + epsilon) / (union + epsilon)
    return iou.mean().item()

# ------------------------------------------------------------
# Pixel accuracy
# ------------------------------------------------------------
def pixel_accuracy(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()
