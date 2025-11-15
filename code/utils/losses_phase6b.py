import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Dice loss for binary masks (Corrected Version)
# -----------------------------
class DiceLossBinary(nn.Module):
    """Numerically stable Dice Loss."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Targets from loader are [B, H, W], unsqueeze to [B, 1, H, W]
        targets = targets.unsqueeze(1).float()
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        num = 2 * (probs * targets).sum(dim=1) + self.smooth
        den = probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        # Take mean of batch dice scores
        dice = 1 - (num / den).mean() 
        return dice

# -----------------------------
# NEW: STABLE Focal + Dice Hybrid
# -----------------------------
class FocalBCEDiceStable(nn.Module):
    """
    Numerically stable Focal Loss + Dice Loss, 100% AMP-compatible.
    Computes Focal Loss based on the stable BCEWithLogitsLoss.
    """
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.dice_loss = DiceLossBinary()
        # Use reduction='none' to apply focal modulation per-pixel
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        # Targets are [B, H, W], unsqueeze to [B, 1, H, W] and set to float
        targets_unsqueezed = targets.unsqueeze(1).float()
        
        # --- Stable Focal Loss Calculation ---
        # 1. Calculate BCE loss (per-pixel) using stable log-sigmoid
        bce_loss = self.bce_loss(logits, targets_unsqueezed)
        
        # 2. Get probs (pt)
        with torch.no_grad(): # Don't need gradients for this part
            probs = torch.sigmoid(logits)
            # Calculate pt (probability of the *correct* class for each pixel)
            pt = probs * targets_unsqueezed + (1 - probs) * (1 - targets_unsqueezed)
        
        # 3. Calculate modulating factor
        modulating_factor = (1.0 - pt) ** self.gamma
        
        # 4. Calculate alpha weight
        alpha_weight = targets_unsqueezed * self.alpha + (1 - targets_unsqueezed) * (1 - self.alpha)
        
        # 5. Apply modulation and alpha to the per-pixel BCE loss
        focal_bce_loss = (alpha_weight * modulating_factor * bce_loss).mean()
        
        # --- Dice Loss Calculation ---
        # We pass targets as [B, H, W] since DiceLossBinary handles the unsqueeze
        dice = self.dice_loss(logits, targets) 
        
        # --- Combine ---
        loss = (1 - self.dice_weight) * focal_bce_loss + self.dice_weight * dice
        
        return loss

# -----------------------------
# Simple BCE + Dice hybrid (Corrected Version)
# -----------------------------
class BCEPlusDice(nn.Module):
    """Numerically stable simple BCE + Dice Loss."""
    def __init__(self, bce_w=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLossBinary()
        self.bce_w = bce_w
    
    def forward(self, logits, targets):
        # Targets are [B, H, W], unsqueeze to [B, 1, H, W] and set to float
        targets_unsqueezed = targets.unsqueeze(1).float() 
        b = self.bce(logits, targets_unsqueezed)
        # Pass the original [B, H, W] target to DiceLossBinary
        d = self.dice(logits, targets) 
        loss = self.bce_w * b + (1 - self.bce_w) * d
        return loss