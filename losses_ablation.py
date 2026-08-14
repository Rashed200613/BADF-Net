"""
=============================================================
Loss Function Ablation Registry (L0-L10)
-------------------------------------------------------------
Every loss variant implements forward(logits, targets) -> scalar,
so the training loop in run_loss_ablation.py never needs to
know which variant it's using.
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        union = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * ((1.0 - pt) ** self.gamma) * ce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        TP = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        FP = torch.sum(probs * (1.0 - targets_one_hot), dim=(0, 2, 3))
        FN = torch.sum((1.0 - probs) * targets_one_hot, dim=(0, 2, 3))
        tversky = (TP + 1.0) / (TP + self.alpha * FP + self.beta * FN + 1.0)
        return 1.0 - tversky.mean()


class CELoss(nn.Module):
    """Wraps CrossEntropyLoss so it matches the forward(logits, targets) signature of the others."""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class CompositeLoss(nn.Module):
    """L = w_dice * Dice + w_focal * Focal + w_tversky * Tversky"""
    def __init__(self, w_dice=0.5, w_focal=0.3, w_tversky=0.2, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.w_dice, self.w_focal, self.w_tversky = w_dice, w_focal, w_tversky
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits, targets):
        return (self.w_dice * self.dice(logits, targets) +
                self.w_focal * self.focal(logits, targets) +
                self.w_tversky * self.tversky(logits, targets))


class WeightedSum(nn.Module):
    """Generic weighted combination of two or more loss modules."""
    def __init__(self, components):
        super().__init__()
        self.losses = nn.ModuleList([c for c, _ in components])
        self.weights = [w for _, w in components]

    def forward(self, logits, targets):
        total = 0.0
        for loss_fn, w in zip(self.losses, self.weights):
            total = total + w * loss_fn(logits, targets)
        return total


# Registry: loss_id -> (description, factory_fn). Call factory_fn() to build a fresh module.
LOSS_CONFIGS = {
    "L0_composite":                  ("0.5 Dice + 0.3 Focal + 0.2 Tversky (current choice)",
                                       lambda: CompositeLoss(0.5, 0.3, 0.2, 0.7, 0.3)),
    "L1_dice_only":                  ("Dice only",
                                       lambda: DiceLoss()),
    "L2_focal_only":                 ("Focal only",
                                       lambda: FocalLoss()),
    "L3_tversky_only":               ("Tversky only (alpha=0.7, beta=0.3)",
                                       lambda: TverskyLoss(0.7, 0.3)),
    "L4_ce_only":                    ("Standard Cross-Entropy only",
                                       lambda: CELoss()),
    "L5_dice_ce":                    ("Dice + CE (0.5 / 0.5)",
                                       lambda: WeightedSum([(DiceLoss(), 0.5), (CELoss(), 0.5)])),
    "L6_dice_focal":                 ("Dice + Focal (0.5 / 0.5), Tversky dropped",
                                       lambda: WeightedSum([(DiceLoss(), 0.5), (FocalLoss(), 0.5)])),
    "L7_dice_tversky":               ("Dice + Tversky (0.5 / 0.5), Focal dropped",
                                       lambda: WeightedSum([(DiceLoss(), 0.5), (TverskyLoss(0.7, 0.3), 0.5)])),
    "L8_equal_composite":            ("Composite with equal weights (0.33/0.33/0.33)",
                                       lambda: CompositeLoss(1/3, 1/3, 1/3, 0.7, 0.3)),
    "L9_composite_tversky_recall":   ("Composite, Tversky alpha=0.3/beta=0.7 (favor recall)",
                                       lambda: CompositeLoss(0.5, 0.3, 0.2, 0.3, 0.7)),
    "L10_composite_tversky_dice_eq": ("Composite, Tversky alpha=0.5/beta=0.5 (Dice-equivalent)",
                                       lambda: CompositeLoss(0.5, 0.3, 0.2, 0.5, 0.5)),
}