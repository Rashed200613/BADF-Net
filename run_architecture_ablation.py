"""
=============================================================
Architectural Ablation Runner
-------------------------------------------------------------
Runs ALL architectural ablation configs (A0-A11, registered in
network/BADF_Net_ablation.py) in a single execution:

    for each config:
        1. build BADF_Net_Ablation(cfg)
        2. train for --epochs (composite loss held FIXED --
           only the architecture changes across configs)
        3. evaluate best checkpoint on the test set
        4. append one row to architecture_ablation_results.csv
           (Test Dice overall + per-class + params + time)

Nothing is overwritten between configs -- every row accumulates
in the same CSV, and every config's best checkpoint is saved
under its own filename so you can inspect any variant later.
=============================================================
"""

import os
import csv
import time
import argparse
import warnings

import numpy as np
import torch
import torch.optim as optim

from network.BADF_Net_ablation import BADF_Net_Ablation, ABLATION_CONFIGS
from dataloader import get_kidney_loader

warnings.filterwarnings("ignore")


# =========================================================================
# Composite loss (fixed across all architecture ablations)
# =========================================================================

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
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.alpha * ((1.0 - pt) ** self.gamma) * ce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.alpha, self.beta = alpha, beta

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        TP = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        FP = torch.sum(probs * (1.0 - targets_one_hot), dim=(0, 2, 3))
        FN = torch.sum((1.0 - probs) * targets_one_hot, dim=(0, 2, 3))
        tversky = (TP + 1.0) / (TP + self.alpha * FP + self.beta * FN + 1.0)
        return 1.0 - tversky.mean()


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice, self.focal, self.tversky = DiceLoss(), FocalLoss(), TverskyLoss()

    def forward(self, logits, targets):
        return 0.5 * self.dice(logits, targets) + 0.3 * self.focal(logits, targets) + 0.2 * self.tversky(logits, targets)


# =========================================================================
# Args
# =========================================================================

def get_args():
    p = argparse.ArgumentParser(description="Architectural Ablation Runner")
    p.add_argument('--train_images_dir', type=str, default='Kidney_dataset/train/images')
    p.add_argument('--train_masks_dir', type=str, default='Kidney_dataset/train/mask')
    p.add_argument('--val_images_dir', type=str, default='Kidney_dataset/val/images')
    p.add_argument('--val_masks_dir', type=str, default='Kidney_dataset/val/mask')
    p.add_argument('--test_images_dir', type=str, default='Kidney_dataset/val/images')
    p.add_argument('--test_masks_dir', type=str, default='Kidney_dataset/val/mask')
    p.add_argument('--num_classes', type=int, default=6)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--base_lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=100, help='Epochs PER ablation config -- ablations commonly use fewer epochs than the final model; raise this if you have the compute budget')
    p.add_argument('--output_dir', type=str, default='ablation_results/architecture')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--only', type=str, nargs='*', default=None, help='Optional: run only these ablation IDs (e.g. --only A0_full_model A7_no_SE). Default: run all.')
    return p.parse_args()


def calculate_classwise_dice(pred, target, num_classes, eps=1e-7):
    scores = {}
    for c in range(1, num_classes):
        pred_c = (pred == c).astype(np.float32)
        gt_c = (target == c).astype(np.float32)
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        scores[c] = float((2.0 * intersection + eps) / (union + eps))
    return scores


def run_one_config(ablation_id, description, cfg, args, device, train_loader, val_loader, test_loader):
    print(f"\n{'='*70}\n[ABLATION] {ablation_id}: {description}\n{'='*70}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = BADF_Net_Ablation(num_classes=args.num_classes, cfg=cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6  # millions

    criterion = CompositeLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{ablation_id}_best.pth")

    best_val_dice = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        model.eval()
        val_dice_scores = []
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device).long().squeeze(1)
                outputs = model(images)
                pred = torch.argmax(outputs, dim=1)
                class_scores = calculate_classwise_dice(pred.cpu().numpy(), masks.cpu().numpy(), args.num_classes)
                val_dice_scores.append(np.mean(list(class_scores.values())) if class_scores else 0.0)

        avg_val_dice = float(np.mean(val_dice_scores)) if val_dice_scores else 0.0
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            print(f"  Epoch {epoch + 1}/{args.epochs} | Val Dice: {avg_val_dice:.4f} | Best: {best_val_dice:.4f}")

    train_time = time.time() - start_time

    # ---- Test-set evaluation using the best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    per_class_dice_all = {c: [] for c in range(1, args.num_classes)}
    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            pred_np, gt_np = pred.cpu().numpy(), masks.cpu().numpy()
            for b in range(pred_np.shape[0]):
                scores = calculate_classwise_dice(pred_np[b], gt_np[b], args.num_classes)
                for c, s in scores.items():
                    per_class_dice_all[c].append(s)

    per_class_mean = {c: (float(np.mean(v)) if v else 0.0) for c, v in per_class_dice_all.items()}
    overall_test_dice = float(np.mean(list(per_class_mean.values()))) if per_class_mean else 0.0

    del model
    torch.cuda.empty_cache()

    return {
        "Ablation_ID": ablation_id,
        "Description": description,
        "Test_Dice_Overall": round(overall_test_dice, 4),
        **{f"Test_Dice_Class{c}": round(v, 4) for c, v in per_class_mean.items()},
        "Params_M": round(n_params, 2),
        "Best_Val_Dice": round(best_val_dice, 4),
        "Train_Time_s": round(train_time, 1),
    }


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[DATA] Loading train/val loaders...")
    train_loader, val_loader = get_kidney_loader(
        train_images_dir=args.train_images_dir, train_masks_dir=args.train_masks_dir,
        val_images_dir=args.val_images_dir, val_masks_dir=args.val_masks_dir,
        batchsize=args.batch_size, trainsize=args.img_size, shuffle=True, augmentation=True)

    print("[DATA] Loading test loader...")
    _, test_loader = get_kidney_loader(
        train_images_dir=args.test_images_dir, train_masks_dir=args.test_masks_dir,
        val_images_dir=args.test_images_dir, val_masks_dir=args.test_masks_dir,
        batchsize=args.batch_size, trainsize=args.img_size, shuffle=False, augmentation=False)

    configs_to_run = {k: v for k, v in ABLATION_CONFIGS.items() if args.only is None or k in args.only}
    print(f"[PLAN] Running {len(configs_to_run)} ablation configs: {list(configs_to_run.keys())}")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "architecture_ablation_results.csv")
    fieldnames = None
    rows = []

    for ablation_id, (description, cfg) in configs_to_run.items():
        result = run_one_config(ablation_id, description, cfg, args, device, train_loader, val_loader, test_loader)
        rows.append(result)

        # Write/append incrementally so partial progress is never lost if a run is interrupted.
        if fieldnames is None:
            fieldnames = list(result.keys())
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(result)
        else:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)

        print(f"[SAVE] {ablation_id} -> Test Dice: {result['Test_Dice_Overall']:.4f} (appended to {csv_path})")

    print(f"\n[SUCCESS] All architectural ablations finished. Full results in {csv_path}")


if __name__ == "__main__":
    main()