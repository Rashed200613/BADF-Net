"""
=============================================================
BADF-Net / BASE-UNet Training Script
-------------------------------------------------------------
Integrated with:
    1. BADF_Net (Boundary-Aware Dynamic Fusion Network)
    2. KidneyDatasetV2 DataLoader
    3. Composite Loss (0.5 Dice + 0.5 Focal)
    4. Hardware fallback (CUDA / CPU)
    5. Training & Validation Loop with Checkpoint Saving
=============================================================
"""

import os
import sys
import time
import random
import argparse
import warnings
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from network.BADF_Net import BADF_Net
from dataloader import get_kidney_loader

warnings.filterwarnings("ignore")


# =========================================================================
# Multi-Objective Loss Functions: Dice, Focal & Composite Loss
# =========================================================================

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
        focal = self.alpha * ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


class CompositeLoss(nn.Module):
    """
    L = 0.5 * Dice + 0.5 * Focal
    """
    def __init__(self, dice_smooth=1.0, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, logits, targets):
        loss = (
            0.5 * self.dice(logits, targets) +
            0.5 * self.focal(logits, targets)
        )
        return loss


# =========================================================================
# Argument Parser
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="BADF-Net Training Script")
    parser.add_argument('--train_images_dir', type=str, default='Kidney_dataset/train/images', help='Path to training images')
    parser.add_argument('--train_masks_dir', type=str, default='Kidney_dataset/train/mask', help='Path to training masks')
    parser.add_argument('--val_images_dir', type=str, default='Kidney_dataset/val/images', help='Path to validation images')
    parser.add_argument('--val_masks_dir', type=str, default='Kidney_dataset/val/mask', help='Path to validation masks')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of segmentation classes')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Input image resolution')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--snapshot_path', type=str, default='checkpoints/', help='Path to save best model checkpoints')
    parser.add_argument('--augmentation', action='store_true', default=True, help='Enable data augmentations')
    parser.add_argument('--print_freq', type=int, default=10, help='Frequency of printing training iterations')
    parser.add_argument('--image_save_freq', type=int, default=50, help='Frequency of saving output overlay plots')
    return parser.parse_args()


# =========================================================================
# Evaluation & Logging Utilities
# =========================================================================

def calculate_dice(pred, target, num_classes, eps=1e-7):
    dice_scores = []
    for class_id in range(1, num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()

        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())

    mean_dice = sum(dice_scores) / len(dice_scores) if len(dice_scores) > 0 else 0.0
    return dice_scores, mean_dice


def save_image_with_overlay(original_image, gt_mask, pred_mask, epoch, iteration, output_dir, prefix="train"):
    original_image = original_image.cpu().numpy() if hasattr(original_image, 'cpu') else original_image
    if isinstance(original_image, str):
        original_image = cv2.imread(original_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if len(original_image.shape) == 3 and original_image.shape[0] == 3:
        original_image = np.transpose(original_image, (1, 2, 0))

    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        height, width, _ = original_image.shape
    elif len(original_image.shape) == 2:
        height, width = original_image.shape
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    gt_mask = gt_mask.cpu().numpy().astype(np.uint8)
    pred_mask = pred_mask.cpu().numpy().astype(np.uint8)

    color_mapping = {0: (0, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 0), 5: (255, 0, 255)}

    gt_color_map = np.zeros((height, width, 3), dtype=np.uint8)
    pred_color_map = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in color_mapping.items():
        gt_color_map[gt_mask == class_id] = color
        pred_color_map[pred_mask == class_id] = color

    original_with_gt_overlay = np.copy(original_image)
    original_with_pred_overlay = np.copy(original_image)

    original_with_gt_overlay[gt_mask != 0] = gt_color_map[gt_mask != 0]
    original_with_pred_overlay[pred_mask != 0] = pred_color_map[pred_mask != 0]

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    axes[0].imshow(original_with_gt_overlay)
    axes[0].set_title('Original with Ground Truth Mask')
    axes[0].axis('off')

    axes[1].imshow(original_with_pred_overlay)
    axes[1].set_title('Original with Predicted Mask')
    axes[1].axis('off')

    save_path = os.path.join(output_dir, f"{prefix}_epoch_{epoch + 1}_iter_{iteration + 1}.png")
    plt.savefig(save_path, bbox_inches='tight', transparent=False)
    plt.close()


def save_training_stats(epoch, dice_score, loss, output_dir):
    csv_file_path = os.path.join(output_dir, "training_stats.csv")
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Training_Dice", "Training_Loss"])
        writer.writerow([epoch, dice_score, loss])


def save_validation_stats(dice_score, loss, epoch, output_dir):
    csv_file_path = os.path.join(output_dir, "validation_stats.csv")
    file_exists = os.path.exists(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Epoch", "Dice_Score", "Loss"])
        writer.writerow([epoch, dice_score, loss])


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    args = get_args()

    # Hardware setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using execution device: {device}")

    # Reproducibility seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    # Output directories
    os.makedirs(args.snapshot_path, exist_ok=True)
    output_dir = os.path.join(args.snapshot_path, "Results")
    os.makedirs(output_dir, exist_ok=True)

    # Data loaders
    print(f"[DATA] Loading Kidney dataset (Batch Size: {args.batch_size}, Image Size: {args.img_size}x{args.img_size})...")
    train_loader, val_loader = get_kidney_loader(
        train_images_dir=args.train_images_dir,
        train_masks_dir=args.train_masks_dir,
        val_images_dir=args.val_images_dir,
        val_masks_dir=args.val_masks_dir,
        batchsize=args.batch_size,
        trainsize=args.img_size,
        shuffle=True,
        augmentation=args.augmentation
    )
    print(f"[DATA] Loaded {len(train_loader)} training batches and {len(val_loader)} validation batches.")

    # Model definition: BADF_Net
    print(f"[MODEL] Building BADF-Net model with {args.num_classes} classes...")
    model = BADF_Net(num_classes=args.num_classes).to(device)

    # Composite Loss & Optimizer
    criterion = CompositeLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=1e-4)

    best_val_dice = 0.0
    num_epochs = args.max_epochs

    print(f"[TRAIN] Starting BADF-Net training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        all_class_dice_scores = {i: [] for i in range(1, args.num_classes)}

        for iteration, (images, masks, patient_ids) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device).long().squeeze(1)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            pred_masks = torch.argmax(outputs, dim=1)
            class_dice_scores, avg_dice = calculate_dice(pred_masks, masks, num_classes=args.num_classes)

            for i, score in enumerate(class_dice_scores, 1):
                all_class_dice_scores[i].append(score)

            if (iteration + 1) % args.image_save_freq == 0:
                save_image_with_overlay(images[0], masks[0], pred_masks[0], epoch, iteration, output_dir)

            if (iteration + 1) % args.print_freq == 0 or (iteration + 1) == len(train_loader):
                print(f"[Epoch {epoch + 1}/{num_epochs}] [Batch {iteration + 1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Train Dice: {avg_dice:.4f}")

        avg_class_dice = [sum(scores) / len(scores) for scores in all_class_dice_scores.values() if len(scores) > 0]
        avg_epoch_dice = sum(avg_class_dice) / len(avg_class_dice) if len(avg_class_dice) > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)

        save_training_stats(epoch + 1, avg_epoch_dice, avg_train_loss, output_dir)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_class_dice_scores = {i: [] for i in range(1, args.num_classes)}

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device).long().squeeze(1)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                pred_masks = torch.argmax(outputs, dim=1)
                class_dice_scores, _ = calculate_dice(pred_masks, masks, num_classes=args.num_classes)

                for i, score in enumerate(class_dice_scores, 1):
                    val_class_dice_scores[i].append(score)

        avg_val_class_dice = [sum(scores) / len(scores) for scores in val_class_dice_scores.values() if len(scores) > 0]
        avg_val_dice = sum(avg_val_class_dice) / len(avg_val_class_dice) if len(avg_val_class_dice) > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader)

        save_validation_stats(avg_val_dice, avg_val_loss, epoch, output_dir)

        print(f"[SUMMARY] Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}")

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            checkpoint_file = os.path.join(args.snapshot_path, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"[SAVE] Saved new best checkpoint to {checkpoint_file} (Val Dice: {best_val_dice:.4f})")

    print("\n[SUCCESS] Training finished successfully!")


if __name__ == "__main__":
    main()