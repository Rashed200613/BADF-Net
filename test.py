"""
=============================================================
BADF-Net / BASE-UNet Testing & Evaluation Script
-------------------------------------------------------------
Loads a trained checkpoint and evaluates it on the test/val
set, reporting:
    Dice Score, HD95, ASD, Avg. Inference Time (s), Loss,
    PA (Pixel Accuracy), MPA (Mean Pixel Accuracy),
    Precision, Recall, F1-Score
Outputs:
    1. metrics_per_image.csv   (row per test image)
    2. metrics_summary.csv     (mean of all metrics, same
                                 layout as the results table)
    3. confusion_matrix.png    (large, readable annotations)
=============================================================
"""

import os
import time
import argparse
import warnings
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from network.BADF_Net import BADF_Net
from dataloader import get_kidney_loader

warnings.filterwarnings("ignore")


# =========================================================================
# Same Composite Loss as training (needed to report comparable Loss value)
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


class CompositeLoss(nn.Module):
    """L = 0.5 * Dice + 0.3 * Focal + 0.2 * Tversky"""
    def __init__(self, dice_smooth=1.0, focal_alpha=0.25, focal_gamma=2.0, tversky_alpha=0.7, tversky_beta=0.3):
        super().__init__()
        self.dice = DiceLoss(smooth=dice_smooth)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, logits, targets):
        return (0.5 * self.dice(logits, targets) +
                0.3 * self.focal(logits, targets) +
                0.2 * self.tversky(logits, targets))


# =========================================================================
# Argument Parser
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="BADF-Net Testing Script")
    parser.add_argument('--test_images_dir', type=str, default='Kidney_dataset/val/images', help='Path to test images')
    parser.add_argument('--test_masks_dir', type=str, default='Kidney_dataset/val/mask', help='Path to test masks')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of segmentation classes')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (1 recommended for accurate per-image timing)')
    parser.add_argument('--img_size', type=int, default=256, help='Input image resolution')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--output_dir', type=str, default='checkpoints/Results/Test', help='Directory to save CSVs and confusion matrix')
    parser.add_argument('--spacing', type=float, default=1.0, help='Pixel spacing (mm) used for HD95/ASD distance conversion; set to 1.0 if unknown')
    parser.add_argument('--cm_fontsize', type=int, default=22, help='Font size for confusion matrix cell annotations')
    parser.add_argument('--class_names', type=str, nargs='+',
                         default=['Background', 'Capsule', 'Central Echo Complex', 'Medulla', 'Cortex', 'Spleen'],
                         help='Anatomical class names, index 0..num_classes-1 (index 0 = Background, excluded from the confusion matrix)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


# =========================================================================
# Metric Utilities
# =========================================================================

def get_surface_points(binary_mask):
    """Boundary voxels of a binary mask via erosion (mask minus its erosion)."""
    if binary_mask.sum() == 0:
        return None
    eroded = binary_erosion(binary_mask)
    surface = binary_mask & (~eroded)
    return surface


def hd95_asd_per_class(pred_bin, gt_bin, spacing=1.0):
    """
    Returns (hd95, asd) for one binary class mask pair.
    Falls back to NaN when a class is absent in both pred and gt
    (no boundary to compare), and to a large penalty when it is
    present in only one of the two (total miss / false positive).
    """
    pred_surface = get_surface_points(pred_bin)
    gt_surface = get_surface_points(gt_bin)

    if pred_surface is None and gt_surface is None:
        return np.nan, np.nan
    if pred_surface is None or gt_surface is None:
        # one mask empty, the other not -> undefined distance, skip from mean
        return np.nan, np.nan

    gt_dist_map = distance_transform_edt(~gt_bin) * spacing
    pred_dist_map = distance_transform_edt(~pred_bin) * spacing

    dists_pred_to_gt = gt_dist_map[pred_surface]
    dists_gt_to_pred = pred_dist_map[gt_surface]

    all_dists = np.concatenate([dists_pred_to_gt, dists_gt_to_pred])
    hd95 = np.percentile(all_dists, 95)
    asd = all_dists.mean()
    return hd95, asd


def calculate_dice(pred, target, num_classes, eps=1e-7):
    dice_scores = []
    for class_id in range(1, num_classes):
        pred_class = (pred == class_id).astype(np.float32)
        target_class = (target == class_id).astype(np.float32)
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(float(dice))
    mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
    return dice_scores, mean_dice


def calculate_pa_mpa(pred, target, num_classes):
    """Pixel Accuracy (overall) and Mean Pixel Accuracy (mean of per-class recall-like accuracy)."""
    pa = float((pred == target).sum()) / target.size
    per_class_acc = []
    for class_id in range(num_classes):
        gt_mask = (target == class_id)
        if gt_mask.sum() == 0:
            continue
        acc = float((pred[gt_mask] == class_id).sum()) / gt_mask.sum()
        per_class_acc.append(acc)
    mpa = sum(per_class_acc) / len(per_class_acc) if per_class_acc else 0.0
    return pa, mpa


# =========================================================================
# Main Entry Point
# =========================================================================

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using execution device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data loader (reuses the val split of get_kidney_loader as the test loader;
    # point --test_images_dir / --test_masks_dir at your held-out test split)
    print(f"[DATA] Loading test data from {args.test_images_dir} ...")
    _, test_loader = get_kidney_loader(
        train_images_dir=args.test_images_dir,
        train_masks_dir=args.test_masks_dir,
        val_images_dir=args.test_images_dir,
        val_masks_dir=args.test_masks_dir,
        batchsize=args.batch_size,
        trainsize=args.img_size,
        shuffle=False,
        augmentation=False
    )
    print(f"[DATA] Loaded {len(test_loader)} test batches.")

    # Model
    print(f"[MODEL] Building BADF-Net with {args.num_classes} classes...")
    model = BADF_Net(num_classes=args.num_classes).to(device)
    print(f"[MODEL] Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = CompositeLoss().to(device)

    per_image_rows = []
    all_gt_flat = []
    all_pred_flat = []

    with torch.no_grad():
        for idx, (images, masks, patient_ids) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start_t = time.time()

            outputs = model(images)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start_t

            loss = criterion(outputs, masks).item()
            pred_masks = torch.argmax(outputs, dim=1)

            pred_np = pred_masks.cpu().numpy()
            gt_np = masks.cpu().numpy()

            for b in range(pred_np.shape[0]):
                pred_b = pred_np[b]
                gt_b = gt_np[b]

                class_dice, mean_dice = calculate_dice(pred_b, gt_b, args.num_classes)
                pa, mpa = calculate_pa_mpa(pred_b, gt_b, args.num_classes)

                hd95_list, asd_list = [], []
                for class_id in range(1, args.num_classes):
                    pred_bin = (pred_b == class_id)
                    gt_bin = (gt_b == class_id)
                    hd95, asd = hd95_asd_per_class(pred_bin, gt_bin, spacing=args.spacing)
                    if not np.isnan(hd95):
                        hd95_list.append(hd95)
                    if not np.isnan(asd):
                        asd_list.append(asd)
                mean_hd95 = float(np.mean(hd95_list)) if hd95_list else np.nan
                mean_asd = float(np.mean(asd_list)) if asd_list else np.nan

                gt_flat = gt_b.flatten()
                pred_flat = pred_b.flatten()
                # Only score classes actually present in this image's ground
                # truth -- scoring absent classes as precision/recall=0 via
                # zero_division is misleading noise, not a real error.
                present_labels = sorted(np.unique(gt_flat).tolist())
                precision = precision_score(gt_flat, pred_flat, labels=present_labels, average='macro', zero_division=0)
                recall = recall_score(gt_flat, pred_flat, labels=present_labels, average='macro', zero_division=0)
                f1 = f1_score(gt_flat, pred_flat, labels=present_labels, average='macro', zero_division=0)

                all_gt_flat.append(gt_flat)
                all_pred_flat.append(pred_flat)

                pid = patient_ids[b] if isinstance(patient_ids, (list, tuple)) else str(patient_ids)

                per_image_rows.append({
                    "PatientID": pid,
                    "Dice Score": round(mean_dice, 4),
                    "HD95": round(mean_hd95, 2) if not np.isnan(mean_hd95) else "",
                    "ASD": round(mean_asd, 2) if not np.isnan(mean_asd) else "",
                    "Time (s)": round(elapsed, 4),
                    "Loss": round(loss, 4),
                    "PA": round(pa, 4),
                    "MPA": round(mpa, 4),
                    "Precision": round(precision, 4),
                    "Recall": round(recall, 4),
                    "F1-Score": round(f1, 4),
                })

            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_loader):
                print(f"[TEST] Processed {idx + 1}/{len(test_loader)} batches")

    # ---------------------------------------------------------------
    # Save per-image CSV
    # ---------------------------------------------------------------
    per_image_csv = os.path.join(args.output_dir, "metrics_per_image.csv")
    fieldnames = ["PatientID", "Dice Score", "HD95", "ASD", "Time (s)", "Loss",
                  "PA", "MPA", "Precision", "Recall", "F1-Score"]
    with open(per_image_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)
    print(f"[SAVE] Per-image metrics saved to {per_image_csv}")

    # ---------------------------------------------------------------
    # Save summary CSV (mirrors the results-table layout)
    # ---------------------------------------------------------------
    def col_mean(key):
        vals = [row[key] for row in per_image_rows if row[key] != ""]
        return round(float(np.mean(vals)), 4) if vals else ""

    # Precision/Recall/F1 for the summary are computed from the aggregated
    # global confusion matrix (all test-set pixels pooled), NOT by averaging
    # the noisy per-image scores above. This avoids classes that are absent
    # from an individual image being counted as a 0 for that image, which
    # would drag the macro average down even though nothing was wrong.
    all_gt_concat = np.concatenate(all_gt_flat)
    all_pred_concat = np.concatenate(all_pred_flat)
    global_labels = list(range(args.num_classes))
    global_precision = precision_score(all_gt_concat, all_pred_concat, labels=global_labels, average='macro', zero_division=0)
    global_recall = recall_score(all_gt_concat, all_pred_concat, labels=global_labels, average='macro', zero_division=0)
    global_f1 = f1_score(all_gt_concat, all_pred_concat, labels=global_labels, average='macro', zero_division=0)

    summary_row = {
        "Dice Score": col_mean("Dice Score"),
        "HD95": col_mean("HD95"),
        "ASD": col_mean("ASD"),
        "Avg. Time (s)": col_mean("Time (s)"),
        "Loss": col_mean("Loss"),
        "PA": col_mean("PA"),
        "MPA": col_mean("MPA"),
        "Precision": round(float(global_precision), 4),
        "Recall": round(float(global_recall), 4),
        "F1-Score": round(float(global_f1), 4),
    }
    summary_csv = os.path.join(args.output_dir, "metrics_summary.csv")
    with open(summary_csv, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)
    print(f"[SAVE] Summary metrics saved to {summary_csv}")
    print("[SUMMARY]", summary_row)

    # ---------------------------------------------------------------
    # Confusion matrix (normalized, large readable annotations)
    # ---------------------------------------------------------------
    all_gt_flat = np.concatenate(all_gt_flat)
    all_pred_flat = np.concatenate(all_pred_flat)
    # Background (class 0) is excluded from the confusion matrix per request
    cm_labels = list(range(1, args.num_classes))
    cm_names = [args.class_names[i] if i < len(args.class_names) else f"Class {i}" for i in cm_labels]
    cm = confusion_matrix(all_gt_flat, all_pred_flat, labels=cm_labels)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig_size = max(8, len(cm_labels) * 1.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(cm_labels)))
    ax.set_yticks(np.arange(len(cm_labels)))
    ax.set_xticklabels(cm_names, fontsize=args.cm_fontsize - 4)
    ax.set_yticklabels(cm_names, fontsize=args.cm_fontsize - 4)
    ax.set_xlabel("Predicted label", fontsize=args.cm_fontsize)
    ax.set_ylabel("True label", fontsize=args.cm_fontsize)
    ax.set_title("Confusion Matrix", fontsize=args.cm_fontsize + 2, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    thresh = 0.5
    for i in range(len(cm_labels)):
        for j in range(len(cm_labels)):
            value = cm_norm[i, j]
            text_color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     color=text_color, fontsize=args.cm_fontsize, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=args.cm_fontsize - 4)

    fig.tight_layout()
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Confusion matrix saved to {cm_path}")

    print("\n[SUCCESS] Testing finished successfully!")


if __name__ == "__main__":
    main()