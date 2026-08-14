"""
=============================================================
Class-wise Average Dice Score — Standalone Script
-------------------------------------------------------------
Runs the trained BADF-Net on the test set and produces ONE
CSV containing the average Dice Score per anatomical class
(Background excluded, matching the confusion matrix).
=============================================================
"""

import os
import csv
import argparse
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from network.BADF_Net import BADF_Net
from dataloader import get_kidney_loader

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Class-wise Average Dice Score")
    parser.add_argument('--test_images_dir', type=str, default='Kidney_dataset/val/images')
    parser.add_argument('--test_masks_dir', type=str, default='Kidney_dataset/val/mask')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='checkpoints/Results/Test')
    parser.add_argument('--class_names', type=str, nargs='+',
                         default=['Background', 'Capsule', 'Central Echo Complex', 'Medulla', 'Cortex', 'Spleen'],
                         help='Index 0 = Background, excluded from the output')
    return parser.parse_args()


def dice_per_class(pred, gt, num_classes, eps=1e-7):
    """Per-class Dice for one image; class index 0 (background) is skipped."""
    scores = {}
    for class_id in range(1, num_classes):
        pred_c = (pred == class_id).astype(np.float32)
        gt_c = (gt == class_id).astype(np.float32)
        intersection = (pred_c * gt_c).sum()
        union = pred_c.sum() + gt_c.sum()
        scores[class_id] = float((2.0 * intersection + eps) / (union + eps))
    return scores


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

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

    model = BADF_Net(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    per_class_scores = {c: [] for c in range(1, args.num_classes)}

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images = images.to(device)
            masks = masks.to(device).long().squeeze(1)

            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)

            pred_np = pred_masks.cpu().numpy()
            gt_np = masks.cpu().numpy()

            for b in range(pred_np.shape[0]):
                scores = dice_per_class(pred_np[b], gt_np[b], args.num_classes)
                for class_id, score in scores.items():
                    per_class_scores[class_id].append(score)

    csv_path = os.path.join(args.output_dir, "classwise_avg_dice.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Class Name", "Mean Dice", "Std Dice", "N Images"])
        for class_id in range(1, args.num_classes):
            scores = per_class_scores[class_id]
            name = args.class_names[class_id] if class_id < len(args.class_names) else f"Class {class_id}"
            mean_dice = round(float(np.mean(scores)), 4) if scores else ""
            std_dice = round(float(np.std(scores)), 4) if scores else ""
            writer.writerow([class_id, name, mean_dice, std_dice, len(scores)])

    print(f"[SAVE] Class-wise average Dice saved to {csv_path}")


if __name__ == "__main__":
    main()