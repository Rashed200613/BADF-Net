"""
=============================================================
Statistical Analysis on the Test Set — Q1-paper-grade Report
-------------------------------------------------------------
Runs the trained BADF-Net on the test set and produces:

    1. overall_statistics.csv
       Mean, Std, Median, Min, Max, 95% CI (bootstrap), and a
       Shapiro-Wilk normality p-value for: Dice, PA, MPA, Loss,
       Precision, Recall, F1-Score, Inference Time.

    2. classwise_statistics.csv
       Same statistics, per anatomical class, for: Dice, HD95,
       ASD, Precision, Recall, F1-Score. An image only
       contributes to a class's distribution if that class is
       present in that image's ground truth (avoids diluting
       the stats with images where the structure doesn't
       appear at all).

    3. dice_boxplot_per_class.png
       Boxplot of per-class Dice distributions -- useful
       supporting figure for a results section.

    4. (optional) significance_test.csv
       If --compare_csv points to another model's
       metrics_per_image.csv (matched by PatientID), runs a
       paired significance test (Wilcoxon signed-rank, plus
       paired t-test for reference) on Dice Score, reporting
       p-value, mean difference, 95% CI of the difference, and
       Cohen's d effect size -- the standard evidence a Q1
       reviewer looks for when a paper claims improvement over
       a baseline/SOTA.
=============================================================
"""

import os
import csv
import time
import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy import stats

from network.BADF_Net import BADF_Net
from dataloader import get_kidney_loader

warnings.filterwarnings("ignore")


# =========================================================================
# Composite Loss (same as training, needed to report a comparable Loss)
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


class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()

    def forward(self, logits, targets):
        return (0.5 * self.dice(logits, targets) +
                0.3 * self.focal(logits, targets) +
                0.2 * self.tversky(logits, targets))


# =========================================================================
# Argument Parser
# =========================================================================

def get_args():
    parser = argparse.ArgumentParser(description="Statistical Analysis on Test Set")
    parser.add_argument('--test_images_dir', type=str, default='Kidney_dataset/val/images')
    parser.add_argument('--test_masks_dir', type=str, default='Kidney_dataset/val/mask')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='checkpoints/Results/Stats')
    parser.add_argument('--class_names', type=str, nargs='+',
                         default=['Background', 'Capsule', 'Central Echo Complex (CEC)', 'Medulla', 'Cortex', 'Spleen'])
    parser.add_argument('--spacing', type=float, default=1.0, help='Pixel spacing (mm) for HD95/ASD; 1.0 = pixel units')
    parser.add_argument('--n_boot', type=int, default=10000, help='Number of bootstrap resamples for 95%% CI')
    parser.add_argument('--compare_csv', type=str, default=None,
                         help='Optional: path to another model\'s metrics_per_image.csv (from test_badf_net.py) to run a paired significance test against, matched on PatientID')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# =========================================================================
# Metric helpers
# =========================================================================

def get_surface(binary_mask):
    if binary_mask.sum() == 0:
        return None
    eroded = binary_erosion(binary_mask)
    return binary_mask & (~eroded)


def hd95_asd(pred_bin, gt_bin, spacing=1.0):
    pred_s, gt_s = get_surface(pred_bin), get_surface(gt_bin)
    if pred_s is None or gt_s is None:
        return np.nan, np.nan
    gt_dist = distance_transform_edt(~gt_bin) * spacing
    pred_dist = distance_transform_edt(~pred_bin) * spacing
    all_d = np.concatenate([gt_dist[pred_s], pred_dist[gt_s]])
    return float(np.percentile(all_d, 95)), float(all_d.mean())


def binary_prf1(pred_bin, gt_bin, eps=1e-7):
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall) / (precision + recall + eps)
    return precision, recall, f1


def bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    """Percentile bootstrap 95% CI for the mean."""
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[i] = sample.mean()
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return (float(lower), float(upper))


def describe(data, n_boot=10000, seed=42):
    data = np.asarray(data, dtype=np.float64)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {"Mean": "", "Std": "", "Median": "", "Min": "", "Max": "",
                "CI95_Lower": "", "CI95_Upper": "", "Shapiro_p": "", "N": 0}
    ci_lo, ci_hi = bootstrap_ci(data, n_boot=n_boot, seed=seed)
    shapiro_p = float(stats.shapiro(data).pvalue) if 3 <= len(data) <= 5000 else np.nan
    return {
        "Mean": round(float(np.mean(data)), 4),
        "Std": round(float(np.std(data)), 4),
        "Median": round(float(np.median(data)), 4),
        "Min": round(float(np.min(data)), 4),
        "Max": round(float(np.max(data)), 4),
        "CI95_Lower": round(ci_lo, 4),
        "CI95_Upper": round(ci_hi, 4),
        "Shapiro_p": round(shapiro_p, 4) if not np.isnan(shapiro_p) else "",
        "N": len(data),
    }


def calculate_pa_mpa(pred, target, num_classes):
    pa = float((pred == target).sum()) / target.size
    accs = []
    for c in range(num_classes):
        gt_mask = (target == c)
        if gt_mask.sum() == 0:
            continue
        accs.append(float((pred[gt_mask] == c).sum()) / gt_mask.sum())
    mpa = sum(accs) / len(accs) if accs else 0.0
    return pa, mpa


# =========================================================================
# Main
# =========================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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
    criterion = CompositeLoss().to(device)

    overall = {"PatientID": [], "Dice": [], "PA": [], "MPA": [], "Loss": [],
               "Precision": [], "Recall": [], "F1": [], "Time": []}
    classwise = {c: {"Dice": [], "HD95": [], "ASD": [], "Precision": [], "Recall": [], "F1": []}
                 for c in range(1, args.num_classes)}

    with torch.no_grad():
        for images, masks, patient_ids in test_loader:
            images = images.to(device)
            masks_t = masks.to(device).long().squeeze(1)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - t0

            loss = criterion(outputs, masks_t).item()
            pred_t = torch.argmax(outputs, dim=1)

            pred_np = pred_t.cpu().numpy()
            gt_np = masks_t.cpu().numpy()

            for b in range(pred_np.shape[0]):
                pred_b, gt_b = pred_np[b], gt_np[b]
                pid = patient_ids[b] if isinstance(patient_ids, (list, tuple)) else str(patient_ids)

                per_class_dice, per_class_p, per_class_r, per_class_f1 = [], [], [], []
                for c in range(1, args.num_classes):
                    gt_bin = (gt_b == c)
                    pred_bin = (pred_b == c)

                    if gt_bin.sum() == 0:
                        # structure absent from this image's ground truth ->
                        # don't let it dilute this class's distribution
                        continue

                    intersection = (pred_bin & gt_bin).sum()
                    union = pred_bin.sum() + gt_bin.sum()
                    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
                    precision, recall, f1 = binary_prf1(pred_bin, gt_bin)
                    hd95, asd = hd95_asd(pred_bin, gt_bin, spacing=args.spacing)

                    classwise[c]["Dice"].append(float(dice))
                    classwise[c]["Precision"].append(precision)
                    classwise[c]["Recall"].append(recall)
                    classwise[c]["F1"].append(f1)
                    if not np.isnan(hd95):
                        classwise[c]["HD95"].append(hd95)
                    if not np.isnan(asd):
                        classwise[c]["ASD"].append(asd)

                    per_class_dice.append(float(dice))
                    per_class_p.append(precision)
                    per_class_r.append(recall)
                    per_class_f1.append(f1)

                pa, mpa = calculate_pa_mpa(pred_b, gt_b, args.num_classes)

                overall["PatientID"].append(pid)
                overall["Dice"].append(np.mean(per_class_dice) if per_class_dice else np.nan)
                overall["PA"].append(pa)
                overall["MPA"].append(mpa)
                overall["Loss"].append(loss)
                overall["Precision"].append(np.mean(per_class_p) if per_class_p else np.nan)
                overall["Recall"].append(np.mean(per_class_r) if per_class_r else np.nan)
                overall["F1"].append(np.mean(per_class_f1) if per_class_f1 else np.nan)
                overall["Time"].append(elapsed)

    # ---------------------------------------------------------------
    # Overall statistics CSV
    # ---------------------------------------------------------------
    overall_path = os.path.join(args.output_dir, "overall_statistics.csv")
    with open(overall_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Mean", "Std", "Median", "Min", "Max", "CI95_Lower", "CI95_Upper", "Shapiro_p", "N"])
        for metric in ["Dice", "PA", "MPA", "Loss", "Precision", "Recall", "F1", "Time"]:
            d = describe(overall[metric], n_boot=args.n_boot, seed=args.seed)
            writer.writerow([metric, d["Mean"], d["Std"], d["Median"], d["Min"], d["Max"],
                              d["CI95_Lower"], d["CI95_Upper"], d["Shapiro_p"], d["N"]])
    print(f"[SAVE] Overall statistics saved to {overall_path}")

    # ---------------------------------------------------------------
    # Class-wise statistics CSV
    # ---------------------------------------------------------------
    classwise_path = os.path.join(args.output_dir, "classwise_statistics.csv")
    with open(classwise_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Class ID", "Class Name", "Metric", "Mean", "Std", "Median", "Min", "Max",
                          "CI95_Lower", "CI95_Upper", "Shapiro_p", "N"])
        for c in range(1, args.num_classes):
            name = args.class_names[c] if c < len(args.class_names) else f"Class {c}"
            for metric in ["Dice", "HD95", "ASD", "Precision", "Recall", "F1"]:
                d = describe(classwise[c][metric], n_boot=args.n_boot, seed=args.seed)
                writer.writerow([c, name, metric, d["Mean"], d["Std"], d["Median"], d["Min"], d["Max"],
                                  d["CI95_Lower"], d["CI95_Upper"], d["Shapiro_p"], d["N"]])
    print(f"[SAVE] Class-wise statistics saved to {classwise_path}")

    # ---------------------------------------------------------------
    # Boxplot of per-class Dice distributions
    # ---------------------------------------------------------------
    box_data = [classwise[c]["Dice"] for c in range(1, args.num_classes)]
    box_names = [args.class_names[c] if c < len(args.class_names) else f"Class {c}" for c in range(1, args.num_classes)]

    fig, ax = plt.subplots(figsize=(max(8, len(box_names) * 1.8), 6))
    ax.boxplot(box_data, labels=box_names, showmeans=True)
    ax.set_ylabel("Dice Score", fontsize=14)
    ax.set_title("Per-class Dice Score Distribution (Test Set)", fontsize=16, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=12)
    fig.tight_layout()
    box_path = os.path.join(args.output_dir, "dice_boxplot_per_class.png")
    fig.savefig(box_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVE] Dice boxplot saved to {box_path}")

    # ---------------------------------------------------------------
    # Optional paired significance test vs a baseline model
    # ---------------------------------------------------------------
    if args.compare_csv:
        import pandas as pd
        this_df = pd.DataFrame({"PatientID": overall["PatientID"], "Dice": overall["Dice"]})
        base_df = pd.read_csv(args.compare_csv)[["PatientID", "Dice Score"]].rename(columns={"Dice Score": "Dice_baseline"})
        merged = this_df.merge(base_df, on="PatientID", how="inner").dropna()

        if len(merged) < 2:
            print("[WARN] Fewer than 2 matched PatientIDs -- skipping significance test.")
        else:
            diffs = merged["Dice"].values - merged["Dice_baseline"].values
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(merged["Dice"], merged["Dice_baseline"])
            ttest_stat, ttest_p = stats.ttest_rel(merged["Dice"], merged["Dice_baseline"])
            cohens_d = float(np.mean(diffs) / np.std(diffs, ddof=1)) if np.std(diffs, ddof=1) > 0 else np.nan
            diff_ci_lo, diff_ci_hi = bootstrap_ci(diffs, n_boot=args.n_boot, seed=args.seed)

            sig_path = os.path.join(args.output_dir, "significance_test.csv")
            with open(sig_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "N_matched", "Mean_Diff", "Diff_CI95_Lower", "Diff_CI95_Upper",
                                  "Wilcoxon_stat", "Wilcoxon_p", "Paired_t_stat", "Paired_t_p", "Cohens_d"])
                writer.writerow(["Dice", len(merged), round(float(np.mean(diffs)), 4),
                                  round(diff_ci_lo, 4), round(diff_ci_hi, 4),
                                  round(float(wilcoxon_stat), 4), round(float(wilcoxon_p), 6),
                                  round(float(ttest_stat), 4), round(float(ttest_p), 6),
                                  round(cohens_d, 4) if not np.isnan(cohens_d) else ""])
            print(f"[SAVE] Paired significance test saved to {sig_path}")
            print(f"[RESULT] Wilcoxon p={wilcoxon_p:.6f} | Paired t-test p={ttest_p:.6f} | Cohen's d={cohens_d:.4f}")

    print("\n[SUCCESS] Statistical analysis finished successfully!")


if __name__ == "__main__":
    main()