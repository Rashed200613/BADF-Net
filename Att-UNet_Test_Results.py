import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import argparse
from tqdm import tqdm

from network.AttU_Net import AttU_Net


class KidneyDatasetV2(Dataset):
    def __init__(self, images_dir, masks_dir, trainsize, image_extension='_anon.png', mask_extension='_anon_gt.png'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.trainsize = trainsize
        self.image_extension = image_extension
        self.mask_extension = mask_extension

        self.image_files = sorted(
            [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.endswith(self.image_extension)]
        )
        self.mask_files = sorted(
            [os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir) if f.endswith(self.mask_extension)]
        )

        print(f"Loaded {len(self.image_files)} images and {len(self.mask_files)} masks")
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks"

        self.image_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.image_transform(image)
        mask = self.mask_transform(mask)  # Assuming masks are 0-4

        patient_id = os.path.basename(image_path).replace(self.image_extension, '')
        return image, mask, patient_id


def get_kidney_loader(val_images_dir, val_masks_dir, batchsize, trainsize, image_extension='_anon.png',
                     mask_extension='_anon_gt.png'):
    val_dataset = KidneyDatasetV2(
        val_images_dir, val_masks_dir, trainsize, image_extension=image_extension, mask_extension=mask_extension
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=batchsize, shuffle=False)
    return val_loader


def calculate_metrics(pred, target, num_classes, eps=1e-7):
    metrics = {'AC': [], 'PR': [], 'SE': [], 'SP': [], 'Dice': [], 'IoU': []}

    for class_id in range(num_classes):
        pred_class = (pred == class_id).float()
        target_class = (target == class_id).float()
        not_target_class = (target != class_id).float()

        TP = (pred_class * target_class).sum().item()
        TN = ((1 - pred_class) * not_target_class).sum().item()
        FP = (pred_class * (1 - target_class)).sum().item()
        FN = ((1 - pred_class) * target_class).sum().item()

        AC = (TP + TN) / (TP + TN + FP + FN + eps) if (TP + TN + FP + FN + eps) > 0 else 0
        PR = TP / (TP + FP + eps) if (TP + FP + eps) > 0 else 0
        SE = TP / (TP + FN + eps) if (TP + FN + eps) > 0 else 0
        SP = TN / (TN + FP + eps) if (TN + FP + eps) > 0 else 0
        Dice = (2 * TP + eps) / (2 * TP + FP + FN + eps) if (2 * TP + FP + FN + eps) > 0 else 0
        IoU = TP / (TP + FP + FN + eps) if (TP + FP + FN + eps) > 0 else 0

        metrics['AC'].append(AC)
        metrics['PR'].append(PR)
        metrics['SE'].append(SE)
        metrics['SP'].append(SP)
        metrics['Dice'].append(Dice)
        metrics['IoU'].append(IoU)

    avg_metrics = {key: np.nanmean(values) if values else 0.0 for key, values in metrics.items()}
    return avg_metrics


def calculate_class_dice(pred, target, class_id, eps=1e-7):
    pred_tensor = torch.from_numpy(pred).float().cuda()
    target_tensor = torch.from_numpy(target).float().cuda()
    pred_class = (pred_tensor == class_id).float()
    target_class = (target_tensor == class_id).float()

    intersection = (pred_class * target_class).sum().item()
    union = pred_class.sum().item() + target_class.sum().item()
    dice = (2 * intersection + eps) / (union + eps) if (union + eps) > 0 else 0.0
    return dice


def save_test_stats(metrics, output_dir):
    csv_file_path = os.path.join(output_dir, "test_stats.csv")
    data = [metrics['AC'], metrics['PR'], metrics['SE'], metrics['SP'], metrics['Dice'], metrics['IoU']]
    header = ["AC", "PR", "SE", "SP", "Dice", "IoU"]
    df = pd.DataFrame([data], columns=header)
    df.to_csv(csv_file_path, index=False, mode='w')


def save_class_wise_test_dice(class_dice_scores, output_dir, num_classes):
    csv_file_path = os.path.join(output_dir, "class_wise_test_dice.csv")
    avg_class_dice = [np.nanmean(class_dice_scores[i]) for i in range(num_classes) if i in class_dice_scores and class_dice_scores[i]]
    header = [f"class_{i}_dice" for i in range(num_classes) if i in class_dice_scores]
    df = pd.DataFrame([avg_class_dice], columns=header)
    df.to_csv(csv_file_path, index=False, mode='w')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_images_dir', type=str, default='Kidney_dataset/val/images')
    parser.add_argument('--val_masks_dir', type=str, default='Kidney_dataset/val/mask')
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--snapshot_path', type=str, default='AttUNet_RESULTS/')
    parser.add_argument('--image_extension', type=str, default='_anon.png')
    parser.add_argument('--mask_extension', type=str, default='_anon_gt.png')
    args = parser.parse_args()

    snapshot_path = os.path.join('Att-UNet_Test_Results')
    output_dir = os.path.join(snapshot_path, "Results")
    os.makedirs(output_dir, exist_ok=True)

    val_loader = get_kidney_loader(
        val_images_dir=args.val_images_dir,
        val_masks_dir=args.val_masks_dir,
        batchsize=args.batch_size,
        trainsize=args.img_size,
        image_extension=args.image_extension,
        mask_extension=args.mask_extension
    )

    model = AttU_Net(num_classes=args.num_classes)
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.snapshot_path, "best_model.pth")))
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='mean')

    test_loss = 0.0
    num_test_samples = 0
    test_class_dice_scores = {i: [] for i in range(args.num_classes)}
    all_pred_masks = []
    all_gt_masks = []

    with torch.no_grad():
        for batch_idx, (images, masks, patient_ids) in enumerate(tqdm(val_loader, desc="Testing")):
            images, masks = images.cuda(), masks.cuda().long().squeeze(1)
            batch_size = images.size(0)
            num_test_samples += batch_size

            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            batch_loss = criterion(outputs, masks)
            test_loss += batch_loss.item() * batch_size

            all_pred_masks.extend(pred_masks.cpu().numpy())
            all_gt_masks.extend(masks.cpu().numpy())

            for i in range(batch_size):
                gt_mask = masks[i].cpu().numpy()
                pred_mask = pred_masks[i].cpu().numpy()
                for class_id in range(args.num_classes):
                    dice = calculate_class_dice(pred_mask, gt_mask, class_id)
                    test_class_dice_scores[class_id].append(dice)

    avg_test_loss = test_loss / num_test_samples
    all_pred_tensor = torch.tensor(np.array(all_pred_masks)).cuda()
    all_gt_tensor = torch.tensor(np.array(all_gt_masks)).cuda()
    overall_metrics = calculate_metrics(all_pred_tensor, all_gt_tensor, num_classes=args.num_classes)

    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print("Overall Metrics:")
    for key, value in overall_metrics.items():
        print(f"  {key}: {value:.4f}")

    save_test_stats(overall_metrics, output_dir)
    save_class_wise_test_dice(test_class_dice_scores, output_dir, args.num_classes)

    print("Test evaluation finished!")
    print(f"Test Stats saved to: {os.path.join(output_dir, 'test_stats.csv')}")
    print(f"Class-wise Dice saved to: {os.path.join(output_dir, 'class_wise_test_dice.csv')}")


if __name__ == "__main__":
    main()