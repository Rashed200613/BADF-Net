import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import cv2

class KidneyDatasetV2(Dataset):
    def __init__(self, images_dir, masks_dir, trainsize=256, augmentations=False, zoom_factor_range=(0.9, 1.1), crop_size=(256, 256), save_augmented=False, save_dir=None, mean=None, std=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.zoom_factor_range = zoom_factor_range  # The range for random zoom
        self.crop_size = crop_size
        self.save_augmented = save_augmented
        self.save_dir = save_dir if save_dir is not None else "augmented_images"
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]  # Default ImageNet mean
        self.std = std if std is not None else [0.229, 0.224, 0.225]  # Default ImageNet std

        if not os.path.exists(self.save_dir) and self.save_augmented:
            os.makedirs(self.save_dir)

        self.images = sorted(
            [os.path.join(self.images_dir, f) for f in os.listdir(self.images_dir) if f.endswith('_anon.png')])
        self.masks = sorted([os.path.join(self.masks_dir, f) for f in os.listdir(self.masks_dir) if f.endswith('_anon_gt.png')])

        valid_pairs = [(img, mask) for img, mask in zip(self.images, self.masks) if
                       os.path.basename(img).replace("_anon.png", "_anon_gt.png") == os.path.basename(mask)]
        if not valid_pairs:
            raise ValueError("No valid image-mask pairs found!")

        self.images, self.masks = zip(*valid_pairs)
        self.size = len(self.images)

        # Base transformation for converting to tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return self.size

    def _apply_augmentations(self, image, mask):
        """Apply the same augmentations to both image and mask."""
        # Convert to PIL Images
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)

        # Resize both to trainsize (always applied)
        image = TF.resize(image, (self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.trainsize, self.trainsize), interpolation=transforms.InterpolationMode.NEAREST)

        if self.augmentations:
            # Set random seed for consistency
            seed = random.randint(0, 9999)
            random.seed(seed)
            torch.manual_seed(seed)

            # Random Rotation (same angle for both)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

            # Random Horizontal Flip (same decision for both)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip (same decision for both)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random Zooming (same zoom factor for both)
            zoom_factor = random.uniform(self.zoom_factor_range[0], self.zoom_factor_range[1])
            new_size = (int(self.trainsize * zoom_factor), int(self.trainsize * zoom_factor))
            image = TF.resize(image, new_size, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.resize(mask, new_size, interpolation=transforms.InterpolationMode.NEAREST)
            image = TF.center_crop(image, self.crop_size)
            mask = TF.center_crop(mask, self.crop_size)

        return image, mask

    def __getitem__(self, index):
        # Load image and mask
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[index], cv2.IMREAD_GRAYSCALE)

        # Apply synchronized augmentations
        image, mask = self._apply_augmentations(image, mask)

        # Convert to tensors
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)

        # Ensure mask is long tensor for segmentation (assuming mask values are 0-255)
        mask = (mask * 255).long()  # Remove *255 if masks are already 0-255
        mask = mask.squeeze(0)  # Remove channel dimension for mask (H, W)

        # Save the augmented image and mask if the flag is True
        if self.save_augmented:
            image_filename = os.path.join(self.save_dir, f"augmented_image_{index}.png")
            mask_filename = os.path.join(self.save_dir, f"augmented_mask_{index}.png")

            image_pil = transforms.ToPILImage()(image)  # Convert tensor back to PIL image
            image_pil.save(image_filename)

            mask_pil = Image.fromarray(mask.squeeze().numpy().astype(np.uint8))  # Convert mask to PIL
            mask_pil.save(mask_filename)

        return image, mask, self.images[index]


def get_kidney_loader(train_images_dir, train_masks_dir, val_images_dir, val_masks_dir, batchsize=4, trainsize=256, shuffle=True, num_workers=0, pin_memory=True, augmentation=True, zoom_factor_range=(0.9, 1.1), crop_size=(256, 256), save_augmented=False, save_dir=None, mean=None, std=None):
    # Create Dataset objects for train and validation datasets
    train_dataset = KidneyDatasetV2(train_images_dir, train_masks_dir, trainsize, augmentations=augmentation, zoom_factor_range=zoom_factor_range, crop_size=crop_size, save_augmented=save_augmented, save_dir=save_dir, mean=mean, std=std)
    val_dataset = KidneyDatasetV2(val_images_dir, val_masks_dir, trainsize, augmentations=False, zoom_factor_range=zoom_factor_range, crop_size=crop_size, save_augmented=False, save_dir=save_dir, mean=mean, std=std)  # No augmentation for validation

    # Define collate function to properly stack images and masks
    def collate_fn(batch):
        images, masks, paths = zip(*batch)
        images = torch.stack(images)
        masks = torch.stack(masks)
        return images, masks, paths

    # DataLoader for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=shuffle, num_workers=num_workers,
                              pin_memory=pin_memory, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory, collate_fn=collate_fn)

    return train_loader, val_loader