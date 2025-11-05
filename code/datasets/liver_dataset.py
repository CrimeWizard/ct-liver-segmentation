import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

# ------------------------------------------------------------
#  LiverDataset
# ------------------------------------------------------------

class LiverDataset(Dataset):
    """
    PyTorch Dataset for LiTS-based liver/tumor segmentation.
    Expects preprocessed .npy slices and masks.
    """

    def __init__(self, images_dir, masks_dir, split_json,
                 augment=False, normalize=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.normalize = normalize

        # Load slice IDs
        with open(split_json, "r") as f:
            self.ids = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        slice_id = self.ids[idx]
        # Handle whether .npy is already in slice_id
        if not slice_id.endswith(".npy"):
            slice_id = f"{slice_id}.npy"

        img_path = os.path.join(self.images_dir, slice_id)
        msk_path = os.path.join(self.masks_dir, slice_id)

        # Load arrays
        image = np.load(img_path).astype(np.float32)
        mask = np.load(msk_path).astype(np.uint8)

        # Normalize image 0–1
        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # Convert masks to binary (1 = liver/tumor, 0 = background)
        mask = np.where(mask > 0, 1, 0).astype(np.float32)

        # Optional augmentations
        if self.augment:
            image, mask = self._augment(image, mask)

        # To tensors (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0)        # [1, H, W]
        mask = torch.from_numpy(mask).unsqueeze(0)          # [1, H, W]

        return image, mask, slice_id

    # --------------------------------------------------------
    def _augment(self, image, mask):
        """Random simple augmentations: flip / rotate / intensity jitter"""
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            angle = random.choice([90, 180, 270])
            image = np.rot90(image, k=angle // 90).copy()
            mask = np.rot90(mask, k=angle // 90).copy()
        if random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            image = np.clip(image * factor, 0, 1)
        return image, mask


# ------------------------------------------------------------
#  DataLoader factory
# ------------------------------------------------------------

def get_loaders(base_dir="../data/processed",
                batch_size=4, num_workers=2, augment_train=True):
    """
    Returns train, val, test DataLoaders using default MEVIS structure.
    """
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    splits_dir = os.path.join(base_dir, "splits")

    train_set = LiverDataset(images_dir, masks_dir,
                             os.path.join(splits_dir, "train_ids.json"),
                             augment=augment_train)
    val_set = LiverDataset(images_dir, masks_dir,
                           os.path.join(splits_dir, "val_ids.json"))
    test_set = LiverDataset(images_dir, masks_dir,
                            os.path.join(splits_dir, "test_ids.json"))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
