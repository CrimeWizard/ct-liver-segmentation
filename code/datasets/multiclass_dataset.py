# code/datasets/multiclass_dataset.py
# Phase 5 — Multi-class Dataset Loader (background / liver / tumor)

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import cv2   # <-- added for resizing


# ------------------------------------------------------------
#  MultiClassLiverDataset
# ------------------------------------------------------------

class MultiClassLiverDataset(Dataset):
    """
    PyTorch Dataset for LiTS-based 3-class segmentation.
    Expects preprocessed .npy slices:
        0 = background
        1 = liver
        2 = tumor
    """

    def __init__(self, images_dir, masks_dir, split_json,
                 augment=False, normalize=True, target_size=(256, 256)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augment
        self.normalize = normalize
        self.target_size = target_size

        # Load slice IDs from same JSON structure (train_ids.json, etc.)
        with open(split_json, "r") as f:
            self.ids = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        slice_id = self.ids[idx]
        if not slice_id.endswith(".npy"):
            slice_id = f"{slice_id}.npy"

        img_path = os.path.join(self.images_dir, slice_id)
        msk_path = os.path.join(self.masks_dir, slice_id)

        # Load arrays
        image = np.load(img_path).astype(np.float32)
        mask = np.load(msk_path).astype(np.uint8)   # <-- already 0/1/2

        # Normalize image 0–1
        if self.normalize:
            image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)

        # --------------------------------------------------------
        # Resize both to 256×256 (same as Phase 3 baseline)
        # --------------------------------------------------------
        H, W = self.target_size
        image = cv2.resize(image, (W, H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (W, H), interpolation=cv2.INTER_NEAREST)
        # --------------------------------------------------------

        # Optional augmentations (flip, rotate, intensity jitter)
        if self.augment:
            image, mask = self._augment(image, mask)

        # To tensors
        image = torch.from_numpy(image).unsqueeze(0).float()   # [1, H, W]
        mask = torch.from_numpy(mask).long()                   # [H, W], int64 for CrossEntropyLoss
        return image, mask, slice_id

    # --------------------------------------------------------
    def _augment(self, image, mask):
        """Random spatial/intensity augmentations applied jointly."""
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            image = np.rot90(image, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        if random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            image = np.clip(image * factor, 0, 1)
        return image, mask


# ------------------------------------------------------------
#  DataLoader factory
# ------------------------------------------------------------

def get_loaders(base_dir="../data/processed",
                batch_size=4, num_workers=2, augment_train=True,
                multiclass=True):
    """
    Returns train, val, test DataLoaders using the same MEVIS JSON split format.
    """
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks_multiclass")
    splits_dir = os.path.join(base_dir, "splits")

    train_set = MultiClassLiverDataset(images_dir, masks_dir,
                                       os.path.join(splits_dir, "train_ids.json"),
                                       augment=augment_train)
    val_set = MultiClassLiverDataset(images_dir, masks_dir,
                                     os.path.join(splits_dir, "val_ids.json"))
    test_set = MultiClassLiverDataset(images_dir, masks_dir,
                                      os.path.join(splits_dir, "test_ids.json"))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)
    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
#  Self-test
# ------------------------------------------------------------
if __name__ == "__main__":
    base_dir = "../data/processed"
    train_loader, val_loader, test_loader = get_loaders(base_dir)
    imgs, masks, ids = next(iter(train_loader))
    print("Images:", imgs.shape, imgs.dtype)
    print("Masks:", masks.shape, masks.unique())
    print("Example ID:", ids[0])
