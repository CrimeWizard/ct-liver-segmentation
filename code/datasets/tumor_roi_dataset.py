# Phase 6B — Tumor-in-ROI Dataset (binary)
import os, json, random, numpy as np, torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

class TumorROIDataset(Dataset):
    def __init__(self, base_dir="../data/processed/roi_tumor", split="train", augment=True):
        self.base_dir = base_dir
        self.img_dir = os.path.join(base_dir, "images")
        self.msk_dir = os.path.join(base_dir, "masks")
        with open(os.path.join(base_dir, "splits", f"{split}_ids.json"), "r") as f:
            self.ids = json.load(f)
        self.augment = augment and (split == "train")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        img = np.load(os.path.join(self.img_dir, sid)).astype(np.float32)  # [H,W] in 0..1
        msk = np.load(os.path.join(self.msk_dir, sid)).astype(np.uint8)    # 0/1 tumor

        # simple augmentations (in-place safe copies)
        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy(); msk = np.flip(msk, axis=1).copy()
            if random.random() < 0.5:
                img = np.flip(img, axis=0).copy(); msk = np.flip(msk, axis=0).copy()
            if random.random() < 0.3:
                k = random.choice([1,2,3])  # 90/180/270
                img = np.rot90(img, k).copy(); msk = np.rot90(msk, k).copy()
            if random.random() < 0.3:
                img = np.clip(img * random.uniform(0.9, 1.1), 0, 1)

        # to tensors: img -> [1,H,W], msk -> [H,W] (train script will unsqueeze channel)
        img = torch.from_numpy(img).unsqueeze(0).float()
        msk = torch.from_numpy(msk).float()  # BCE expects float target (0/1)
        return img, msk, sid

def get_loaders(base_dir="../data/processed/roi_tumor", batch_size=4, num_workers=2, only_tumor=False, use_sampler=False):
    """
    Returns (train_loader, val_loader, test_loader).
    - only_tumor: if True, train loader contains only tumor-positive slices.
    - use_sampler: if True and only_tumor is False, returns train loader with WeightedRandomSampler that oversamples tumor slices.
    """
    train = TumorROIDataset(base_dir, "train", augment=True)
    val   = TumorROIDataset(base_dir, "val",   augment=False)
    test  = TumorROIDataset(base_dir, "test",  augment=False)

    # If requested, filter train dataset to only tumor-positive examples
    if only_tumor:
        idxs = [i for i in range(len(train)) if train[i][1].sum().item() > 0]
        if len(idxs) == 0:
            raise RuntimeError("No tumor-positive slices found for only_tumor=True")
        train_subset = Subset(train, idxs)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    elif use_sampler:
        # Weighted sampler: give more weight to tumor slices
        weights = []
        for i in range(len(train)):
            mask = train[i][1].numpy()
            weights.append(20.0 if mask.sum() > 0 else 1.0)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tr, va, te = get_loaders()
    x,y,ids = next(iter(tr))
    print(x.shape, y.shape, ids[0])
