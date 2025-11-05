import os, numpy as np
from tqdm import tqdm

mask_dir = "../data/processed/masks"
mask_files = os.listdir(mask_dir)

tumor_slices = 0
liver_only_slices = 0
empty_slices = 0

for f in tqdm(mask_files):
    mask = np.load(os.path.join(mask_dir, f))
    unique_vals = np.unique(mask)
    if 2 in unique_vals:          # tumor label
        tumor_slices += 1
    elif 1 in unique_vals:        # liver only
        liver_only_slices += 1
    else:
        empty_slices += 1

total = tumor_slices + liver_only_slices + empty_slices

print(f"Total slices: {total}")
print(f"Slices with tumor: {tumor_slices} ({100 * tumor_slices/total:.2f}%)")
print(f"Slices with liver only: {liver_only_slices} ({100 * liver_only_slices/total:.2f}%)")
print(f"Empty slices: {empty_slices} ({100 * empty_slices/total:.2f}%)")
