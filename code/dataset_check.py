import nibabel as nib
import numpy as np
import os

# === paths to your data ===
base_dir = os.path.expanduser("~/Documents/Programming/MEVIS_Project/data")
vol_path = os.path.join(base_dir, "volume_pt1", "volume-0.nii")
seg_path = os.path.join(base_dir, "segmentations", "segmentation-0.nii")

print("Loading volume and segmentation...")
vol_img = nib.load(vol_path)
seg_img = nib.load(seg_path)

# Convert to NumPy arrays
vol = vol_img.get_fdata()
seg = seg_img.get_fdata()

# === Basic info ===
print("Volume shape:", vol.shape)
print("Segmentation shape:", seg.shape)
print("Unique labels:", np.unique(seg))
print(f"Number of slices (Z-axis): {vol.shape[2]}")

# === Find where the liver/tumor appear ===
z_indices = np.where((seg > 0).any(axis=(0, 1)))[0]  # slices containing any label
if len(z_indices) > 0:
    print(f"Liver or tumor present between slices {z_indices[0]} and {z_indices[-1]}")
    print(f"Total slices containing anatomy: {len(z_indices)}")
else:
    print("No labeled regions found (check segmentation file).")

# === (Optional) show a few sample slices from the region of interest ===
import matplotlib.pyplot as plt

if len(z_indices) > 0:
    middle = z_indices[len(z_indices)//2]
    sample_slices = [z_indices[0], middle, z_indices[-1]]
    fig, axs = plt.subplots(1, len(sample_slices), figsize=(15,5))
    for i, idx in enumerate(sample_slices):
        axs[i].imshow(vol[:,:,idx], cmap='gray')
        axs[i].imshow(np.ma.masked_where(seg[:,:,idx]==0, seg[:,:,idx]),
                      cmap='autumn', alpha=0.4)
        axs[i].set_title(f"Slice {idx}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
