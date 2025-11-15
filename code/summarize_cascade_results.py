import numpy as np
import re

log_file = "cascade_output.txt"    # the file containing your cascade printout

# Regex to catch the lines with Dice values
pattern = re.compile(
    r"(.+\.npy) \| Liver Dice=([0-9\.]+) \| Tumor Dice=([0-9\.]+)"
)

liver_dices = []
tumor_dices = []
tumor_pos_dices = []
slices_with_tumor_gt = 0
slices_with_liver_gt = 0
slices_no_liver_found = 0

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()

        # catch "no liver found"
        if "no liver found" in line:
            slices_no_liver_found += 1
            continue

        m = pattern.match(line)
        if m:
            fname, liver, tumor = m.groups()
            liver = float(liver)
            tumor = float(tumor)

            # liver always computed
            liver_dices.append(liver)

            # tumor: count only where GT tumor > 0
            # Your GT is inside the mask folder — this checks the reference mask
            vol = fname
            import os
            import numpy as np
            gt_mask = np.load(f"../data/processed/masks_multiclass/{vol}")

            if (gt_mask == 1).sum() > 0:   # has liver
                slices_with_liver_gt += 1

            if (gt_mask == 2).sum() > 0:   # has tumor
                slices_with_tumor_gt += 1
                tumor_pos_dices.append(tumor)

# ---- REPORT ----
print("===== CASCADE SUMMARY =====")

print(f"Total slices processed: {len(liver_dices)}")
print(f"Slices with 'no liver found': {slices_no_liver_found}")

print("\n--- Liver Segmentation ---")
print(f"Mean Liver Dice: {np.mean(liver_dices):.4f}")
print(f"Median Liver Dice: {np.median(liver_dices):.4f}")
print(f"Liver Dice std: {np.std(liver_dices):.4f}")

print("\n--- Tumor Segmentation (tumor slices only) ---")
if len(tumor_pos_dices) > 0:
    print(f"Tumor-positive slices: {slices_with_tumor_gt}")
    print(f"Tumor slices with nonzero prediction: {sum(d>0 for d in tumor_pos_dices)}")
    print(f"Mean Tumor Dice: {np.mean(tumor_pos_dices):.4f}")
    print(f"Median Tumor Dice: {np.median(tumor_pos_dices):.4f}")
    print(f"Max Tumor Dice: {np.max(tumor_pos_dices):.4f}")
else:
    print("No tumor-positive slices detected in predictions")
