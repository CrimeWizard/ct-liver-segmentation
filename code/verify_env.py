import torch, nibabel as nib, SimpleITK as sitk, monai, cv2, numpy as np

print("✅ Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
print("✅ nibabel:", nib.__version__)
print("✅ SimpleITK:", sitk.Version_VersionString())
print("✅ MONAI:", monai.__version__)
print("✅ OpenCV:", cv2.__version__)
