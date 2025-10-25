# 🩻 CT Liver Segmentation (MEVIS-Style AI Pipeline)

AI-driven liver segmentation and uncertainty estimation using a Fraunhofer MEVIS-inspired workflow.  
Implements preprocessing, 2D U-Net training, and Monte Carlo Dropout uncertainty analysis on the **LiTS** dataset.

---

## 📘 Overview
This project replicates and extends Fraunhofer MEVIS research workflows in medical image computing.

### 🧩 Workflow Phases
| Phase | Description | Status |
|-------|--------------|--------|
| 1️⃣ | Data verification and alignment | ✅ Complete |
| 2️⃣ | Preprocessing and slice extraction | ✅ Complete |
| 3️⃣ | Baseline U-Net training (Chlebus et al., 2018) | 🔄 In progress |
| 4️⃣ | Bayesian uncertainty estimation (Cangalovic et al., 2024) | ⏳ Planned |
| 5️⃣ | Visualization with MEVIS Draw | ⏳ Planned |

---

## ⚙️ Environment
**Environment name:** `fraunhofer`  
**Python version:** 3.10  

### Required libraries
```bash
pip install numpy matplotlib scikit-image tqdm SimpleITK torch torchvision
