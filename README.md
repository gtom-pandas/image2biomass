# Image2Biomass
My first kaggle challenge where i finish with a bronze medal

"" For the short story : I got a bronze medal in the public LB and miserably forget to unselect my first attempts so i miss the silver medal in private LB (my best attempt was 0,61724 = 167th place)
It was my first kaggle competition, I learn so many things thanks to the forum and participants. Next time I will choose my submissions correctly :) ""

# CSIRO Image2Biomass Prediction:
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)](https://pytorch.org/)
[![Competition](https://img.shields.io/badge/Kaggle-CSIRO_Biomass-20beff)](https://www.kaggle.com/competitions/csiro-biomass)

This repository contains the source code and documentation for my solution to the **CSIRO - Image2Biomass Prediction** challenge on Kaggle.

**Performance:**
* **Ranking:** Top ~8% (Unofficial Silver Medal) on Private Leaderboard.

## Challenge Overview
The goal was to estimate pasture biomass (green, dead, clover, and total) from images to help Australian farmers optimize grazing management. The challenge involved:
* **Input:** High-resolution field images.
* **Targets:** 5 continuous variables (`Dry_Green_g`, `Dry_Dead_g`, `Dry_Clover_g`, `GDM_g`, `Dry_Total_g`).
* **Constraints:** Physical consistency (e.g., `Green + Clover = GDM`). In data, all samples from the state of "WA" have a dead value of 0. Therefore, I applied a post-processing step, setting the dead value of all samples with wa_prob > 0.75 to 0.
---

## Solution Architecture

Although it was my first kaggle competition, I've tried multiple architectures and techniques (94 submissions in 2 months) to obtain the most reliable results, I detail my improved submission below:
My approach relies on a **dual-stream ensemble** leveraging Vision Transformers (ViT) and Mamba (SSM) for feature fusion.

### 1. Preprocessing Pipeline
* **Artifact Removal:** Automated inpainting of orange timestamps using HSV color masking to prevent the model from learning metadata shortcuts.
* **Crop & Clean:** Removal of the bottom 10% of images to eliminate tripod artifacts.
* **Split-View Strategy:** Images are split into Left/Right halves to handle the wide aspect ratio and double the effective batch size. (image 2000x1000px)

### 2. Model 1: DINOv3 + Mamba Fusion
This is the primary regression model (~85% ensemble weight).
* **Backbone:** `vit_huge_plus_patch16_dinov3` (Frozen for first 2 epochs, then fine-tuned with gradient checkpointing).
* **Neck - Mamba Fusion:** A **State Space Model (SSM)** block is used to fuse tokens from the Left and Right image crops. This allows the model to capture global context across the entire panoramic view without quadratic complexity.
* **Heads:** Separate regression heads for *Green*, *Clover*, and *Dead* components.

### 3. Model 2: SigLIP Semantic Extractor
Used as a secondary model (~15% ensemble weight) to capture semantic features.
* **Backbone:** `google/siglip-so400m-patch14-384` (frozen, used as feature extractor).
* **Method:** 
  1. Extract image embeddings (1152-dim) from frozen SigLIP
  2. Generate semantic features via text probing (green/dead/clover concepts)
  3. Apply feature engineering (PCA, PLS, GMM)
  4. Train an ensemble of Gradient Boosting models (CatBoost, LightGBM, HistGB, GB)

### 4. Training Strategy
* **Loss Function:** `HuberLoss` (SmoothL1) with `beta=5.0` to be robust against outliers.
* **Optimizer:** AdamW with Cosine Annealing Warmup scheduler.
* **Augmentation:** Horizontal/Vertical flips, Rotations, and Color Jittering via `Albumentations`.

---

## Inference & Post-Processing

The inference pipeline is designed for robustness and physical consistency.

### 1. Test-Time Augmentation (TTA)
Apply **4x TTA** during inference:
* Original
* Horizontal Flip
* Vertical Flip
* Both Flips (H+V)
Predictions are averaged across views to reduce variance.
After the challenge, many participants noticed that TTA wasn't useful, I did notice that TTA doesn't change anything in the public score but I prefer to keep it in case it impacts the private LB

### 2. Weighted Ensemble
The final prediction is a weighted average of the DINOv3 and SigLIP models.
* *Note:* For `Dry_Clover_g`, the system relies exclusively on the DINOv3 model, which showed superior detection capabilities for small details.

### 3. Physics-Constrained Post-Processing (Mass Balance)
The raw model outputs are not guaranteed to sum up correctly. I implemented an **Orthogonal Projection** method to enforce biological constraints:
```python
# Constraints enforced:
# 1. Dry_Green + Dry_Clover = GDM
# 2. GDM + Dry_Dead = Dry_Total
```

### Repository Structure
image2biomass/
│
├── README.md                 # Documentation du projet
├── requirements.txt          # Dépendances Python
│
├── Training
│   ├── train_dinov3.py          # Entraînement DINOv3 + Mamba Fusion (~85% poids)
│   └── train_siglip.py          # Entraînement SigLIP + Gradient Boosting (~15% poids)
│
├── Inference
    └── inference.py             # Inférence ensemble avec TTA et post-processing
  
## Usage
### Training
```bash
# Train DINOv3 model (5-fold CV)
python train_dinov3.py

# Train SigLIP model
python train_siglip.py### Usage
```
### Inférence
```bash
python inference.py
