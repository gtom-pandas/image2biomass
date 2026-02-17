"""
CSIRO Image2Biomass - DINOv3 + Mamba Fusion Training Script
============================================================
Primary regression model using Vision Transformer backbone with 
State Space Model (Mamba) for left/right image fusion.

Author: gtom-pandas
Competition: Kaggle CSIRO Biomass Challenge
*due to privacy rules, I cannot share the dataset*
"""

import os
import gc
import math
import random
import warnings
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from timm.utils import ModelEmaV2
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============
class CFG:
    """Training configuration."""
    # Paths
    BASE_PATH = '/kaggle/input/csiro-biomass'
    TRAIN_CSV = os.path.join(BASE_PATH, 'train.csv')
    TRAIN_IMAGE_DIR = os.path.join(BASE_PATH, 'train')
    OUTPUT_DIR = './models'
    
    # Model
    MODEL_NAME = 'vit_huge_plus_patch16_dinov3.lvd1689m'
    BACKBONE_PATH = None  # Path to pretrained weights (optional)
    DINO_GRAD_CHECKPOINTING = True
    
    # Training
    SEED = 2100667
    N_FOLDS = 5
    FOLDS_TO_TRAIN = [0, 1, 2, 3, 4]
    IMG_SIZE = 512
    BATCH_SIZE = 2
    GRAD_ACC = 4
    NUM_WORKERS = 4
    EPOCHS = 30
    FREEZE_EPOCHS = 2
    WARMUP_EPOCHS = 3
    
    # Optimizer
    LR_BACKBONE = 5e-5
    LR_HEAD = 1e-3
    WEIGHT_DECAY = 1e-2
    CLIP_GRAD_NORM = 1.0
    
    # EMA
    EMA_DECAY = 0.9
    
    # Early Stopping
    PATIENCE = 5
    
    # Targets
    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    LOSS_WEIGHTS = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    R2_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2, 0.5])
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA PREPROCESSING
# ==================
def clean_image(img: np.ndarray) -> np.ndarray:
    """
    Clean image by removing artifacts.
    
    1. Crop bottom 10% to remove tripod artifacts
    2. Inpaint orange date stamps using HSV color masking
    """
    h, w = img.shape[:2]
    
    # Crop bottom 10%
    img = img[0:int(h * 0.90), :]
    
    # Inpaint orange date stamps
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return img


def get_train_transforms(img_size: int):
    """Training augmentations using Albumentations."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=10, p=0.3, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size: int):
    """Validation transforms (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class BiomassDataset(Dataset):
    """
    Dataset for biomass images.
    
    Splits each 2000x1000 image into left and right 1000x1000 halves
    to handle the wide aspect ratio.
    """
    
    def __init__(self, df: pd.DataFrame, transform, img_dir: str):
        self.df = df
        self.transform = transform
        self.img_dir = img_dir
        self.paths = df['image_path'].values
        self.labels = df[CFG.TARGET_COLS].values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, os.path.basename(self.paths[idx]))
        
        # Load and preprocess image
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        
        # Split into left and right halves
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid]
        right = img[:, mid:]
        
        # Apply transforms
        left = self.transform(image=left)['image']
        right = self.transform(image=right)['image']
        
        label = torch.from_numpy(self.labels[idx])
        return left, right, label


# =============================================================================
# MODEL ARCHITECTURE
# ==================
class LocalMambaBlock(nn.Module):
    """
    Lightweight Mamba-style block (Gated CNN) for token mixing.
    Efficiently mixes tokens with linear complexity using depthwise convolutions.
    """
    
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, 
                                 padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        shortcut = x
        x = self.norm(x)
        
        # Gating mechanism
        g = torch.sigmoid(self.gate(x))
        x = x * g
        
        # Spatial mixing via 1D Conv
        x = x.transpose(1, 2)  # (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, N, D)
        
        # Projection
        x = self.proj(x)
        x = self.drop(x)
        
        return shortcut + x


class BiomassModel(nn.Module):
    """
    DINOv3 + Mamba Fusion Model for biomass prediction.
    
    Architecture:
    - Dual-stream ViT backbone processes left and right image halves
    - Mamba fusion neck combines tokens from both views
    - Separate regression heads for Green, Clover, and Dead components
    - GDM and Total are derived from primary predictions
    """
    
    def __init__(self, model_name: str, pretrained: bool = True, backbone_path: str = None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path
        
        # Load backbone with global_pool='' to keep patch tokens
        self.backbone = timm.create_model(
            model_name, 
            pretrained=False, 
            num_classes=0, 
            global_pool=''
        )
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.backbone, 'set_grad_checkpointing') and CFG.DINO_GRAD_CHECKPOINTING:
            self.backbone.set_grad_checkpointing(True)
            print("✓ Gradient Checkpointing enabled (saves ~50% VRAM)")
        
        nf = self.backbone.num_features
        
        # Mamba Fusion Neck - mixes tokens from left and right images
        self.fusion = nn.Sequential(
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1),
            LocalMambaBlock(nf, kernel_size=5, dropout=0.1)
        )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression heads with Softplus for non-negative outputs
        def make_head():
            return nn.Sequential(
                nn.Linear(nf, nf // 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(nf // 2, 1),
                nn.Softplus()
            )
        
        self.head_green = make_head()
        self.head_clover = make_head()
        self.head_dead = make_head()
        
        if pretrained:
            self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained weights for the backbone."""
        try:
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone weights from: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                if 'model' in sd:
                    sd = sd['model']
                elif 'state_dict' in sd:
                    sd = sd['state_dict']
            else:
                print("Downloading backbone weights from timm...")
                sd = timm.create_model(
                    self.model_name, 
                    pretrained=True, 
                    num_classes=0, 
                    global_pool=''
                ).state_dict()
            
            self.backbone.load_state_dict(sd, strict=False)
            print('✓ Pretrained weights loaded')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')
    
    def forward(self, left, right):
        # Extract tokens from both image halves
        x_l = self.backbone(left)  # (B, N, D)
        x_r = self.backbone(right)  # (B, N, D)
        
        # Concatenate tokens along sequence dimension
        x_cat = torch.cat([x_l, x_r], dim=1)  # (B, 2N, D)
        
        # Mamba fusion - allows cross-view interaction
        x_fused = self.fusion(x_cat)
        
        # Global pooling
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)  # (B, D)
        
        # Prediction heads
        green = self.head_green(x_pool)
        clover = self.head_clover(x_pool)
        dead = self.head_dead(x_pool)
        
        # Derived targets (physical constraints)
        gdm = green + clover
        total = gdm + dead
        
        return total, gdm, green, clover, dead


# =============================================================================
# LOSS FUNCTION
# =============
def biomass_loss(outputs, labels, weights=None):
    """
    Huber loss (SmoothL1) for robust regression against outliers.
    
    Args:
        outputs: Tuple of (total, gdm, green, clover, dead) predictions
        labels: Ground truth tensor (B, 5) in order [Green, Dead, Clover, GDM, Total]
        weights: Optional per-target loss weights
    """
    total, gdm, green, clover, dead = outputs
    huber = nn.SmoothL1Loss(beta=5.0)
    
    l_green = huber(green.squeeze(), labels[:, 0])
    l_dead = huber(dead.squeeze(), labels[:, 1])
    l_clover = huber(clover.squeeze(), labels[:, 2])
    l_gdm = huber(gdm.squeeze(), labels[:, 3])
    l_total = huber(total.squeeze(), labels[:, 4])
    
    losses = torch.stack([l_green, l_dead, l_clover, l_gdm, l_total])
    
    if weights is None:
        return losses.mean()
    
    w = torch.as_tensor(weights, device=losses.device, dtype=losses.dtype)
    w = w / w.sum()
    return (losses * w).sum()


# =============================================================================
# METRICS
# =======
def weighted_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculate weighted R² score across all targets."""
    weights = CFG.R2_WEIGHTS
    r2_scores = []
    
    for i in range(y_true.shape[1]):
        yt, yp = y_true[:, i], y_pred[:, i]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        r2_scores.append(r2)
    
    r2_scores = np.array(r2_scores)
    weighted = np.sum(r2_scores * weights) / np.sum(weights)
    return weighted, r2_scores


# =============================================================================
# TRAINING FUNCTIONS
# ==================
def build_optimizer(model: BiomassModel):
    """Build optimizer with different learning rates for backbone and heads."""
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    
    backbone_params = []
    head_params = []
    
    for p in model.parameters():
        if p.requires_grad:
            if id(p) in backbone_ids:
                backbone_params.append(p)
            else:
                head_params.append(p)
    
    return optim.AdamW([
        {'params': backbone_params, 'lr': CFG.LR_BACKBONE, 'weight_decay': CFG.WEIGHT_DECAY},
        {'params': head_params, 'lr': CFG.LR_HEAD, 'weight_decay': CFG.WEIGHT_DECAY},
    ])


def build_scheduler(optimizer, num_epochs: int, warmup_epochs: int):
    """Cosine annealing scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def set_backbone_requires_grad(model: BiomassModel, requires_grad: bool):
    """Freeze or unfreeze backbone parameters."""
    for p in model.backbone.parameters():
        p.requires_grad = requires_grad


def train_epoch(model, loader, optimizer, scheduler, device, ema=None, scaler=None):
    """Train for one epoch with gradient accumulation."""
    model.train()
    running_loss = 0.0
    
    optimizer.zero_grad(set_to_none=True)
    
    for i, (left, right, labels) in enumerate(tqdm(loader, desc='Training', leave=False)):
        bs = left.size(0)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(left, right)
            loss = biomass_loss(outputs, labels, w=CFG.LOSS_WEIGHTS)
            loss = loss / CFG.GRAD_ACC
        
        # Skip non-finite losses
        if not torch.isfinite(loss):
            print("Non-finite loss detected, skipping batch")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        # Backward pass
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        running_loss += loss.detach().float().item() * bs * CFG.GRAD_ACC
        
        # Optimizer step with gradient accumulation
        if ((i + 1) % CFG.GRAD_ACC == 0) or ((i + 1) == len(loader)):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.CLIP_GRAD_NORM)
                optimizer.step()
            
            # Update EMA
            if ema is not None:
                ema.update(model)
            
            optimizer.zero_grad(set_to_none=True)
    
    scheduler.step()
    return running_loss / len(loader.dataset)


@torch.no_grad()
def validate_epoch(model, loader, device):
    """Validate for one epoch."""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    for left, right, labels in tqdm(loader, desc='Validating', leave=False):
        bs = left.size(0)
        left = left.to(device, non_blocking=True)
        right = right.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            total, gdm, green, clover, dead = model(left, right)
            loss = biomass_loss((total, gdm, green, clover, dead), labels, w=CFG.LOSS_WEIGHTS)
        
        total_loss += loss.detach().float().item() * bs
        
        # Collect predictions in target order
        preds = torch.cat([
            green.view(-1, 1),
            dead.view(-1, 1),
            clover.view(-1, 1),
            gdm.view(-1, 1),
            total.view(-1, 1)
        ], dim=1)
        
        all_preds.append(preds.float().cpu().numpy())
        all_labels.append(labels.float().cpu().numpy())
    
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    
    avg_loss = total_loss / len(loader.dataset)
    weighted_r2, per_target_r2 = weighted_r2_score(labels, preds)
    
    return avg_loss, weighted_r2, per_target_r2


# =============================================================================
# MAIN TRAINING LOOP
# ==================
def prepare_data():
    """Load and prepare training data with folds."""
    print("Loading training data...")
    df = pd.read_csv(CFG.TRAIN_CSV)
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        values='target',
        index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
        columns='target_name',
        aggfunc='mean'
    ).reset_index()
    
    # Create stratification groups
    df_wide['group'] = df_wide['image_path'].apply(lambda x: os.path.basename(x).split('_')[0])
    df_wide['strat_bin'] = pd.qcut(df_wide['Dry_Total_g'], q=10, labels=False, duplicates='drop')
    
    # Create folds using StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=CFG.N_FOLDS, shuffle=True, random_state=CFG.SEED)
    df_wide['fold'] = -1
    
    for fold, (_, val_idx) in enumerate(sgkf.split(df_wide, df_wide['strat_bin'], df_wide['group'])):
        df_wide.loc[val_idx, 'fold'] = fold
    
    print(f"✓ Data loaded: {len(df_wide)} samples")
    print(f"✓ Fold distribution:\n{df_wide['fold'].value_counts().sort_index()}")
    
    return df_wide


def train_fold(fold: int, df: pd.DataFrame):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"TRAINING FOLD {fold}")
    print(f"{'='*60}")
    
    # Split data
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    # Create datasets
    train_dataset = BiomassDataset(train_df, get_train_transforms(CFG.IMG_SIZE), CFG.TRAIN_IMAGE_DIR)
    val_dataset = BiomassDataset(val_df, get_val_transforms(CFG.IMG_SIZE), CFG.TRAIN_IMAGE_DIR)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.BATCH_SIZE,
        shuffle=True,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=CFG.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    model = BiomassModel(CFG.MODEL_NAME, pretrained=True, backbone_path=CFG.BACKBONE_PATH)
    model.to(CFG.DEVICE)
    
    # Freeze backbone initially
    set_backbone_requires_grad(model, False)
    
    # Optimizer, scheduler, EMA
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, CFG.EPOCHS, CFG.WARMUP_EPOCHS)
    ema = ModelEmaV2(model, decay=CFG.EMA_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # Training loop
    best_r2 = -float('inf')
    patience_counter = 0
    
    for epoch in range(CFG.EPOCHS):
        # Unfreeze backbone after freeze epochs
        if epoch == CFG.FREEZE_EPOCHS:
            print("✓ Unfreezing backbone")
            set_backbone_requires_grad(model, True)
            optimizer = build_optimizer(model)
            scheduler = build_scheduler(optimizer, CFG.EPOCHS - epoch, CFG.WARMUP_EPOCHS)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, CFG.DEVICE, ema, scaler)
        
        # Validate with EMA model
        val_loss, val_r2, per_r2 = validate_epoch(ema.module, val_loader, CFG.DEVICE)
        
        # Logging
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d}/{CFG.EPOCHS} | "
              f"LR: {lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val R²: {val_r2:.4f}")
        print(f"  Per-target R²: " + " | ".join([f"{c}: {r:.3f}" for c, r in zip(CFG.TARGET_COLS, per_r2)]))
        
        # Save best model
        if val_r2 > best_r2:
            best_r2 = val_r2
            patience_counter = 0
            
            save_path = os.path.join(CFG.OUTPUT_DIR, f'best_model_fold{fold}.pth')
            torch.save(ema.module.state_dict(), save_path)
            print(f"  ✓ New best model saved: R² = {best_r2:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= CFG.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    # Cleanup
    del model, ema, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_r2


def main():
    """Main training function."""
    print("="*60)
    print("CSIRO Image2Biomass - DINOv3 Training")
    print("="*60)
    
    # Setup
    seed_everything(CFG.SEED)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Device: {CFG.DEVICE}")
    print(f"  Model: {CFG.MODEL_NAME}")
    print(f"  Image Size: {CFG.IMG_SIZE}")
    print(f"  Batch Size: {CFG.BATCH_SIZE} x {CFG.GRAD_ACC} (effective)")
    print(f"  Epochs: {CFG.EPOCHS}")
    print(f"  Folds to train: {CFG.FOLDS_TO_TRAIN}")
    
    # Prepare data
    df = prepare_data()
    
    # Train folds
    fold_scores = {}
    for fold in CFG.FOLDS_TO_TRAIN:
        best_r2 = train_fold(fold, df)
        fold_scores[fold] = best_r2
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print("\nFold Results:")
    for fold, r2 in fold_scores.items():
        print(f"  Fold {fold}: R² = {r2:.4f}")
    print(f"\nMean R²: {np.mean(list(fold_scores.values())):.4f}")
    print(f"\nModels saved to: {CFG.OUTPUT_DIR}")


if __name__ == '__main__':
    main()