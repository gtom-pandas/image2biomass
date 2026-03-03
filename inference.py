"""
CSIRO Image2Biomass - Inference Script   
=======================================
Ensemble inference with DINOv3 (FiLM fusion) and SigLIP models.
Includes TTA and physics-constrained post-processing.

Author: gtom-pandas
Competition: Kaggle CSIRO Biomass Challenge
"""

import os
import gc
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============
class CFG:
    """Inference configuration."""
    # Paths
    TEST_IMAGE_DIR = '/kaggle/input/csiro-biomass/test'
    TEST_CSV = '/kaggle/input/csiro-biomass/test.csv'
    DINOV3_MODEL_DIR = './models'
    SIGLIP_SUBMISSION = './submission_siglip.csv'
    OUTPUT_DIR = './'
    
    # Model
    MODEL_NAME = 'vit_large_patch16_dinov3_qkvb'
    
    # Inference
    IMG_SIZE = 512
    BATCH_SIZE = 1
    NUM_WORKERS = 4
    N_FOLDS = 5
    TTA_STEPS = 4  # Number of TTA views
    
    # Ensemble weights
    W_DINOV3 = 0.85  # ~85% weight for DINOv3
    W_SIGLIP = 0.15  # ~15% weight for SigLIP
    
    # Targets
    TARGET_COLS = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g', 'GDM_g', 'Dry_Total_g']
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# MODEL DEFINITION (must match training)
# ======================================
class FiLM(nn.Module):
    """Feature-wise Linear Modulation for dual-stream fusion."""
    def __init__(self, feat_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, feat_dim * 2)
        )
    def forward(self, context: torch.Tensor):
        gamma_beta = self.mlp(context)
        return torch.chunk(gamma_beta, 2, dim=1)  # γ, β


class BiomassModel(nn.Module):
    """
    DINOv3 + FiLM Fusion Model for biomass prediction.

    Architecture:
    - Dual-stream ViT backbone (vit_large_patch16_dinov3_qkvb)
      processes left and right image halves independently
    - FiLM module modulates both streams from their mean context
    - Concatenation of modulated features → (B, 2*nf)
    - Three regression heads for Green, Clover, Dead components
    - GDM and Total derived from primary predictions
    """
    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 backbone_path: str = None):
        super().__init__()
        self.model_name = model_name
        self.backbone_path = backbone_path

        self.backbone = timm.create_model(
            model_name, pretrained=False,
            num_classes=0, global_pool='avg')   # → (B, nf)

        if hasattr(self.backbone, 'set_grad_checkpointing') \
                and CFG.DINO_GRAD_CHECKPOINTING:
            self.backbone.set_grad_checkpointing(True)
            print("Gradient Checkpointing enabled")

        nf = self.backbone.num_features  # 1024

        self.film = FiLM(nf)

        def make_head():
            return nn.Sequential(
                nn.Linear(nf * 2, 8),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(8, 1),
                nn.Softplus()
            )

        self.head_green  = make_head()
        self.head_clover = make_head()
        self.head_dead   = make_head()

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        try:
            if self.backbone_path and os.path.exists(self.backbone_path):
                print(f"Loading backbone from: {self.backbone_path}")
                sd = torch.load(self.backbone_path, map_location='cpu')
                if 'model'        in sd: sd = sd['model']
                elif 'state_dict' in sd: sd = sd['state_dict']
            else:
                print("Downloading backbone weights from timm...")
                sd = timm.create_model(
                    self.model_name, pretrained=True,
                    num_classes=0, global_pool='avg').state_dict()
            self.backbone.load_state_dict(sd, strict=False)
            print('Pretrained weights loaded')
        except Exception as e:
            print(f'Warning: pretrained load failed: {e}')

    def forward(self, left: torch.Tensor,
                right: torch.Tensor):
        fl = self.backbone(left)               # (B, nf)
        fr = self.backbone(right)              # (B, nf)
        context = (fl + fr) / 2               # (B, nf)
        gamma, beta = self.film(context)       # each (B, nf)
        fl_mod = fl * (1 + gamma) + beta
        fr_mod = fr * (1 + gamma) + beta
        h = torch.cat([fl_mod, fr_mod], dim=1) # (B, 2*nf)
        g = self.head_green(h)
        c = self.head_clover(h)
        d = self.head_dead(h)
        gdm   = g + c
        total = gdm + d
        return total, gdm, g, c, d
# =============================================================================
# PREPROCESSING
# =============
def clean_image(img: np.ndarray) -> np.ndarray:
    """Clean image by removing artifacts."""
    h, w = img.shape[:2]
    img = img[0:int(h * 0.90), :]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([5, 150, 150])
    upper = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    
    if np.sum(mask) > 0:
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    
    return img


def get_tta_transforms(num_transforms: int):
    """
    Get TTA transform pipelines.
    
    Views:
    0: Original
    1: Horizontal Flip
    2: Vertical Flip
    3: Both Flips
    """
    normalize = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = ToTensorV2()
    
    all_transforms = [
        # Original
        A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), normalize, to_tensor]),
        # Horizontal Flip
        A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), A.HorizontalFlip(p=1.0), normalize, to_tensor]),
        # Vertical Flip
        A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), A.VerticalFlip(p=1.0), normalize, to_tensor]),
        # Both Flips
        A.Compose([A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE), A.HorizontalFlip(p=1.0), 
                   A.VerticalFlip(p=1.0), normalize, to_tensor]),
    ]
    
    return all_transforms[:num_transforms]


# =============================================================================
# DATASET
# ========
class BiomassTestDataset(Dataset):
    """Test dataset for biomass images."""
    
    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = [os.path.basename(p) for p in self.paths]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clean_image(img)
        
        h, w = img.shape[:2]
        mid = w // 2
        left = img[:, :mid].copy()
        right = img[:, mid:].copy()
        
        return left, right, self.filenames[idx]


# =============================================================================
# INFERENCE FUNCTIONS
# ===================
@torch.no_grad()
def predict_with_tta(model, left_np, right_np, tta_transforms):
    """Predict with test-time augmentation."""
    all_preds = []
    
    for tfm in tta_transforms:
        left_tensor = tfm(image=left_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        right_tensor = tfm(image=right_np)['image'].unsqueeze(0).to(CFG.DEVICE)
        
        total, gdm, green, clover, dead = model(left_tensor, right_tensor)
        
        preds = [total.cpu().item(), gdm.cpu().item(), green.cpu().item()]
        all_preds.append(preds)
    
    return np.mean(all_preds, axis=0)


def run_dinov3_inference():
    """Run DINOv3 model inference with fold ensemble."""
    print("\n" + "="*60)
    print("DINOv3 INFERENCE")
    print("="*60)
    
    dataset = BiomassTestDataset(CFG.TEST_IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)
    
    tta_transforms = get_tta_transforms(CFG.TTA_STEPS)
    accumulated_preds = np.zeros((len(dataset), 3), dtype=np.float32)
    filenames = dataset.filenames.copy()
    
    successful_folds = 0
    
    for fold in range(CFG.N_FOLDS):
        weight_path = os.path.join(CFG.DINOV3_MODEL_DIR, f'best_model_fold{fold}.pth')
        
        if not os.path.exists(weight_path):
            print(f"Warning: Model file {weight_path} not found, skipping fold {fold}")
            continue
        
        print(f"Loading fold {fold}...")
        model = BiomassModel(CFG.MODEL_NAME, pretrained=False)
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        model.to(CFG.DEVICE)
        model.eval()
        
        for i, (left, right, _) in enumerate(tqdm(loader, desc=f"Fold {fold}")):
            left_np = left[0].numpy()
            right_np = right[0].numpy()
            pred = predict_with_tta(model, left_np, right_np, tta_transforms)
            accumulated_preds[i] += pred
        
        successful_folds += 1
        del model
        torch.cuda.empty_cache()
        gc.collect()
    
    if successful_folds == 0:
        raise FileNotFoundError("No model weights found!")
    
    final_preds = accumulated_preds / successful_folds
    
    # Post-process: derive all 5 targets from [total, gdm, green]
    pred_total = final_preds[:, 0]
    pred_gdm = final_preds[:, 1]
    pred_green = final_preds[:, 2]
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    pred_dead = np.maximum(0, pred_total - pred_gdm)
    
    preds_all = np.stack([pred_green, pred_dead, pred_clover, pred_gdm, pred_total], axis=1)
    
    return preds_all, filenames


# =============================================================================
# POST-PROCESSING
# ================
def enforce_mass_balance(df_wide: pd.DataFrame, fixed_clover: bool = True) -> pd.DataFrame:
    """
    Enforce biological constraints using orthogonal projection:
    1. Dry_Green + Dry_Clover = GDM
    2. GDM + Dry_Dead = Dry_Total
    """
    ordered_cols = ['Dry_Green_g', 'Dry_Clover_g', 'Dry_Dead_g', 'GDM_g', 'Dry_Total_g']
    Y = df_wide[ordered_cols].values.T
    
    if fixed_clover:
        clover_fixed = Y[1, :].copy()
        Y[3, :] = Y[0, :] + clover_fixed  # GDM
        Y[4, :] = Y[3, :] + Y[2, :]  # Total
        Y_reconciled = Y
    else:
        C = np.array([
            [1, 1, 0, -1, 0],
            [0, 0, 1, 1, -1]
        ])
        C_T = C.T
        inv_CCt = np.linalg.inv(C @ C_T)
        P = np.eye(5) - C_T @ inv_CCt @ C
        Y_reconciled = P @ Y
    
    Y_reconciled = Y_reconciled.T
    Y_reconciled = np.maximum(0, Y_reconciled)
    
    df_out = df_wide.copy()
    df_out[ordered_cols] = Y_reconciled
    return df_out


def ensemble_predictions(dinov3_preds, dinov3_filenames):
    """
    Ensemble DINOv3 and SigLIP predictions.
    Uses DINOv3 only for Dry_Clover_g.
    """
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTIONS")
    print("="*60)
    
    
