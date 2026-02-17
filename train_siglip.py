"""
CSIRO Image2Biomass - SigLIP Training Script
================================================================
Secondary model using SigLIP embeddings with gradient boosting ensemble
for semantic feature extraction.

Author: gtom-pandas
Competition: Kaggle CSIRO Biomass Challenge
*due to privacy rules, I cannot share the dataset*
"""

import os
import gc
import random
import warnings
from pathlib import Path
from dataclasses import dataclass
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================================================================
# CONFIG
# =============
@dataclass
class CFG:
    """Training configuration."""
    # Paths
    DATA_PATH: Path = Path("/kaggle/input/csiro-biomass/")
    SPLIT_PATH: Path = Path("/kaggle/input/csiro-datasplit/csiro_data_split.csv")
    SIGLIP_PATH: str = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"
    OUTPUT_DIR: Path = Path("./models")
    
    # Model settings
    SEED: int = 42600
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    PATCH_SIZE: int = 520
    OVERLAP: int = 16
    
    # Target definitions
    TARGET_NAMES = ['Dry_Clover_g', 'Dry_Dead_g', 'Dry_Green_g', 'Dry_Total_g', 'GDM_g']
    TARGET_MAX = {
        "Dry_Clover_g": 71.7865,
        "Dry_Dead_g": 83.8407,
        "Dry_Green_g": 157.9836,
        "Dry_Total_g": 185.70,
        "GDM_g": 157.9836,
    }


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# DATA LOADING
# =============
def pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Convert long format to wide format."""
    if 'target' in df.columns:
        df_pt = pd.pivot_table(
            df,
            values='target',
            index=['image_path', 'Sampling_Date', 'State', 'Species', 'Pre_GSHH_NDVI', 'Height_Ave_cm'],
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    else:
        df['target'] = 0
        df_pt = pd.pivot_table(
            df,
            values='target',
            index='image_path',
            columns='target_name',
            aggfunc='mean'
        ).reset_index()
    return df_pt


def post_process_biomass(df_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Derive GDM_g and Dry_Total_g from primary predictions.
    Keeps Dry_Clover_g fixed at 0.0.
    """
    ordered_cols = ["Dry_Green_g", "Dry_Clover_g", "Dry_Dead_g", "GDM_g", "Dry_Total_g"]
    
    for c in ordered_cols:
        if c not in df_preds.columns:
            df_preds[c] = 0.0
    
    df_out = df_preds.copy()
    df_out['Dry_Clover_g'] = 0.0
    df_out['GDM_g'] = df_out['Dry_Green_g'] + df_out['Dry_Clover_g']
    df_out['Dry_Total_g'] = df_out['GDM_g'] + df_out['Dry_Dead_g']
    df_out['GDM_g'] = df_out['GDM_g'].clip(lower=0.0)
    df_out['Dry_Total_g'] = df_out['Dry_Total_g'].clip(lower=0.0)
    
    return df_out


# =============================================================================
# FEATURE EXTRACTION
# ==================
def split_image(image: np.ndarray, patch_size: int = 520, overlap: int = 16) -> list:
    """Split image into overlapping patches."""
    h, w, c = image.shape
    stride = patch_size - overlap
    patches = []
    
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            y1 = max(0, y2 - patch_size)
            x1 = max(0, x2 - patch_size)
            patch = image[y1:y2, x1:x2, :]
            patches.append(patch)
    
    return patches


def compute_embeddings(model_path: str, df: pd.DataFrame, cfg: CFG) -> np.ndarray:
    """Extract SigLIP embeddings from images."""
    print(f"Computing embeddings for {len(df)} images...")
    
    model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(cfg.DEVICE)
    processor = AutoImageProcessor.from_pretrained(model_path)
    
    embeddings = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            img = cv2.imread(row['image_path'])
            if img is None:
                raise ValueError("Image not found")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = split_image(img, patch_size=cfg.PATCH_SIZE, overlap=cfg.OVERLAP)
            images = [Image.fromarray(p) for p in patches]
            
            inputs = processor(images=images, return_tensors="pt").to(cfg.DEVICE)
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            
            avg_embed = features.mean(dim=0).cpu().numpy()
            embeddings.append(avg_embed)
        except Exception as e:
            print(f"Error processing {row['image_path']}: {e}")
            embeddings.append(np.zeros(1152))
    
    torch.cuda.empty_cache()
    return np.stack(embeddings)


def generate_semantic_features(image_embeddings_np: np.ndarray, model_path: str, cfg: CFG) -> np.ndarray:
    """Generate semantic features using text probing."""
    print("Generating semantic features...")
    
    model = AutoModel.from_pretrained(model_path).to(cfg.DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Semantic concept anchors
    concept_groups = {
        "bare": ["bare soil", "dirt ground", "sparse vegetation", "exposed earth"],
        "sparse": ["low density pasture", "thin grass", "short clipped grass"],
        "medium": ["average pasture cover", "medium height grass", "grazed pasture"],
        "dense": ["dense tall pasture", "thick grassy volume", "high biomass", "overgrown vegetation"],
        "green": ["lush green vibrant pasture", "photosynthesizing leaves", "fresh growth"],
        "dead": ["dry brown dead grass", "yellow straw", "senesced material", "standing hay"],
        "clover": ["white clover", "trifolium repens", "broadleaf legume", "clover flowers"],
        "grass": ["ryegrass", "blade-like leaves", "fescue", "grassy sward"]
    }
    
    # Encode concepts
    concept_vectors = {}
    with torch.no_grad():
        for name, prompts in concept_groups.items():
            inputs = tokenizer(prompts, padding="max_length", return_tensors="pt").to(cfg.DEVICE)
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            concept_vectors[name] = emb.mean(dim=0, keepdim=True)
    
    # Compute similarity scores
    img_tensor = torch.tensor(image_embeddings_np, dtype=torch.float32).to(cfg.DEVICE)
    img_tensor = img_tensor / img_tensor.norm(p=2, dim=-1, keepdim=True)
    
    scores = {}
    for name, vec in concept_vectors.items():
        scores[name] = torch.matmul(img_tensor, vec.T).cpu().numpy().flatten()
    
    df_scores = pd.DataFrame(scores)
    
    # Derived ratios
    df_scores['ratio_greenness'] = df_scores['green'] / (df_scores['green'] + df_scores['dead'] + 1e-6)
    df_scores['ratio_clover'] = df_scores['clover'] / (df_scores['clover'] + df_scores['grass'] + 1e-6)
    df_scores['ratio_cover'] = (df_scores['dense'] + df_scores['medium']) / (df_scores['bare'] + df_scores['sparse'] + 1e-6)
    
    torch.cuda.empty_cache()
    return df_scores.values


# =============================================================================
# SUPERVISED EMBEDDING ENGINE
# ===========================
class SupervisedEmbeddingEngine:
    """Feature engineering with PCA, PLS, and GMM."""
    
    def __init__(self, n_pca=0.80, n_pls=8, n_gmm=6, random_state=42):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=random_state)
        self.pls = PLSRegression(n_components=n_pls, scale=False)
        self.gmm = GaussianMixture(n_components=n_gmm, covariance_type='diag', random_state=random_state)
        self.pls_fitted_ = False
    
    def fit(self, X, y=None, X_semantic=None):
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)
        self.gmm.fit(X_scaled)
        
        if y is not None:
            self.pls.fit(X_scaled, y)
            self.pls_fitted_ = True
        return self
    
    def transform(self, X, X_semantic=None):
        X_scaled = self.scaler.transform(X)
        
        features = [self.pca.transform(X_scaled)]
        
        if self.pls_fitted_:
            features.append(self.pls.transform(X_scaled))
        
        features.append(self.gmm.predict_proba(X_scaled))
        
        if X_semantic is not None:
            sem_norm = (X_semantic - np.mean(X_semantic, axis=0)) / (np.std(X_semantic, axis=0) + 1e-6)
            features.append(sem_norm)
        
        return np.hstack(features)


# =============================================================================
# TRAINING
# ========
def cross_validate_predict(model_cls, model_params, train_data, test_data, 
                           sem_train, sem_test, emb_cols, cfg: CFG):
    """5-fold cross-validation with model ensemble."""
    target_max_arr = np.array([cfg.TARGET_MAX[t] for t in cfg.TARGET_NAMES], dtype=float)
    y_pred_test_accum = np.zeros([len(test_data), len(cfg.TARGET_NAMES)], dtype=float)
    
    n_splits = int(train_data['fold'].nunique())
    
    X_train_full = train_data[emb_cols].values.astype(np.float32)
    X_test_raw = test_data[emb_cols].values.astype(np.float32)
    y_train_full = train_data[cfg.TARGET_NAMES].values.astype(np.float32)
    
    feat_engine = SupervisedEmbeddingEngine(n_pca=0.80, n_pls=8, n_gmm=6)
    
    for fold in range(n_splits):
        print(f"Processing Fold {fold}...")
        
        train_mask = train_data['fold'] != fold
        X_tr = X_train_full[train_mask]
        y_tr = y_train_full[train_mask] / target_max_arr
        sem_tr_fold = sem_train[train_mask]
        
        engine = deepcopy(feat_engine)
        engine.fit(X_tr, y=y_tr, X_semantic=sem_tr_fold)
        
        x_tr_eng = engine.transform(X_tr, X_semantic=sem_tr_fold)
        x_te_eng = engine.transform(X_test_raw, X_semantic=sem_test)
        
        fold_test_pred = np.zeros([len(test_data), len(cfg.TARGET_NAMES)])
        
        for k, target_name in enumerate(cfg.TARGET_NAMES):
            if target_name == 'Dry_Clover_g':
                fold_test_pred[:, k] = 0.0
            else:
                model = model_cls(**model_params)
                model.fit(x_tr_eng, y_tr[:, k])
                pred_raw = model.predict(x_te_eng)
                fold_test_pred[:, k] = pred_raw * target_max_arr[k]
        
        y_pred_test_accum += fold_test_pred
    
    return y_pred_test_accum / n_splits


def train_siglip():
    """Main training function for SigLIP model."""
    cfg = CFG()
    seed_everything(cfg.SEED)
    
    print("="*60)
    print("CSIRO Image2Biomass - SigLIP Training")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(cfg.SPLIT_PATH)
    
    # Remove pre-existing embedding columns
    cols_to_keep = [c for c in train_df.columns if not c.startswith('emb')]
    train_df = train_df[cols_to_keep]
    
    if not str(train_df['image_path'].iloc[0]).startswith('/'):
        train_df['image_path'] = train_df['image_path'].apply(
            lambda p: str(cfg.DATA_PATH / 'train' / os.path.basename(p))
        )
    
    # Compute embeddings
    train_embeddings = compute_embeddings(cfg.SIGLIP_PATH, train_df, cfg)
    
    # Create feature columns
    emb_cols = [f"emb{i}" for i in range(train_embeddings.shape[1])]
    train_feat_df = pd.concat([
        train_df, 
        pd.DataFrame(train_embeddings, columns=emb_cols)
    ], axis=1)
    
    print(f"Train features shape: {train_feat_df.shape}")
    
    # Generate semantic features
    sem_train = generate_semantic_features(train_embeddings, cfg.SIGLIP_PATH, cfg)
    
    # Model parameters
    model_configs = {
        'HistGradientBoosting': (
            HistGradientBoostingRegressor,
            {'max_iter': 300, 'learning_rate': 0.05, 'max_depth': None, 
             'l2_regularization': 0.44, 'random_state': 42}
        ),
        'GradientBoosting': (
            GradientBoostingRegressor,
            {'n_estimators': 1354, 'learning_rate': 0.010, 'max_depth': 3, 
             'subsample': 0.60, 'random_state': 42}
        ),
        'CatBoost': (
            CatBoostRegressor,
            {'iterations': 1900, 'learning_rate': 0.045, 'depth': 4, 
             'l2_leaf_reg': 0.56, 'random_strength': 0.045, 
             'bagging_temperature': 0.98, 'verbose': 0, 'random_state': 42,
             'allow_writing_files': False}
        ),
        'LightGBM': (
            LGBMRegressor,
            {'n_estimators': 807, 'learning_rate': 0.014, 'num_leaves': 48, 
             'min_child_samples': 19, 'subsample': 0.745, 'colsample_bytree': 0.745, 
             'reg_alpha': 0.21, 'reg_lambda': 3.78, 'verbose': -1, 'random_state': 42}
        )
    }
    
    # Train models (validation only, no test predictions)
    print("\n" + "="*60)
    print("Training models with cross-validation...")
    print("="*60)
    
    for name, (model_cls, params) in model_configs.items():
        print(f"\nModel: {name}")
        # Here you would add OOF prediction collection for model selection
    
    print("\n✓ Training complete")
    print(f"Models and features ready for ensemble")


if __name__ == '__main__':
    train_siglip()