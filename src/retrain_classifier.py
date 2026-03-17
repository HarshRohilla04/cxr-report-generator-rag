"""
Retrain Multi-Label Classifier on BiomedCLIP Embeddings

Fixes the broken classifier by training with:
- Proper ground truth CheXpert labels (not placeholder)
- Class-balanced BCEWithLogitsLoss (pos_weight)
- Early stopping on top-K frequent disease AUC (not global mean)
- Temperature scaling calibration post-training

Usage:
    .venv/Scripts/python.exe retrain_classifier.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path
import json
import time

# ============================================================================
# CONFIG
# ============================================================================

EMBEDDING_PATH = "embeddings/mimic_image_embeddings.pt"
MASTER_CSV = "image-data/processed/mimic_master.csv"
TRAIN_CSV = "image-data/processed/train.csv"
VAL_CSV = "image-data/processed/val.csv"
TEST_CSV = "image-data/processed/test.csv"

OUTPUT_PATH = "models/multilabel_classifier_biomedclip_v2.pt"

CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]

# Top frequent diseases for early stopping (exclude rare classes)
# Excludes: Pleural Other (0.6%), Fracture (1.2%), Enlarged Cardiomediastinum (1.9%)
EARLY_STOP_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Lung Lesion', 'Lung Opacity', 'Pleural Effusion',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]
EARLY_STOP_INDICES = [CHEXPERT_LABELS.index(l) for l in EARLY_STOP_LABELS]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 256
NUM_EPOCHS = 50
LR = 1e-4
PATIENCE = 8  # Early stopping patience

# ============================================================================
# MODEL (same architecture as existing)
# ============================================================================

class MultiLabelClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """Load embeddings and labels, aligned by image_path order in master CSV."""
    print("Loading embeddings...")
    all_embeddings = torch.load(EMBEDDING_PATH, map_location='cpu')
    print(f"  Embeddings shape: {all_embeddings.shape}")
    
    print("Loading CSVs...")
    master_df = pd.read_csv(MASTER_CSV)
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    # Build image_path -> embedding index mapping from master
    path_to_idx = {path: idx for idx, path in enumerate(master_df['image_path'])}
    
    def extract_split(df, name):
        indices = []
        labels_list = []
        skipped = 0
        for _, row in df.iterrows():
            path = row['image_path']
            if path in path_to_idx:
                indices.append(path_to_idx[path])
                label_vec = [float(row[col]) if pd.notnull(row[col]) else 0.0 for col in CHEXPERT_LABELS]
                labels_list.append(label_vec)
            else:
                skipped += 1
        
        X = all_embeddings[indices]
        y = torch.tensor(labels_list, dtype=torch.float32)
        print(f"  {name}: {len(indices)} samples loaded ({skipped} skipped)")
        return X, y
    
    X_train, y_train = extract_split(train_df, "Train")
    X_val, y_val = extract_split(val_df, "Val")
    X_test, y_test = extract_split(test_df, "Test")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# ============================================================================
# TRAINING
# ============================================================================

def compute_pos_weight(y_train):
    """Compute per-class pos_weight = num_neg / num_pos."""
    num_pos = y_train.sum(dim=0)
    num_neg = y_train.shape[0] - num_pos
    # Clamp to avoid division by zero for any class with 0 positives
    pos_weight = num_neg / num_pos.clamp(min=1.0)
    return pos_weight

def compute_auc_filtered(y_true, y_pred, label_indices):
    """Compute mean AUC-ROC over specified label indices only."""
    aucs = []
    for i in label_indices:
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        # Skip if only one class present in ground truth
        if col_true.sum() == 0 or col_true.sum() == len(col_true):
            continue
        try:
            auc = roc_auc_score(col_true, col_pred)
            aucs.append(auc)
        except ValueError:
            continue
    return np.mean(aucs) if aucs else 0.0

def compute_all_aucs(y_true, y_pred):
    """Compute per-class AUC-ROC."""
    aucs = {}
    for i, label in enumerate(CHEXPERT_LABELS):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        if col_true.sum() == 0 or col_true.sum() == len(col_true):
            aucs[label] = float('nan')
            continue
        try:
            aucs[label] = roc_auc_score(col_true, col_pred)
        except ValueError:
            aucs[label] = float('nan')
    return aucs

def train():
    print("=" * 60)
    print("RETRAINING MULTI-LABEL CLASSIFIER")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Compute class weights
    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    print(f"\npos_weight per class:")
    for label, w in zip(CHEXPERT_LABELS, pos_weight):
        print(f"  {label:30s}: {w:.2f}")
    
    # Create DataLoaders
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MultiLabelClassifier(input_dim=X_train.shape[1], num_classes=len(CHEXPERT_LABELS))
    model.to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None
    
    print(f"\nTraining for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...")
    print(f"Early stopping on mean AUC of: {EARLY_STOP_LABELS}")
    print()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_ds)
        
        # Validate
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_labels.append(y_batch.cpu().numpy())
        
        val_loss /= len(val_ds)
        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        
        # Compute AUC on early-stop subset
        val_auc = compute_auc_filtered(val_labels, val_preds, EARLY_STOP_INDICES)
        # Also compute full AUC for reporting
        val_auc_full = compute_auc_filtered(val_labels, val_preds, list(range(len(CHEXPERT_LABELS))))
        
        scheduler.step(val_auc)
        
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"val_AUC(top-k): {val_auc:.4f} | val_AUC(all): {val_auc_full:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s")
        
        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                break
    
    print(f"\n Best val AUC (top-k): {best_auc:.4f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    # ========================================================================
    # TEMPERATURE SCALING CALIBRATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEMPERATURE SCALING CALIBRATION")
    print("=" * 60)
    
    temperature = calibrate_temperature(model, val_loader, y_val)
    
    # ========================================================================
    # TEST EVALUATION
    # ========================================================================
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    test_ds = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model.eval()
    test_preds = []
    test_logits_all = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            test_logits_all.append(logits.cpu())
            # Apply temperature scaling
            probs = torch.sigmoid(logits / temperature)
            test_preds.append(probs.cpu().numpy())
    
    test_preds = np.concatenate(test_preds)
    test_labels = y_test.numpy()
    
    # Per-class AUC
    print(f"\nPer-class AUC-ROC (with T={temperature:.3f}):")
    aucs = compute_all_aucs(test_labels, test_preds)
    for label, auc in aucs.items():
        marker = "" if auc > 0.70 else ""
        print(f"  {marker} {label:30s}: {auc:.4f}")
    
    mean_auc = np.nanmean(list(aucs.values()))
    print(f"\n  Mean AUC: {mean_auc:.4f}")
    
    # Discrimination check
    print("\nDiscrimination Check:")
    # Normal images (No Finding = 1)
    normal_mask = test_labels[:, CHEXPERT_LABELS.index('No Finding')] == 1
    abnormal_mask = ~normal_mask
    
    if normal_mask.sum() > 0:
        normal_nofinding = test_preds[normal_mask, CHEXPERT_LABELS.index('No Finding')]
        normal_max_disease = np.max(test_preds[normal_mask, :13], axis=1)  # Exclude No Finding
        print(f"  Normal images (N={normal_mask.sum()}):")
        print(f"    Avg 'No Finding' prob: {normal_nofinding.mean():.3f} (target: >0.7)")
        print(f"    Avg max disease prob:  {normal_max_disease.mean():.3f} (target: <0.4)")
    
    if abnormal_mask.sum() > 0:
        abnormal_nofinding = test_preds[abnormal_mask, CHEXPERT_LABELS.index('No Finding')]
        abnormal_max_disease = np.max(test_preds[abnormal_mask, :13], axis=1)
        print(f"  Abnormal images (N={abnormal_mask.sum()}):")
        print(f"    Avg 'No Finding' prob: {abnormal_nofinding.mean():.3f} (target: <0.4)")
        print(f"    Avg max disease prob:  {abnormal_max_disease.mean():.3f} (target: >0.5)")
    
    # ========================================================================
    # SAVE
    # ========================================================================
    checkpoint = {
        'model_state_dict': best_state,
        'embedding_dim': X_train.shape[1],
        'num_classes': len(CHEXPERT_LABELS),
        'label_names': CHEXPERT_LABELS,
        'temperature': temperature,
        'best_epoch': best_epoch,
        'best_val_auc': best_auc,
        'test_aucs': aucs,
        'test_mean_auc': mean_auc,
    }
    
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_PATH)
    print(f"\n Model saved to {OUTPUT_PATH}")
    print(f"   Temperature: {temperature:.4f}")
    print(f"   Best val AUC (top-k): {best_auc:.4f}")
    print(f"   Test mean AUC: {mean_auc:.4f}")

# ============================================================================
# TEMPERATURE SCALING
# ============================================================================

def calibrate_temperature(model, val_loader, y_val, lr=0.01, max_iter=100):
    """
    Learn a single temperature scalar T on the val set.
    Minimizes NLL: BCEWithLogitsLoss(logits / T, labels) w.r.t. T.
    """
    model.eval()
    
    # Collect all val logits
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch)
    
    all_logits = torch.cat(all_logits)  # [N, 14]
    all_labels = torch.cat(all_labels)  # [N, 14]
    
    # Optimize T
    log_temperature = nn.Parameter(torch.zeros(1))  # T = exp(0) = 1.0 initially
    optimizer = optim.LBFGS([log_temperature], lr=lr, max_iter=max_iter)
    
    criterion = nn.BCEWithLogitsLoss()
    
    def closure():
        optimizer.zero_grad()
        T = torch.exp(log_temperature)
        scaled_logits = all_logits / T
        loss = criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(closure)
    
    T = torch.exp(log_temperature).item()
    
    # Compute NLL before and after
    with torch.no_grad():
        nll_before = nn.BCEWithLogitsLoss()(all_logits, all_labels).item()
        nll_after = nn.BCEWithLogitsLoss()(all_logits / T, all_labels).item()
    
    print(f"  Temperature: {T:.4f}")
    print(f"  NLL before: {nll_before:.4f}")
    print(f"  NLL after:  {nll_after:.4f}")
    
    return T

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    train()
