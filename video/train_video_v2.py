"""
VIDEO V2 TRAINING
=================

Cached MobileNetV2 frame features
        ↓
BiGRU
        ↓
Attention Pooling
        ↓
7-class classifier

Input cache:
    data/video_v2_feature_cache_train.pt
    data/video_v2_feature_cache_dev.pt

Best checkpoint selected using Validation Macro-F1.
"""

import os
import sys
import random
import numpy as np

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from config import DEVICE, NUM_CLASSES, LABEL2IDX
from video.model import AttentionPool


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TRAIN_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "video_v2_feature_cache_train.pt"
)

DEV_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "video_v2_feature_cache_dev.pt"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "video_v2"
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "video_v2_best.pt"
)

BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

GRU_HIDDEN = 256
DROPOUT = 0.30

GRAD_CLIP = 1.0

SEED = 42


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class VideoV2Classifier(nn.Module):
    """
    Cached frame sequence:
        [B, 4, 1280]

    BiGRU:
        [B, 4, 512]

    Attention:
        [B, 512]

    Classifier:
        [B, 7]
    """

    def __init__(
        self,
        input_dim=1280,
        hidden_dim=256,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.pool = AttentionPool(
            hidden_dim=2 * hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim),

            nn.Linear(
                2 * hidden_dim,
                128
            ),

            nn.GELU(),

            nn.Dropout(DROPOUT),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        # x:
        # [B, 4, 1280]

        sequence, _ = self.gru(x)

        # sequence:
        # [B, 4, 512]

        pooled, attention_weights = self.pool(
            sequence
        )

        # pooled:
        # [B, 512]

        logits = self.classifier(
            pooled
        )

        return logits, attention_weights


# ---------------------------------------------------------
# Load cache
# ---------------------------------------------------------

def load_cache(path, name):

    print(
        f"\nLoading {name} cache:"
    )

    print(path)

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"]
    labels = data["labels"]

    print(
        f"Features: {features.shape}"
    )

    print(
        f"Labels:   {labels.shape}"
    )

    return features.float(), labels.long()


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(
    labels,
    predictions
):

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return (
        accuracy,
        macro_f1,
        weighted_f1
    )


# ---------------------------------------------------------
# Run one epoch
# ---------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None
):

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for features, labels in loader:

        features = features.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if is_training:
            optimizer.zero_grad(
                set_to_none=True
            )

        # -------------------------------------------------
        # Forward
        # -------------------------------------------------

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            logits, _ = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )

        # -------------------------------------------------
        # Backward
        # -------------------------------------------------

        if is_training:

            if scaler is not None:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP
                )

                optimizer.step()

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    avg_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy, macro_f1, weighted_f1 = calculate_metrics(
        all_labels,
        all_predictions
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# ---------------------------------------------------------
# Class weights
# ---------------------------------------------------------

def create_class_weights(labels):

    counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES
    ).float()

    print("\nTraining class distribution:")

    for label, index in LABEL2IDX.items():

        print(
            f"  {label:10s}: "
            f"{int(counts[index].item())}"
        )

    # Balanced inverse-frequency weights
    weights = len(labels) / (
        NUM_CLASSES * counts
    )

    print("\nClass weights:")

    for label, index in LABEL2IDX.items():

        print(
            f"  {label:10s}: "
            f"{weights[index].item():.4f}"
        )

    return weights.to(DEVICE)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("VIDEO V2 TRAINING")
    print("=" * 70)

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

    # -----------------------------------------------------
    # Load cached features
    # -----------------------------------------------------

    train_features, train_labels = load_cache(
        TRAIN_CACHE,
        "TRAIN"
    )

    dev_features, dev_labels = load_cache(
        DEV_CACHE,
        "DEV"
    )

    # -----------------------------------------------------
    # Verify shapes
    # -----------------------------------------------------

    assert train_features.shape == (
        9988,
        4,
        1280
    )

    assert dev_features.shape == (
        1108,
        4,
        1280
    )

    assert train_labels.shape[0] == 9988
    assert dev_labels.shape[0] == 1108

    print("\nCache shape verification:")
    print("  TRAIN: PASS")
    print("  DEV:   PASS")

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    dev_dataset = TensorDataset(
        dev_features,
        dev_labels
    )

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"\nBatch size: {BATCH_SIZE}"
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = VideoV2Classifier(
        input_dim=1280,
        hidden_dim=GRU_HIDDEN,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nModel:")
    print(
        "  MobileNetV2 features: 1280-D"
    )
    print(
        f"  Sequence length:       4"
    )
    print(
        f"  BiGRU hidden:           {GRU_HIDDEN}"
    )
    print(
        "  Bidirectional:          True"
    )
    print(
        "  Attention pooling:      True"
    )
    print(
        f"  Total parameters:       {total_params:,}"
    )
    print(
        f"  Trainable parameters:   {trainable_params:,}"
    )

    # -----------------------------------------------------
    # Class-weighted loss
    # -----------------------------------------------------

    class_weights = create_class_weights(
        train_labels
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    # -----------------------------------------------------
    # AMP
    # -----------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    print("\nTraining configuration:")
    print(
        f"  Max epochs:       {MAX_EPOCHS}"
    )
    print(
        f"  Learning rate:    {LEARNING_RATE}"
    )
    print(
        f"  Weight decay:     {WEIGHT_DECAY}"
    )
    print(
        f"  Label smoothing:  0.05"
    )
    print(
        f"  Early stopping:   {PATIENCE}"
    )
    print(
        f"  AMP:              {torch.cuda.is_available()}"
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\n" + "=" * 70)
    print("STARTING VIDEO V2 TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        train_loss, train_acc, train_macro, train_weighted = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scaler=scaler
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        dev_loss, dev_acc, dev_macro, dev_weighted = run_epoch(
            model,
            dev_loader,
            criterion,
            optimizer=None,
            scaler=None
        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(
            dev_macro
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Train Macro-F1: {train_macro:.4f} | "
            f"Train Weighted-F1: {train_weighted:.4f}"
        )

        print(
            f"Dev Loss:   {dev_loss:.4f} | "
            f"Dev Acc:   {dev_acc:.4f} | "
            f"Dev Macro-F1: {dev_macro:.4f} | "
            f"Dev Weighted-F1: {dev_weighted:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.7f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_macro,
            "train_weighted_f1": train_weighted,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_macro,
            "dev_weighted_f1": dev_weighted,
            "learning_rate": current_lr,
        })

        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        if dev_macro > best_macro_f1:

            best_macro_f1 = dev_macro
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "dev_macro_f1": dev_macro,
                "dev_accuracy": dev_acc,
                "dev_weighted_f1": dev_weighted,
                "config": {
                    "input_dim": 1280,
                    "num_frames": 4,
                    "gru_hidden": GRU_HIDDEN,
                    "num_classes": NUM_CLASSES,
                }
            }

            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )

            print(
                f"  ✓ BEST MODEL SAVED "
                f"(Macro-F1: {dev_macro:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"  No Macro-F1 improvement "
                f"(patience: "
                f"{patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break

    # -----------------------------------------------------
    # Final training summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO V2 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best Epoch:     {best_epoch}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print("\nTraining history:")

    for item in history:

        print(
            f"Epoch {item['epoch']:02d}: "
            f"Dev Acc={item['dev_accuracy']:.4f}, "
            f"Dev Macro-F1={item['dev_macro_f1']:.4f}, "
            f"Dev Weighted-F1={item['dev_weighted_f1']:.4f}"
        )


if __name__ == "__main__":
    main()