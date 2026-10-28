"""
VIDEO V3 - BALANCED TRAINING FROM CACHED FEATURES
=================================================

Input:
    data/video_v2_feature_cache_train.pt
    data/video_v2_feature_cache_dev.pt

Feature shape:
    [N, 4, 1280]

Pipeline:
    MobileNetV2 cached frame features
            ↓
          BiGRU
            ↓
      Attention Pooling
            ↓
       Classifier
            ↓
       7 emotions

V3 changes:
    - Moderate class weighting using inverse sqrt frequency
    - No label smoothing
    - Lower learning rate
    - Macro-F1 checkpoint selection
    - Gradient clipping
    - AMP
    - Early stopping
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# PROJECT IMPORTS
# =========================================================

from config import (
    DEVICE,
    NUM_CLASSES,
    LABEL2IDX
)


# =========================================================
# PATHS
# =========================================================

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
    "video_v3"
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "video_v3_best.pt"
)


# =========================================================
# TRAINING CONFIGURATION
# =========================================================

BATCH_SIZE = 64

MAX_EPOCHS = 30

EARLY_STOP_PATIENCE = 6

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

GRU_HIDDEN = 256

DROPOUT = 0.30

GRAD_CLIP = 1.0

SEED = 42


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =========================================================
# MODEL
# =========================================================

class VideoV3Classifier(nn.Module):

    def __init__(
        self,
        input_dim=1280,
        hidden_dim=256,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        # -------------------------------------------------
        # BiGRU
        # -------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # -------------------------------------------------
        # Attention
        # -------------------------------------------------

        self.attention = nn.Linear(
            hidden_dim * 2,
            1
        )

        # -------------------------------------------------
        # Classifier
        # -------------------------------------------------

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                hidden_dim * 2
            ),

            nn.Linear(
                hidden_dim * 2,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

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

        scores = self.attention(
            sequence
        ).squeeze(-1)

        # [B, 4]

        weights = torch.softmax(
            scores,
            dim=1
        )

        # [B, 4]

        pooled = torch.bmm(
            weights.unsqueeze(1),
            sequence
        ).squeeze(1)

        # [B, 512]

        logits = self.classifier(
            pooled
        )

        return logits, weights


# =========================================================
# LOAD CACHE
# =========================================================

def load_cache(
    path,
    name
):

    print(
        f"\nLoading {name} cache:"
    )

    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Cache not found: {path}"
        )

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    features = data["features"].float()

    labels = data["labels"].long()

    print(
        f"Features: {features.shape}"
    )

    print(
        f"Labels:   {labels.shape}"
    )

    return features, labels


# =========================================================
# MODERATE CLASS WEIGHTS
# =========================================================

def create_balanced_weights(
    labels
):

    counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES
    ).float()

    print(
        "\nTraining class distribution:"
    )

    for label, index in LABEL2IDX.items():

        print(
            f"  {label:10s}: "
            f"{int(counts[index].item())}"
        )

    # -----------------------------------------------------
    # Moderate balancing
    #
    # Instead of:
    #
    #     1 / frequency
    #
    # use:
    #
    #     1 / sqrt(frequency)
    #
    # This reduces the extreme influence of rare classes.
    # -----------------------------------------------------

    weights = 1.0 / torch.sqrt(
        counts
    )

    # Normalize weights so average = 1
    weights = weights / weights.mean()

    print(
        "\nModerate class weights:"
    )

    for label, index in LABEL2IDX.items():

        print(
            f"  {label:10s}: "
            f"{weights[index].item():.4f}"
        )

    return weights.to(DEVICE)


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# ONE EPOCH
# =========================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None
):

    training = optimizer is not None

    if training:

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


        if training:

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

        if training:

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


        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

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
        /
        len(loader.dataset)
    )


    accuracy, macro_f1, weighted_f1 = (
        calculate_metrics(
            all_labels,
            all_predictions
        )
    )


    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# =========================================================
# VALIDATION CLASS DETECTION
# =========================================================

@torch.no_grad()
def get_predictions(
    model,
    loader
):

    model.eval()

    all_labels = []

    all_predictions = []


    for features, labels in loader:

        features = features.to(
            DEVICE
        )

        logits, _ = model(
            features
        )

        predictions = logits.argmax(
            dim=1
        )

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )


    return (
        np.array(all_labels),
        np.array(all_predictions)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )


    print(
        "=" * 70
    )

    print(
        "VIDEO V3 - BALANCED CACHED FEATURE TRAINING"
    )

    print(
        "=" * 70
    )


    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

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
    # Load caches
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
    # Shape checks
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


    print(
        "\nCache verification:"
    )

    print(
        "  TRAIN: PASS"
    )

    print(
        "  DEV:   PASS"
    )


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

    model = VideoV3Classifier(
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


    print(
        "\nModel:"
    )

    print(
        "  Input:              4 × 1280"
    )

    print(
        "  BiGRU hidden:       256"
    )

    print(
        "  Bidirectional:      True"
    )

    print(
        "  Attention:          True"
    )

    print(
        f"  Total parameters:   {total_params:,}"
    )

    print(
        f"  Trainable params:   {trainable_params:,}"
    )


    # -----------------------------------------------------
    # Moderate class weights
    # -----------------------------------------------------

    class_weights = create_balanced_weights(
        train_labels
    )


    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # -----------------------------------------------------
    # Scheduler
    # -----------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )
    )


    # -----------------------------------------------------
    # AMP
    # -----------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )


    print(
        "\nTraining configuration:"
    )

    print(
        f"  Max epochs:        {MAX_EPOCHS}"
    )

    print(
        f"  Learning rate:     {LEARNING_RATE}"
    )

    print(
        f"  Weight decay:      {WEIGHT_DECAY}"
    )

    print(
        "  Label smoothing:   0.0"
    )

    print(
        "  Class weighting:   inverse sqrt frequency"
    )

    print(
        f"  Early stopping:    {EARLY_STOP_PATIENCE}"
    )

    print(
        f"  AMP:               {torch.cuda.is_available()}"
    )


    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    best_macro_f1 = -1.0

    best_epoch = 0

    patience_counter = 0


    history = []


    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STARTING VIDEO V3 TRAINING"
    )

    print(
        "=" * 70
    )


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

        train_loss, train_acc, train_macro, train_weighted = (
            run_epoch(
                model,
                train_loader,
                criterion,
                optimizer=optimizer,
                scaler=scaler
            )
        )


        # -------------------------------------------------
        # Dev
        # -------------------------------------------------

        dev_loss, dev_acc, dev_macro, dev_weighted = (
            run_epoch(
                model,
                dev_loader,
                criterion,
                optimizer=None,
                scaler=None
            )
        )


        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        scheduler.step(
            dev_macro
        )


        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # -------------------------------------------------
        # Print results
        # -------------------------------------------------

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
            "learning_rate": current_lr
        })


        # -------------------------------------------------
        # Best checkpoint
        # -------------------------------------------------

        if dev_macro > best_macro_f1:

            best_macro_f1 = dev_macro

            best_epoch = epoch

            patience_counter = 0


            torch.save(
                {
                    "model_state_dict": model.state_dict(),

                    "epoch": epoch,

                    "dev_accuracy": dev_acc,

                    "dev_macro_f1": dev_macro,

                    "dev_weighted_f1": dev_weighted,

                    "config": {
                        "input_dim": 1280,
                        "num_frames": 4,
                        "gru_hidden": GRU_HIDDEN,
                        "num_classes": NUM_CLASSES,
                        "learning_rate": LEARNING_RATE,
                        "weighting": "inverse_sqrt_frequency"
                    }
                },
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
                f"{patience_counter}/"
                f"{EARLY_STOP_PATIENCE})"
            )


            if patience_counter >= EARLY_STOP_PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "VIDEO V3 TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )


    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # =====================================================
    # FINAL DEV PREDICTIONS
    # =====================================================

    labels, predictions = get_predictions(
        model,
        dev_loader
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        "BEST MODEL DEV CLASSIFICATION"
    )

    print(
        "=" * 70
    )


    target_names = [
        label
        for label, index
        in sorted(
            LABEL2IDX.items(),
            key=lambda x: x[1]
        )
    ]


    print(
        classification_report(
            labels,
            predictions,
            labels=list(range(NUM_CLASSES)),
            target_names=target_names,
            zero_division=0,
            digits=4
        )
    )


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    print(
        "Class detection:"
    )


    detected = 0


    for index, label in enumerate(
        target_names
    ):

        count = int(
            np.sum(
                predictions == index
            )
        )


        if count > 0:

            detected += 1


        print(
            f"  {label:10s}: {count}"
        )


    print(
        f"\nDetected classes: "
        f"{detected}/{NUM_CLASSES}"
    )


    print(
        "\nTraining history:"
    )


    for item in history:

        print(
            f"Epoch {item['epoch']:02d}: "
            f"Dev Acc={item['dev_accuracy']:.4f}, "
            f"Dev Macro-F1={item['dev_macro_f1']:.4f}, "
            f"Dev Weighted-F1={item['dev_weighted_f1']:.4f}"
        )


if __name__ == "__main__":

    main()