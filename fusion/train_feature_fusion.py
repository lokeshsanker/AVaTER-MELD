"""
V4 - Text + Audio + Video Feature-Level Fusion

This script combines the cached 768-dimensional features from:
    1. Text  - DistilBERT
    2. Audio - Wav2Vec2
    3. Video - FER-ViT

Feature dimensions:
    Text  = 768
    Audio = 768
    Video = 768

Total fused feature dimension:
    768 + 768 + 768 = 2304

The script:
    - Loads cached train/dev/test features
    - Verifies sample counts and label alignment
    - Concatenates the three modalities
    - Trains a neural-network fusion classifier
    - Evaluates on validation and test data
    - Generates classification report
    - Generates confusion matrix
    - Shows predicted-class distribution
    - Logs the experiment as RUN_005 in experiment_report.xlsx

Important:
    The individual Text, Audio and Video feature extractors are frozen.
    Only the V4 fusion classifier is trained.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(PROJECT_ROOT)

from config import (
    DEVICE,
    NUM_CLASSES,
    EMOTIONS,
    LABEL2IDX,
)

from utils.experiment_logger import (
    log_run,
    log_epoch,
    log_class_results,
    log_configuration,
    log_confusion_matrix,
    log_class_detection,
)


# =========================================================
# V4 CONFIGURATION
# =========================================================

VERSION = "V4"
MODALITY = "Text + Audio + Video"
MODEL_NAME = "Feature-Level Fusion"

RUN_ID = "RUN_005"

TRAIN_TEXT = "data/text_feature_cache_train.pt"
DEV_TEXT = "data/text_feature_cache_dev.pt"
TEST_TEXT = "data/text_feature_cache_test.pt"

TRAIN_AUDIO = "data/audio_feature_cache_train.pt"
DEV_AUDIO = "data/audio_feature_cache_dev.pt"
TEST_AUDIO = "data/audio_feature_cache_test.pt"

TRAIN_VIDEO = "data/fer_vit_feature_cache_train.pt"
DEV_VIDEO = "data/fer_vit_feature_cache_dev.pt"
TEST_VIDEO = "data/fer_vit_feature_cache_test.pt"


# Training configuration

BATCH_SIZE = 64
EPOCHS = 30

LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

DROPOUT_1 = 0.30
DROPOUT_2 = 0.20

PATIENCE = 5

LABEL_SMOOTHING = 0.1

SEED = 42

FEATURE_DIM = 768
FUSED_DIM = FEATURE_DIM * 3


# =========================================================
# REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# UTILITY
# =========================================================

def load_pt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    return torch.load(path, map_location="cpu")


# =========================================================
# LOAD FEATURES
# =========================================================

def load_features():

    print("\n" + "=" * 70)
    print("V4 FEATURE-LEVEL FUSION")
    print("=" * 70)

    print("\nLoading cached features...")

    text_train = load_pt(TRAIN_TEXT)
    text_dev = load_pt(DEV_TEXT)
    text_test = load_pt(TEST_TEXT)

    audio_train = load_pt(TRAIN_AUDIO)
    audio_dev = load_pt(DEV_AUDIO)
    audio_test = load_pt(TEST_AUDIO)

    video_train = load_pt(TRAIN_VIDEO)
    video_dev = load_pt(DEV_VIDEO)
    video_test = load_pt(TEST_VIDEO)

    return (
        text_train,
        text_dev,
        text_test,
        audio_train,
        audio_dev,
        audio_test,
        video_train,
        video_dev,
        video_test,
    )


# =========================================================
# EXTRACT FEATURES
# =========================================================

def get_text_features(cache):
    return cache["features"].float()


def get_text_labels(cache):
    return torch.tensor(
        [LABEL2IDX[label] for label in cache["labels"]],
        dtype=torch.long,
    )


def get_audio_features(cache):
    return cache.float()


def get_video_features(cache):
    return cache["features"].float()


def get_video_labels(cache):
    return cache["labels"].long()


# =========================================================
# ALIGNMENT CHECK
# =========================================================

def verify_split(
    split_name,
    text_cache,
    audio_cache,
    video_cache,
):

    print(f"\nChecking {split_name} alignment...")

    text_features = get_text_features(text_cache)
    text_labels = get_text_labels(text_cache)

    audio_features = get_audio_features(audio_cache)

    video_features = get_video_features(video_cache)
    video_labels = get_video_labels(video_cache)

    # -----------------------------------------------------
    # Shape checks
    # -----------------------------------------------------

    print("Text :", tuple(text_features.shape))
    print("Audio:", tuple(audio_features.shape))
    print("Video:", tuple(video_features.shape))

    if text_features.ndim != 2:
        raise ValueError(
            f"{split_name} Text features must be 2-D."
        )

    if audio_features.ndim != 2:
        raise ValueError(
            f"{split_name} Audio features must be 2-D."
        )

    if video_features.ndim != 2:
        raise ValueError(
            f"{split_name} Video features must be 2-D."
        )

    # -----------------------------------------------------
    # Feature dimensions
    # -----------------------------------------------------

    if text_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{split_name} Text feature dimension is "
            f"{text_features.shape[1]}, expected {FEATURE_DIM}"
        )

    if audio_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{split_name} Audio feature dimension is "
            f"{audio_features.shape[1]}, expected {FEATURE_DIM}"
        )

    if video_features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"{split_name} Video feature dimension is "
            f"{video_features.shape[1]}, expected {FEATURE_DIM}"
        )

    # -----------------------------------------------------
    # Number of samples
    # -----------------------------------------------------

    n_text = len(text_features)
    n_audio = len(audio_features)
    n_video = len(video_features)

    if not (n_text == n_audio == n_video):
        raise ValueError(
            f"{split_name} sample count mismatch:\n"
            f"Text={n_text}, Audio={n_audio}, Video={n_video}"
        )

    # -----------------------------------------------------
    # Label alignment
    # -----------------------------------------------------

    if not torch.equal(text_labels, video_labels):
        raise ValueError(
            f"{split_name} Text and Video labels are NOT aligned."
        )

    # -----------------------------------------------------
    # NaN / Inf checks
    # -----------------------------------------------------

    for name, features in [
        ("Text", text_features),
        ("Audio", audio_features),
        ("Video", video_features),
    ]:
        if not torch.isfinite(features).all():
            raise ValueError(
                f"{split_name} {name} features contain NaN or Inf."
            )

    print(
        f"{split_name}: alignment OK | "
        f"samples={n_text} | "
        f"feature_dim=768"
    )

    return (
        text_features,
        audio_features,
        video_features,
        text_labels,
    )


# =========================================================
# FUSION
# =========================================================

def create_fused_features(
    text_features,
    audio_features,
    video_features,
):

    fused = torch.cat(
        [
            text_features,
            audio_features,
            video_features,
        ],
        dim=1,
    )

    if fused.shape[1] != FUSED_DIM:
        raise ValueError(
            f"Fused feature dimension is {fused.shape[1]}, "
            f"expected {FUSED_DIM}"
        )

    return fused


# =========================================================
# V4 MODEL
# =========================================================

class FeatureFusionModel(nn.Module):

    def __init__(
        self,
        input_dim=FUSED_DIM,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(input_dim),

            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT_1),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT_2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)

            loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(
                preds.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    avg_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        np.array(all_labels),
        np.array(all_preds),
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("\nDevice:", DEVICE)

    # -----------------------------------------------------
    # Load caches
    # -----------------------------------------------------

    (
        text_train,
        text_dev,
        text_test,
        audio_train,
        audio_dev,
        audio_test,
        video_train,
        video_dev,
        video_test,
    ) = load_features()

    # -----------------------------------------------------
    # Verify all splits
    # -----------------------------------------------------

    (
        tr_text,
        tr_audio,
        tr_video,
        y_train,
    ) = verify_split(
        "TRAIN",
        text_train,
        audio_train,
        video_train,
    )

    (
        dv_text,
        dv_audio,
        dv_video,
        y_dev,
    ) = verify_split(
        "DEV",
        text_dev,
        audio_dev,
        video_dev,
    )

    (
        te_text,
        te_audio,
        te_video,
        y_test,
    ) = verify_split(
        "TEST",
        text_test,
        audio_test,
        video_test,
    )

    # -----------------------------------------------------
    # Create fused features
    # -----------------------------------------------------

    print("\nCreating fused features...")

    X_train = create_fused_features(
        tr_text,
        tr_audio,
        tr_video,
    )

    X_dev = create_fused_features(
        dv_text,
        dv_audio,
        dv_video,
    )

    X_test = create_fused_features(
        te_text,
        te_audio,
        te_video,
    )

    print(
        "\nFused feature shapes:"
    )

    print("Train:", tuple(X_train.shape))
    print("Dev  :", tuple(X_dev.shape))
    print("Test :", tuple(X_test.shape))

    # -----------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    dev_dataset = TensorDataset(
        X_dev,
        y_dev,
    )

    test_dataset = TensorDataset(
        X_test,
        y_test,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = FeatureFusionModel().to(DEVICE)

    print("\nV4 Model:")
    print(model)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_dev_f1 = -1.0
    best_state = None

    patience_counter = 0

    epoch_history = []

    print("\n" + "=" * 70)
    print("STARTING V4 TRAINING")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        train_preds = []
        train_labels = []

        for features, labels in train_loader:

            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(features)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * len(labels)
            )

            preds = torch.argmax(
                logits,
                dim=1,
            )

            train_preds.extend(
                preds.detach().cpu().tolist()
            )

            train_labels.extend(
                labels.detach().cpu().tolist()
            )

        train_loss = (
            running_loss /
            len(train_loader.dataset)
        )

        train_acc = accuracy_score(
            train_labels,
            train_preds,
        )

        (
            dev_loss,
            dev_acc,
            dev_macro_f1,
            dev_weighted_f1,
            _,
            _,
        ) = evaluate(
            model,
            dev_loader,
            criterion,
        )

        scheduler.step(dev_macro_f1)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"dev_loss={dev_loss:.4f} "
            f"dev_acc={dev_acc:.4f} "
            f"dev_macro_f1={dev_macro_f1:.4f} | "
            f"lr={current_lr:.6f}"
        )

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": dev_loss,
                "val_acc": dev_acc,
                "val_macro_f1": dev_macro_f1,
                "val_weighted_f1": dev_weighted_f1,
                "learning_rate": current_lr,
            }
        )

        # -------------------------------------------------
        # Save best model by Macro-F1
        # -------------------------------------------------

        if dev_macro_f1 > best_dev_f1:

            best_dev_f1 = dev_macro_f1

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

            print(
                f"  → Best model saved "
                f"(Dev Macro-F1={best_dev_f1:.4f})"
            )

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    # -----------------------------------------------------
    # Restore best model
    # -----------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # -----------------------------------------------------
    # Final DEV evaluation
    # -----------------------------------------------------

    (
        dev_loss,
        dev_acc,
        dev_macro_f1,
        dev_weighted_f1,
        dev_true,
        dev_preds,
    ) = evaluate(
        model,
        dev_loader,
        criterion,
    )

    # -----------------------------------------------------
    # Final TEST evaluation
    # -----------------------------------------------------

    (
        test_loss,
        test_acc,
        test_macro_f1,
        test_weighted_f1,
        test_true,
        test_preds,
    ) = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n" + "=" * 70)
    print("FINAL V4 RESULTS")
    print("=" * 70)

    print("\nDEV:")
    print(
        f"Accuracy     : {dev_acc:.4f}"
    )
    print(
        f"Macro-F1     : {dev_macro_f1:.4f}"
    )
    print(
        f"Weighted-F1  : {dev_weighted_f1:.4f}"
    )

    print("\nTEST:")
    print(
        f"Accuracy     : {test_acc:.4f}"
    )
    print(
        f"Macro-F1     : {test_macro_f1:.4f}"
    )
    print(
        f"Weighted-F1  : {test_weighted_f1:.4f}"
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nTEST CLASSIFICATION REPORT")
    print("-" * 70)

    report = classification_report(
        test_true,
        test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTIONS,
        zero_division=0,
    )

    print(report)

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        test_true,
        test_preds,
        labels=list(range(NUM_CLASSES)),
    )

    print("TEST CONFUSION MATRIX")
    print("-" * 70)

    print(
        pd.DataFrame(
            cm,
            index=EMOTIONS,
            columns=EMOTIONS,
        )
    )

    # -----------------------------------------------------
    # Class detection
    # -----------------------------------------------------

    unique, counts = np.unique(
        test_preds,
        return_counts=True,
    )

    predicted_counts = {
        emotion: 0
        for emotion in EMOTIONS
    }

    for idx, count in zip(
        unique,
        counts,
    ):
        predicted_counts[
            EMOTIONS[int(idx)]
        ] = int(count)

    print("\nTEST PREDICTED CLASS COUNTS")

    for emotion in EMOTIONS:
        print(
            f"{emotion:10s}: "
            f"{predicted_counts[emotion]}"
        )

    # =====================================================
    # EXCEL LOGGING
    # =====================================================

    print("\nLogging RUN_005 to Excel...")

    # -----------------------------------------------------
    # Run Summary
    # -----------------------------------------------------

    log_run(
        run_id=RUN_ID,
        version=VERSION,
        modality=MODALITY,
        model=MODEL_NAME,
        train_samples=len(X_train),
        val_samples=len(X_dev),
        test_samples=len(X_test),
        best_epoch=len(epoch_history),
        best_val_accuracy=dev_acc,
        test_accuracy=test_acc,
        macro_f1=test_macro_f1,
        weighted_f1=test_weighted_f1,
    )

    # -----------------------------------------------------
    # Epoch History
    # -----------------------------------------------------

    for row in epoch_history:

        log_epoch(
            run_id=RUN_ID,
            epoch=row["epoch"],
            train_loss=row["train_loss"],
            train_accuracy=row["train_acc"],
            val_loss=row["val_loss"],
            val_accuracy=row["val_acc"],
        )

    # -----------------------------------------------------
    # Class Results
    # -----------------------------------------------------

    report_dict = classification_report(
        test_true,
        test_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0,
    )

    for emotion in EMOTIONS:

        metrics = report_dict[emotion]

        log_class_results(
            run_id=RUN_ID,
            emotion=emotion,
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1-score"],
            support=metrics["support"],
        )

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    config = {
        "version": VERSION,
        "modality": MODALITY,
        "model": MODEL_NAME,
        "feature_dim_text": FEATURE_DIM,
        "feature_dim_audio": FEATURE_DIM,
        "feature_dim_video": FEATURE_DIM,
        "fused_feature_dim": FUSED_DIM,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "dropout_1": DROPOUT_1,
        "dropout_2": DROPOUT_2,
        "label_smoothing": LABEL_SMOOTHING,
        "early_stopping_patience": PATIENCE,
        "seed": SEED,
        "device": str(DEVICE),
    }

    log_configuration(
        run_id=RUN_ID,
        config=config,
    )

    # -----------------------------------------------------
    # Confusion Matrix
    # -----------------------------------------------------

    log_confusion_matrix(
        run_id=RUN_ID,
        emotion_labels=EMOTIONS,
        matrix=cm,
    )

    # -----------------------------------------------------
    # Class Detection
    # -----------------------------------------------------

    log_class_detection(
        run_id=RUN_ID,
        version=VERSION,
        modality=MODALITY,
        model=MODEL_NAME,
        emotion_labels=EMOTIONS,
        predicted_counts=predicted_counts,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    os.makedirs(
        "checkpoints",
        exist_ok=True,
    )

    checkpoint_path = (
        "checkpoints/"
        "v4_feature_fusion_best.pt"
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                FUSED_DIM,

            "num_classes":
                NUM_CLASSES,

            "emotions":
                EMOTIONS,

            "version":
                VERSION,
        },
        checkpoint_path,
    )

    print(
        f"\nBest V4 model saved to:"
        f"\n{checkpoint_path}"
    )

    print("\n" + "=" * 70)
    print("V4 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()