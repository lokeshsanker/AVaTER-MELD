"""
Balanced Text Branch Training
DistilBERT + class-weighted CrossEntropyLoss

Goal:
    Improve detection across all 7 MELD emotion classes.

This script does NOT overwrite the original Text checkpoint.

Usage:
    python text/train_text_balanced.py --manifest data/train.csv --val_manifest data/val.csv
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import (
    DEVICE,
    TEXT_BATCH_SIZE,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    LABEL2IDX,
)

from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


# ============================================================
# CONFIGURATION
# ============================================================

BALANCED_CKPT_DIR = os.path.join("checkpoints", "text_balanced")
BALANCED_CKPT_FILE = os.path.join(
    BALANCED_CKPT_DIR,
    "pytorch_model.bin"
)

EMOTION_LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

NUM_CLASSES = len(EMOTION_LABELS)

# Separate learning rate for this experiment.
# We keep it explicit instead of changing config.py.
BALANCED_LR = LR

# Small weight cap to avoid extremely large gradients
MAX_CLASS_WEIGHT = 5.0


# ============================================================
# CLASS WEIGHTS
# ============================================================

def compute_class_weights(train_df):
    """
    Compute balanced class weights from TRAINING data only.

    Formula:
        weight_i = total_samples / (num_classes * class_count_i)

    The weights are then normalized so the average weight is 1.
    """

    counts = np.zeros(NUM_CLASSES, dtype=np.float64)

    for label in train_df["label"]:
        if label not in LABEL2IDX:
            raise ValueError(
                f"Unknown label found in training data: {label}"
            )

        class_idx = LABEL2IDX[label]
        counts[class_idx] += 1

    total = counts.sum()

    if total == 0:
        raise ValueError("Training dataset is empty.")

    weights = total / (NUM_CLASSES * counts)

    # Normalize average weight to 1
    weights = weights / weights.mean()

    # Prevent extreme weighting
    weights = np.minimum(weights, MAX_CLASS_WEIGHT)

    # Re-normalize after clipping
    weights = weights / weights.mean()

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )

    print("\n============================================================")
    print("CLASS BALANCING")
    print("============================================================")

    for i, label in enumerate(EMOTION_LABELS):
        print(
            f"{label:<10}: "
            f"count={int(counts[i]):5d} | "
            f"weight={weights[i]:.4f}"
        )

    print("============================================================\n")

    return weights_tensor


# ============================================================
# EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_predictions = []

    context = (
        torch.enable_grad()
        if is_train
        else torch.no_grad()
    )

    with context:

        for (
            input_ids,
            attention_mask,
            labels,
            _
        ) in tqdm(loader, leave=False):

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(
                input_ids,
                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            if is_train:

                optimizer.zero_grad()

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            predictions = logits.argmax(dim=-1)

            total_loss += (
                loss.item() * labels.size(0)
            )

            total += labels.size(0)

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

            all_predictions.extend(
                predictions.detach().cpu().numpy()
            )

    avg_loss = total_loss / total

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_predictions,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--val_manifest",
        required=True
    )

    args = parser.parse_args()

    print("\n============================================================")
    print("BALANCED TEXT MODEL TRAINING")
    print("============================================================")

    print(f"Device : {DEVICE}")
    print(f"Learning rate : {BALANCED_LR}")
    print(f"Batch size : {TEXT_BATCH_SIZE}")
    print(f"Max epochs : {EPOCHS}")
    print(f"Early stopping patience : {EARLY_STOP_PATIENCE}")

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading manifests...")

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    print(f"Training samples   : {len(train_df)}")
    print(f"Validation samples : {len(val_df)}")

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_dataset = TextEmotionDataset(train_df)
    val_dataset = TextEmotionDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    class_weights = compute_class_weights(
        train_df
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("Loading DistilBERT model...")

    model = TextEmotionModel().to(DEVICE)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BALANCED_LR
    )

    # --------------------------------------------------------
    # TRAINING STATE
    # --------------------------------------------------------

    best_macro_f1 = -1.0
    patience_left = EARLY_STOP_PATIENCE

    os.makedirs(
        BALANCED_CKPT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        print(
            f"\n---------------- Epoch {epoch:02d} ----------------"
        )

        (
            train_loss,
            train_acc,
            train_macro_f1,
            train_weighted_f1,
            _,
            _
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            val_loss,
            val_acc,
            val_macro_f1,
            val_weighted_f1,
            val_labels,
            val_predictions
        ) = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train | "
            f"loss={train_loss:.4f} "
            f"acc={train_acc:.4f} "
            f"macro_f1={train_macro_f1:.4f} "
            f"weighted_f1={train_weighted_f1:.4f}"
        )

        print(
            f"Val   | "
            f"loss={val_loss:.4f} "
            f"acc={val_acc:.4f} "
            f"macro_f1={val_macro_f1:.4f} "
            f"weighted_f1={val_weighted_f1:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST BY MACRO-F1
        # ----------------------------------------------------

        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1
            patience_left = EARLY_STOP_PATIENCE

            torch.save(
                model.state_dict(),
                BALANCED_CKPT_FILE
            )

            print(
                f"  -> NEW BEST MACRO-F1: "
                f"{best_macro_f1:.4f}"
            )

            print(
                f"  -> checkpoint saved to: "
                f"{BALANCED_CKPT_FILE}"
            )

        else:

            patience_left -= 1

            print(
                f"  -> no improvement | "
                f"patience remaining: "
                f"{patience_left}"
            )

            if patience_left <= 0:

                print(
                    "\nEarly stopping."
                )

                break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n============================================================")
    print("LOADING BEST BALANCED TEXT MODEL")
    print("============================================================")

    model.load_state_dict(
        torch.load(
            BALANCED_CKPT_FILE,
            map_location=DEVICE
        )
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    (
        final_loss,
        final_accuracy,
        final_macro_f1,
        final_weighted_f1,
        labels,
        predictions
    ) = run_epoch(
        model,
        val_loader,
        criterion
    )

    print("\n============================================================")
    print("BALANCED TEXT MODEL - FINAL DEV RESULTS")
    print("============================================================")

    print(
        f"Accuracy    : "
        f"{final_accuracy:.4f} "
        f"({final_accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1    : "
        f"{final_macro_f1:.4f} "
        f"({final_macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1 : "
        f"{final_weighted_f1:.4f} "
        f"({final_weighted_f1 * 100:.2f}%)"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:")

    report = classification_report(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_LABELS,
        zero_division=0
    )

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(NUM_CLASSES))
    )

    print("Confusion Matrix:")
    print(
        "               "
        + " ".join(
            f"{label:>9}"
            for label in EMOTION_LABELS
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{EMOTION_LABELS[i]:<12}"
            + " ".join(
                f"{value:9d}"
                for value in row
            )
        )

    # ========================================================
    # CLASS DETECTION
    # ========================================================

    print("\n============================================================")
    print("7-CLASS DETECTION")
    print("============================================================")

    detected_count = 0

    for i, label in enumerate(EMOTION_LABELS):

        count = int(
            (np.array(predictions) == i).sum()
        )

        if count > 0:

            print(
                f"{label:<10}: Detected ({count})"
            )

            detected_count += 1

        else:

            print(
                f"{label:<10}: NOT detected"
            )

    print(
        f"\nDetected classes: "
        f"{detected_count}/{NUM_CLASSES}"
    )

    print(
        f"Best validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"\nCheckpoint: "
        f"{BALANCED_CKPT_FILE}"
    )


if __name__ == "__main__":
    main()