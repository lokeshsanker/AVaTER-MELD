
"""
V2 Text Training — MELD
DistilBERT with class-aware sampling, differential learning rates,
Macro-F1 checkpoint selection, and per-class evaluation.

Run:
python text/train_text_v2.py --manifest data/manifests/train.csv --val_manifest data/manifests/dev.csv
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import (
    DEVICE,
    EMOTIONS,
    LABEL2IDX,
    NUM_CLASSES,
    TEXT_BATCH_SIZE,
)
from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


SEED = 42
MAX_EPOCHS = 12
PATIENCE = 4

ENCODER_LR = 2e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 0.01

# Sampling exponent:
# 0 = natural distribution
# 1 = inverse-frequency sampling
SAMPLING_POWER = 0.5

CHECKPOINT_DIR = ROOT / "checkpoints" / "text_v2"
CHECKPOINT_PATH = CHECKPOINT_DIR / "pytorch_model.bin"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sampler(train_df):
    labels = train_df["label"].map(LABEL2IDX)

    if labels.isna().any():
        raise ValueError("Unknown emotion labels in training data")

    labels = labels.astype(int).to_numpy()

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    if np.any(counts == 0):
        raise ValueError("A training emotion class has zero samples")

    class_weights = (
        1.0 / counts.astype(np.float64)
    ) ** SAMPLING_POWER

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=len(labels),
        replacement=True
    )

    print("\nTraining class distribution:")
    for name, count in zip(EMOTIONS, counts):
        print(f"{name:10s}: {count}")

    print("\nSampling power:", SAMPLING_POWER)

    return sampler


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    losses = []
    true_labels = []
    predictions = []

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for input_ids, attention_mask, labels, _ in tqdm(
            loader,
            leave=False
        ):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            losses.append(
                (loss.item(), labels.size(0))
            )

            true_labels.extend(
                labels.cpu().tolist()
            )

            predictions.extend(
                logits.argmax(dim=-1).cpu().tolist()
            )

    total = sum(count for _, count in losses)

    avg_loss = sum(
        loss * count for loss, count in losses
    ) / total

    accuracy = accuracy_score(
        true_labels,
        predictions
    )

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "true": true_labels,
        "pred": predictions,
    }


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

    set_seed(SEED)

    print("=" * 65)
    print("V2 TEXT TRAINING - CLASS-AWARE DISTILBERT")
    print("=" * 65)
    print("Device:", DEVICE)
    print("Encoder LR:", ENCODER_LR)
    print("Classifier LR:", CLASSIFIER_LR)
    print("Max epochs:", MAX_EPOCHS)
    print("Checkpoint:", CHECKPOINT_PATH)

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    print("\nTrain samples:", len(train_df))
    print("Dev samples:", len(val_df))

    sampler = make_sampler(train_df)

    train_dataset = TextEmotionDataset(train_df)
    val_dataset = TextEmotionDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TEXT_BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model = TextEmotionModel().to(DEVICE)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": ENCODER_LR
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LR
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

        train_result = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_result = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(
            f"Train | Loss: {train_result['loss']:.4f}"
            f" | Accuracy: {train_result['accuracy']:.4f}"
            f" | Macro-F1: {train_result['macro_f1']:.4f}"
        )

        print(
            f"Dev   | Loss: {val_result['loss']:.4f}"
            f" | Accuracy: {val_result['accuracy']:.4f}"
            f" | Macro-F1: {val_result['macro_f1']:.4f}"
            f" | Weighted-F1: {val_result['weighted_f1']:.4f}"
        )

        print("\nDev per-class recall:")

        cm = confusion_matrix(
            val_result["true"],
            val_result["pred"],
            labels=list(range(NUM_CLASSES))
        )

        for i, name in enumerate(EMOTIONS):
            support = cm[i].sum()

            recall = (
                cm[i, i] / support
                if support > 0
                else 0.0
            )

            print(f"  {name:10s}: {recall:.4f}")

        if val_result["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_result["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print("\nSaved new best V2 checkpoint")

        else:
            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("\nEarly stopping triggered")
                break

    print("\nLoading best checkpoint...")

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state)

    final_result = run_epoch(
        model,
        val_loader,
        criterion
    )

    print("\n" + "=" * 65)
    print("V2 TEXT - FINAL DEV RESULTS")
    print("=" * 65)

    print(
        f"Accuracy    : {final_result['accuracy']:.4f}"
    )

    print(
        f"Macro-F1    : {final_result['macro_f1']:.4f}"
    )

    print(
        f"Weighted-F1 : {final_result['weighted_f1']:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            final_result["true"],
            final_result["pred"],
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTIONS,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        final_result["true"],
        final_result["pred"],
        labels=list(range(NUM_CLASSES))
    )

    print(cm)

    print("\n7-CLASS DETECTION:")

    detected = 0

    for i, name in enumerate(EMOTIONS):
        count = final_result["pred"].count(i)

        if count > 0:
            detected += 1

        print(
            f"{name:10s}: {count} predictions"
        )

    print(f"\nPredicted classes: {detected}/7")
    print(f"Best epoch: {best_epoch}")
    print(f"Best Dev Macro-F1: {best_macro_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    print("\nV2 experiment completed.")


if __name__ == "__main__":
    main()
