"""
Audio V2-A: Wav2Vec2 + attention pooling + balanced sampling.

Goal:
    Improve minority emotion detection without changing
    the underlying AudioEmotionModel architecture.

Selection metric:
    Validation Macro-F1

Usage:
    python audio/train_audio_v2.py `
        --manifest data/manifests/train.csv `
        --val_manifest data/manifests/dev.csv
"""

import argparse
import os
import sys
import random

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import (
    DEVICE,
    AUDIO_BATCH_SIZE,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
)

from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel


EMOTIONS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

NUM_CLASSES = len(EMOTIONS)

CHECKPOINT_DIR = "checkpoints/audio_v2"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "audio_v2_balanced_sampling.pt"
)

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_counts(df):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for label in df["label"]:
        label_idx = EMOTIONS.index(label)
        counts[label_idx] += 1

    return counts


def create_balanced_sampler(df):
    class_counts = get_class_counts(df)

    print("\nTraining class distribution:")

    for idx, emotion in enumerate(EMOTIONS):
        print(
            f"  {emotion:10s}: {class_counts[idx]}"
        )

    # Inverse-frequency class weights
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = np.array([
        class_weights[EMOTIONS.index(label)]
        for label in df["label"]
    ])

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []

    context = (
        torch.enable_grad()
        if is_train
        else torch.no_grad()
    )

    with context:

        for input_values, labels, _ in tqdm(
            loader,
            leave=False
        ):

            input_values = input_values.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_values)

            loss = criterion(
                logits,
                labels
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

            total_loss += (
                loss.item() * labels.size(0)
            )

            preds = logits.argmax(dim=-1)

            all_labels.extend(
                labels.detach().cpu().numpy()
            )

            all_preds.extend(
                preds.detach().cpu().numpy()
            )

    avg_loss = total_loss / len(all_labels)

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_preds,
    )


def main():

    set_seed()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--val_manifest",
        required=True
    )

    parser.add_argument(
        "--unfreeze_encoder",
        action="store_true",
        help="Fine-tune Wav2Vec2 encoder."
    )

    args = parser.parse_args()

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("AUDIO V2-A: BALANCED SAMPLING")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    train_df = load_manifest(
        args.manifest
    )

    val_df = load_manifest(
        args.val_manifest
    )

    print(
        f"\nTraining samples: {len(train_df)}"
    )

    print(
        f"Validation samples: {len(val_df)}"
    )

    train_sampler = create_balanced_sampler(
        train_df
    )

    train_loader = DataLoader(
        AudioEmotionDataset(train_df),
        batch_size=AUDIO_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        AudioEmotionDataset(val_df),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioEmotionModel(
        freeze_encoder=not args.unfreeze_encoder
    ).to(DEVICE)

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_count = sum(
        p.numel()
        for p in trainable_params
    )

    print(
        f"\nTotal parameters: {total_params:,}"
    )

    print(
        f"Trainable parameters: {trainable_count:,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=LR,
        weight_decay=1e-4,
    )

    criterion = torch.nn.CrossEntropyLoss()

    best_macro_f1 = -1.0
    patience_left = EARLY_STOP_PATIENCE

    print("\nStarting Audio V2-A training...")
    print(
        f"Maximum epochs: {EPOCHS}"
    )

    for epoch in range(1, EPOCHS + 1):

        (
            train_loss,
            train_acc,
            train_macro,
            train_weighted,
            _,
            _,
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        (
            val_loss,
            val_acc,
            val_macro,
            val_weighted,
            val_labels,
            val_preds,
        ) = run_epoch(
            model,
            val_loader,
            criterion,
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: {train_acc:.4f}"
        )

        print(
            f"Train Macro-F1: {train_macro:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: {val_acc:.4f}"
        )

        print(
            f"Val Macro-F1: {val_macro:.4f}"
        )

        print(
            f"Val Weighted-F1: {val_weighted:.4f}"
        )

        if val_macro > best_macro_f1:

            best_macro_f1 = val_macro
            patience_left = EARLY_STOP_PATIENCE

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro,
                    "val_weighted_f1": val_weighted,
                    "emotions": EMOTIONS,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"  -> New best Macro-F1!"
            )

            print(
                f"  -> Saved: {CHECKPOINT_PATH}"
            )

        else:

            patience_left -= 1

            print(
                f"  -> No improvement."
                f" Patience: {patience_left}"
            )

            if patience_left <= 0:

                print(
                    "\nEarly stopping."
                )

                break

    print("\n" + "=" * 70)
    print("AUDIO V2-A BEST VALIDATION RESULT")
    print("=" * 70)

    print(
        f"Best Validation Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    # Reload best model
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        val_loss,
        val_acc,
        val_macro,
        val_weighted,
        val_labels,
        val_preds,
    ) = run_epoch(
        model,
        val_loader,
        criterion,
    )

    print(
        f"\nAccuracy: {val_acc:.4f}"
    )

    print(
        f"Macro-F1: {val_macro:.4f}"
    )

    print(
        f"Weighted-F1: {val_weighted:.4f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            val_labels,
            val_preds,
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTIONS,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()