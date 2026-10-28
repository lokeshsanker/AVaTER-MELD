"""
Train the video branch (MobileNetV2 + BiGRU + attention pooling).

Usage:
    python video/train_video.py --manifest data/manifests/train.csv --val_manifest data/manifests/dev.csv
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE,
    VIDEO_BATCH_SIZE,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    VIDEO_CKPT,
    LABEL2IDX,
)

from data.dataset import load_manifest
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for frames, labels, _ in tqdm(loader, leave=False):

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(frames)

            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)

            correct += (
                (logits.argmax(-1) == labels)
                .sum()
                .item()
            )

            total += labels.size(0)

    return total_loss / total, correct / total


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

    parser.add_argument(
        "--freeze_backbone",
        action="store_true"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load manifests
    # ---------------------------------------------------------

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    train_loader = DataLoader(
        VideoEmotionDataset(
            train_df,
            augment=True
        ),
        batch_size=VIDEO_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        VideoEmotionDataset(
            val_df,
            augment=False
        ),
        batch_size=VIDEO_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Batch size: {VIDEO_BATCH_SIZE}")

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = VideoEmotionModel(
        freeze_backbone=args.freeze_backbone
    ).to(DEVICE)

    print(f"Device: {DEVICE}")

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LR
    )

    # ---------------------------------------------------------
    # Class-weighted loss
    #
    # This compensates for MELD's class imbalance.
    # ---------------------------------------------------------

    class_counts = train_df["label"].value_counts()

    class_weights = torch.tensor(
        [
            len(train_df)
            / (
                len(class_counts)
                * class_counts[label]
            )
            for label in LABEL2IDX.keys()
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    print("\nClass distribution:")

    for label in LABEL2IDX.keys():
        print(
            f"  {label:10s}: "
            f"{class_counts[label]}"
        )

    print("\nClass weights:")

    for label, weight in zip(
        LABEL2IDX.keys(),
        class_weights
    ):
        print(
            f"  {label:10s}: "
            f"{weight.item():.4f}"
        )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_val_acc = 0.0
    patience_left = EARLY_STOP_PATIENCE

    print("\nStarting video training...\n")

    for epoch in range(1, EPOCHS + 1):

        print(
            f"\n========== Epoch {epoch:02d}/{EPOCHS} =========="
        )

        # -----------------------------
        # Training
        # -----------------------------

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        # -----------------------------
        # Validation
        # -----------------------------

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None
        )

        # -----------------------------
        # Results
        # -----------------------------

        print(
            f"\n[Video] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        # -----------------------------
        # Save best model
        # -----------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            patience_left = EARLY_STOP_PATIENCE

            torch.save(
                model.state_dict(),
                VIDEO_CKPT
            )

            print(
                f"  -> saved new best checkpoint to "
                f"{VIDEO_CKPT}"
            )

        else:

            patience_left -= 1

            print(
                f"  -> no improvement "
                f"(patience left: {patience_left})"
            )

            if patience_left == 0:

                print(
                    "\nEarly stopping."
                )

                break

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print(
        f"\nBest val accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Best checkpoint: "
        f"{VIDEO_CKPT}"
    )


if __name__ == "__main__":
    main()