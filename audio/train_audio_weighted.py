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
    AUDIO_BATCH_SIZE,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE,
    EMOTIONS,
)
from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel


OUTPUT_CKPT = "checkpoints/audio_model_weighted.pt"


def get_class_weights(df):
    counts = df["label"].value_counts()

    total = len(df)
    n_classes = len(EMOTIONS)

    weights = []

    for emotion in EMOTIONS:
        count = counts.get(emotion, 0)
        weight = total / (n_classes * count) if count > 0 else 0.0
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    print("\nClass distribution and weights:")
    for emotion, weight in zip(EMOTIONS, weights):
        count = counts.get(emotion, 0)
        print(f"{emotion:>10}: {count:5d} samples | weight={weight:.4f}")

    return weights


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
        for input_values, labels, _ in tqdm(loader, leave=False):
            input_values = input_values.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_values)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
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
        "--unfreeze_encoder",
        action="store_true"
    )

    args = parser.parse_args()

    print("Loading manifests...")

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    print(f"Training samples:   {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    train_loader = DataLoader(
        AudioEmotionDataset(train_df),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        AudioEmotionDataset(val_df),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("\nLoading Wav2Vec2 audio model...")

    model = AudioEmotionModel(
        freeze_encoder=not args.unfreeze_encoder
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    class_weights = get_class_weights(train_df).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    best_val_acc = 0.0
    patience_left = EARLY_STOP_PATIENCE

    print("\nStarting weighted-loss training...\n")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
        )

        print(
            f"[Audio Weighted] Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            patience_left = EARLY_STOP_PATIENCE

            torch.save(
                model.state_dict(),
                OUTPUT_CKPT
            )

            print(
                f"  -> saved new best checkpoint to "
                f"{OUTPUT_CKPT}"
            )

        else:

            patience_left -= 1

            if patience_left == 0:

                print("Early stopping.")
                break

    print(
        f"\nBest validation accuracy: "
        f"{best_val_acc:.4f} "
        f"({best_val_acc * 100:.2f}%)"
    )

    print(f"Checkpoint: {OUTPUT_CKPT}")


if __name__ == "__main__":
    main()
