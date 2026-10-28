"""
Train the audio branch (Wav2Vec2 + attention pooling + MLP head).

Usage:
    python audio/train_audio.py --manifest data/train.csv --val_manifest data/val.csv
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, AUDIO_BATCH_SIZE, LR, EPOCHS, EARLY_STOP_PATIENCE, AUDIO_CKPT
from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for input_values, labels, _ in tqdm(loader, leave=False):
            input_values, labels = input_values.to(DEVICE), labels.to(DEVICE)

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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--unfreeze_encoder", action="store_true",
                         help="Fine-tune Wav2Vec2 weights too (slower, needs more data).")
    args = parser.parse_args()

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    train_loader = DataLoader(AudioEmotionDataset(train_df),
                               batch_size=AUDIO_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(AudioEmotionDataset(val_df),
                             batch_size=AUDIO_BATCH_SIZE, shuffle=False, num_workers=2)

    model = AudioEmotionModel(freeze_encoder=not args.unfreeze_encoder).to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, patience_left = 0.0, EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"[Audio] Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = EARLY_STOP_PATIENCE
            torch.save(model.state_dict(), AUDIO_CKPT)
            print(f"  -> saved new best checkpoint to {AUDIO_CKPT}")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
