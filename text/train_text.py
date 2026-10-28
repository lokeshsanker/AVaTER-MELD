"""
Fine-tune the text branch (DistilBERT) on EMOTION labels.

Usage:
    python text/train_text.py --manifest data/train.csv --val_manifest data/val.csv
"""

import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, TEXT_BATCH_SIZE, LR, EPOCHS, EARLY_STOP_PATIENCE, TEXT_CKPT
from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for input_ids, attention_mask, labels, _ in tqdm(loader, leave=False):
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(input_ids, attention_mask)
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
    args = parser.parse_args()

    train_df = load_manifest(args.manifest)
    val_df = load_manifest(args.val_manifest)

    train_loader = DataLoader(TextEmotionDataset(train_df),
                               batch_size=TEXT_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(TextEmotionDataset(val_df),
                             batch_size=TEXT_BATCH_SIZE, shuffle=False, num_workers=2)

    model = TextEmotionModel().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, patience_left = 0.0, EARLY_STOP_PATIENCE

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"[Text] Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_left = EARLY_STOP_PATIENCE
            os.makedirs(TEXT_CKPT, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(TEXT_CKPT, "pytorch_model.bin"))
            print(f"  -> saved new best checkpoint to {TEXT_CKPT}")
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break

    print(f"Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
