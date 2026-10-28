import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL2IDX, DEVICE
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel


MANIFEST = "data/manifests/train.csv"

TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 50

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3


print("Device:", DEVICE)


df = pd.read_csv(MANIFEST)


# =========================================================
# NON-OVERLAPPING BALANCED DATA
# =========================================================

train_parts = []
val_parts = []

for label in LABEL2IDX.keys():

    class_df = df[df["label"] == label].reset_index(drop=True)

    train_parts.append(
        class_df.iloc[:TRAIN_PER_CLASS]
    )

    val_parts.append(
        class_df.iloc[
            TRAIN_PER_CLASS:
            TRAIN_PER_CLASS + VAL_PER_CLASS
        ]
    )


train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)


print("\nTRAIN:")
print(train_df["label"].value_counts())

print("\nVALIDATION:")
print(val_df["label"].value_counts())


# =========================================================
# DATA
# =========================================================

train_dataset = VideoEmotionDataset(
    train_df,
    augment=False
)

val_dataset = VideoEmotionDataset(
    val_df,
    augment=False
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# MODEL
# =========================================================

model = VideoEmotionModel(
    freeze_backbone=True
).to(DEVICE)


# IMPORTANT:
# Keep the frozen MobileNet backbone in evaluation mode
# so BatchNorm running statistics do not change.

model.frame_encoder.eval()


criterion = nn.CrossEntropyLoss()


# Only train GRU + attention + classifier
trainable_params = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR
)


print("\nTrainable parameters:")
print(sum(p.numel() for p in trainable_params))


# =========================================================
# TRAINING
# =========================================================

for epoch in range(1, EPOCHS + 1):

    model.train()

    # model.train() changes all modules to train mode,
    # so explicitly restore frozen MobileNet to eval mode.
    model.frame_encoder.eval()


    train_loss = 0.0
    train_correct = 0
    train_total = 0


    for frames, labels, clip_ids in train_loader:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(frames)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total
    train_acc = train_correct / train_total


    # =====================================================
    # VALIDATION
    # =====================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for frames, labels, clip_ids in val_loader:

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            logits, _ = model(frames)

            loss = criterion(logits, labels)

            val_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss /= val_total
    val_acc = val_correct / val_total


    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )
