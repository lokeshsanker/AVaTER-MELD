import os
import sys
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL2IDX, DEVICE
from video.video_dataset import VideoEmotionDataset
from video.model import FrameEncoder


MANIFEST = "data/manifests/train.csv"

SAMPLES_PER_CLASS = 100
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3


print("Device:", DEVICE)

df = pd.read_csv(MANIFEST)


# =========================================================
# 100 samples from every class
# =========================================================

parts = []

for label in LABEL2IDX.keys():

    class_df = df[df["label"] == label].sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    parts.append(class_df)


balanced_df = pd.concat(
    parts,
    ignore_index=True
)


print("\nBalanced dataset:")
print(balanced_df["label"].value_counts())


# =========================================================
# RANDOM STRATIFIED SPLIT
# =========================================================

train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.20,
    random_state=42,
    stratify=balanced_df["label"]
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)


print("\nTrain:")
print(train_df["label"].value_counts())

print("\nValidation:")
print(val_df["label"].value_counts())


# =========================================================
# DATASETS
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
# FROZEN MOBILENET
# =========================================================

encoder = FrameEncoder(
    pretrained=True,
    freeze_backbone=True
).to(DEVICE)

encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False


# =========================================================
# SIMPLE CLASSIFIER
# =========================================================

classifier = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 7)
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    classifier.parameters(),
    lr=LR
)


# =========================================================
# TRAIN
# =========================================================

for epoch in range(1, EPOCHS + 1):

    classifier.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0


    for frames, labels, clip_ids in train_loader:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)


        # ---------------------------------------------
        # Extract MobileNet features
        # ---------------------------------------------

        with torch.no_grad():

            B, T, C, H, W = frames.shape

            flat = frames.view(
                B * T,
                C,
                H,
                W
            )

            features = encoder(flat)

            features = features.view(
                B,
                T,
                1280
            )

            # Average the 4 frames
            features = features.mean(dim=1)


        # ---------------------------------------------
        # Classifier
        # ---------------------------------------------

        optimizer.zero_grad()

        logits = classifier(features)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item() * labels.size(0)
        )

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

    classifier.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for frames, labels, clip_ids in val_loader:

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)


            B, T, C, H, W = frames.shape

            flat = frames.view(
                B * T,
                C,
                H,
                W
            )

            features = encoder(flat)

            features = features.view(
                B,
                T,
                1280
            )

            features = features.mean(dim=1)


            logits = classifier(features)

            loss = criterion(
                logits,
                labels
            )


            val_loss += (
                loss.item() * labels.size(0)
            )

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
