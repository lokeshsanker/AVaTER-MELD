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

NUM_PER_CLASS = 100
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4


print("Device:", DEVICE)

df = pd.read_csv(MANIFEST)

# ---------------------------------------------------------
# Balanced subset: 100 samples from every emotion
# ---------------------------------------------------------
parts = []

for label in LABEL2IDX.keys():
    class_df = df[df["label"] == label].head(NUM_PER_CLASS)

    if len(class_df) < NUM_PER_CLASS:
        raise RuntimeError(
            f"Not enough samples for {label}: {len(class_df)}"
        )

    parts.append(class_df)

small_df = pd.concat(parts, ignore_index=True)

print("\nBalanced dataset:")
print(small_df["label"].value_counts())

print("\nTotal samples:", len(small_df))


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
dataset = VideoEmotionDataset(
    small_df,
    augment=False
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
model = VideoEmotionModel(
    freeze_backbone=False
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------
for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for frames, labels, clip_ids in loader:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits, _ = model(frames)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    print(
        f"Epoch {epoch:02d} | "
        f"loss={avg_loss:.4f} | "
        f"accuracy={accuracy:.4f}"
    )
