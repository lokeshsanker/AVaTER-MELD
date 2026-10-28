import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL2IDX, DEVICE
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel


MANIFEST = "data/manifests/train.csv"
NUM_PER_CLASS = 10
BATCH_SIZE = 7
EPOCHS = 30
LR = 1e-4


print("Device:", DEVICE)

df = pd.read_csv(MANIFEST)

print("\nFull training distribution:")
print(df["label"].value_counts())


# ---------------------------------------------------------
# Select 10 samples from EACH emotion
# ---------------------------------------------------------
indices = []

for label in LABEL2IDX.keys():
    label_indices = df.index[df["label"] == label].tolist()

    if len(label_indices) < NUM_PER_CLASS:
        raise RuntimeError(
            f"Not enough samples for {label}: {len(label_indices)}"
        )

    indices.extend(label_indices[:NUM_PER_CLASS])


small_df = df.loc[indices].reset_index(drop=True)

print("\nTiny dataset:")
print(small_df["label"].value_counts())


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
dataset = VideoEmotionDataset(small_df, augment=False)

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

print("\nFinal predictions:")
model.eval()

with torch.no_grad():
    for frames, labels, clip_ids in loader:

        frames = frames.to(DEVICE)

        logits, _ = model(frames)

        preds = logits.argmax(dim=1).cpu().tolist()

        labels_cpu = labels.tolist()

        for clip_id, true_idx, pred_idx in zip(
            clip_ids,
            labels_cpu,
            preds
        ):
            true_label = list(LABEL2IDX.keys())[true_idx]
            pred_label = list(LABEL2IDX.keys())[pred_idx]

            print(
                f"{clip_id:15s} "
                f"true={true_label:9s} "
                f"pred={pred_label:9s}"
            )
