import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from config import DEVICE, VIDEO_BATCH_SIZE, VIDEO_CKPT, IDX2LABEL
from data.dataset import load_manifest
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel


# ---------------------------------------------------------
# Load validation data
# ---------------------------------------------------------

val_df = load_manifest(
    "data/manifests/dev.csv"
)

val_dataset = VideoEmotionDataset(
    val_df,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=VIDEO_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ---------------------------------------------------------
# Load best video model
# ---------------------------------------------------------

model = VideoEmotionModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        VIDEO_CKPT,
        map_location=DEVICE
    )
)

model.eval()


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for frames, labels, _ in val_loader:

        frames = frames.to(DEVICE)

        logits, _ = model(frames)

        predictions = logits.argmax(
            dim=1
        ).cpu().tolist()

        all_predictions.extend(
            predictions
        )

        all_labels.extend(
            labels.tolist()
        )


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n================================")
print("VIDEO MODEL EVALUATION")
print("================================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

target_names = [
    IDX2LABEL[i]
    for i in range(len(IDX2LABEL))
]

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(IDX2LABEL))),
        target_names=target_names,
        digits=4,
        zero_division=0
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(IDX2LABEL)))
)

print("\nConfusion Matrix:")
print("\nLabels:")

print(
    " | ".join(target_names)
)

print(cm)


# ---------------------------------------------------------
# Prediction distribution
# ---------------------------------------------------------

from collections import Counter

print("\nPrediction distribution:")

prediction_counts = Counter(
    all_predictions
)

for i in range(len(IDX2LABEL)):

    print(
        f"{IDX2LABEL[i]:10s}: "
        f"{prediction_counts[i]}"
    )