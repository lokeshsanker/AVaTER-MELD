import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL2IDX, DEVICE
from video.video_dataset import VideoEmotionDataset


MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

SAMPLES_PER_CLASS = 100
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3


print("Device:", DEVICE)


# =========================================================
# Load FER-ViT
# =========================================================

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

vit = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
).to(DEVICE)

vit.eval()

for p in vit.parameters():
    p.requires_grad = False


HIDDEN_SIZE = vit.config.hidden_size

print("ViT hidden size:", HIDDEN_SIZE)


# =========================================================
# Balanced MELD subset
# =========================================================

df = pd.read_csv(
    "data/manifests/train.csv"
)

parts = []

for label in LABEL2IDX.keys():

    class_df = df[
        df["label"] == label
    ].sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    parts.append(class_df)


balanced_df = pd.concat(
    parts,
    ignore_index=True
)


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
# Datasets
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
# GRU + Attention + Classifier
# =========================================================

class FERVideoModel(nn.Module):

    def __init__(self, feature_dim=768):

        super().__init__()

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attn = nn.Linear(
            256,
            1
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7)
        )


    def forward(self, features):

        # features:
        # B x T x 768

        seq, _ = self.gru(features)

        scores = self.attn(seq).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.bmm(
            weights.unsqueeze(1),
            seq
        ).squeeze(1)

        logits = self.classifier(
            pooled
        )

        return logits


model = FERVideoModel(
    feature_dim=HIDDEN_SIZE
).to(DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)


# =========================================================
# Extract ViT features
# =========================================================

def extract_features(frames):

    B, T, C, H, W = frames.shape

    all_features = []

    for b in range(B):

        frame_images = []

        for t in range(T):

            frame = frames[b, t]

            # Undo ImageNet normalization used by dataset
            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=frame.device
            ).view(3, 1, 1)

            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=frame.device
            ).view(3, 1, 1)

            frame = frame * std + mean

            frame = (
                frame.clamp(0, 1) * 255
            ).byte()

            frame = frame.permute(
                1, 2, 0
            ).cpu().numpy()

            frame_images.append(
                Image.fromarray(frame)
            )


        inputs = processor(
            images=frame_images,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }


        with torch.no_grad():

            outputs = vit.vit(
                **inputs
            )

            # CLS token
            features = outputs.last_hidden_state[:, 0]

        all_features.append(
            features
        )


    return torch.stack(
        all_features
    )


# =========================================================
# Training
# =========================================================

for epoch in range(1, EPOCHS + 1):

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0


    for frames, labels, clip_ids in train_loader:

        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)


        features = extract_features(
            frames
        )


        optimizer.zero_grad()

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()


        train_loss += (
            loss.item() *
            labels.size(0)
        )

        train_correct += (
            logits.argmax(dim=1) ==
            labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total
    train_acc = (
        train_correct /
        train_total
    )


    # =====================================================
    # Validation
    # =====================================================

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0


    with torch.no_grad():

        for frames, labels, clip_ids in val_loader:

            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            features = extract_features(
                frames
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )


            val_loss += (
                loss.item() *
                labels.size(0)
            )

            val_correct += (
                logits.argmax(dim=1) ==
                labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss /= val_total
    val_acc = (
        val_correct /
        val_total
    )


    print(
        f"Epoch {epoch:02d} | "
        f"train_loss={train_loss:.4f} | "
        f"train_acc={train_acc:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )
