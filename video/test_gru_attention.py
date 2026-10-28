import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from video.model import VideoEmotionModel
from config import NUM_FRAMES, EMOTIONS, DEVICE


MANIFEST = "data/manifests/dev.csv"
CACHE_ROOT = "data/face_cache/dev"
CHECKPOINT = "checkpoints/video_model.pt"


print("\n================================")
print("GRU + ATTENTION DIAGNOSTIC")
print("================================\n")


df = pd.read_csv(MANIFEST)

# One sample from each of the 7 emotions
samples = []

for emotion in EMOTIONS:
    subset = df[df["label"] == emotion]

    if len(subset) > 0:
        samples.append(subset.iloc[0])


# Load model
model = VideoEmotionModel(freeze_backbone=True).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()


all_gru = []
all_attention = []
all_logits = []


with torch.no_grad():

    for row in samples:

        emotion = row["label"]
        clip_id = row["clip_id"]

        cache_path = os.path.join(
            CACHE_ROOT,
            clip_id + ".npy"
        )

        if not os.path.exists(cache_path):
            print("MISSING:", cache_path)
            continue

        frames = np.load(cache_path)

        # Exactly 4 frames
        if len(frames) > NUM_FRAMES:
            indices = np.linspace(
                0,
                len(frames) - 1,
                NUM_FRAMES
            ).astype(int)

            frames = frames[indices]

        # H,W,C -> C,H,W
        frames = torch.from_numpy(
            frames
        ).float() / 255.0

        frames = frames.permute(
            0, 3, 1, 2
        )

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        frames = (frames - mean) / std

        frames = frames.to(DEVICE)

        # =================================
        # 1. MobileNet
        # =================================

        frame_features = model.frame_encoder(frames)

        # Add batch dimension:
        # [4,1280] -> [1,4,1280]
        x = frame_features.unsqueeze(0)

        # =================================
        # 2. BiGRU
        # =================================

        gru_output, _ = model.gru(x)

        # =================================
        # 3. Attention Pool
        # =================================

        attention_scores = model.pool.attn(
            gru_output
        )

        attention_weights = F.softmax(
            attention_scores,
            dim=1
        )

        pooled = (
            gru_output * attention_weights
        ).sum(dim=1)

        # =================================
        # 4. Classifier
        # =================================

        logits = model.classifier(
            pooled
        )

        probabilities = F.softmax(
            logits,
            dim=1
        )[0]

        # Save representations
        all_gru.append(
            gru_output.mean(dim=1)[0].cpu()
        )

        all_attention.append(
            pooled[0].cpu()
        )

        all_logits.append(
            logits[0].cpu()
        )

        # =================================
        # Print results
        # =================================

        print("--------------------------------")
        print("Clip       :", clip_id)
        print("Actual     :", emotion)

        print(
            "MobileNet  :",
            tuple(frame_features.shape)
        )

        print(
            "GRU        :",
            tuple(gru_output.shape)
        )

        print(
            "Attention  :",
            tuple(pooled.shape)
        )

        print(
            "Attention weights:",
            [
                round(v, 4)
                for v in attention_weights[0, :, 0]
                .cpu()
                .tolist()
            ]
        )

        print(
            "Logits:",
            [
                round(v, 4)
                for v in logits[0]
                .cpu()
                .tolist()
            ]
        )

        print(
            "Prediction:",
            EMOTIONS[
                probabilities.argmax().item()
            ]
        )

        print(
            "Probabilities:",
            [
                round(v, 4)
                for v in probabilities.cpu().tolist()
            ]
        )


# =================================
# Representation distance analysis
# =================================

print("\n================================")
print("REPRESENTATION DISTANCES")
print("================================\n")


def calculate_distance(features, name):

    matrix = torch.stack(features)

    distances = torch.cdist(
        matrix,
        matrix,
        p=2
    )

    mask = ~torch.eye(
        len(matrix),
        dtype=torch.bool
    )

    average = distances[mask].mean().item()

    print(
        f"{name:20s}: {average:.6f}"
    )


if len(all_gru) >= 2:

    calculate_distance(
        all_gru,
        "BiGRU"
    )

    calculate_distance(
        all_attention,
        "Attention"
    )

    calculate_distance(
        all_logits,
        "Logits"
    )


print("\nDiagnostic complete.")