import os
import numpy as np
import pandas as pd
import torch

from video.model import FrameEncoder
from config import NUM_FRAMES, EMOTIONS, DEVICE


MANIFEST = "data/manifests/dev.csv"
CACHE_ROOT = "data/face_cache/dev"

print("\n================================")
print("MOBILENET FEATURE DIAGNOSTIC")
print("================================\n")

df = pd.read_csv(MANIFEST)

# Pick one sample from each of the 7 emotions
samples = []

for emotion in EMOTIONS:
    subset = df[df["label"] == emotion]

    if len(subset) == 0:
        print("WARNING: no samples for", emotion)
        continue

    samples.append(subset.iloc[0])


# Load MobileNetV2 feature extractor
encoder = FrameEncoder().to(DEVICE)
encoder.eval()

print("Device:", DEVICE)
print("Samples:", len(samples))
print()


all_features = []

with torch.no_grad():

    for row in samples:

        true_label = row["label"]
        clip_id = row["clip_id"]

        cache_path = os.path.join(
            CACHE_ROOT,
            clip_id + ".npy"
        )

        if not os.path.exists(cache_path):
            print("MISSING CACHE:", cache_path)
            continue

        frames = np.load(cache_path)

        # Use exactly NUM_FRAMES
        if len(frames) > NUM_FRAMES:
            indices = np.linspace(
                0,
                len(frames) - 1,
                NUM_FRAMES
            ).astype(int)

            frames = frames[indices]

        # uint8 H,W,C -> float C,H,W
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        frames = (frames - mean) / std

        frames = frames.to(DEVICE)

        # MobileNet features
        features = encoder(frames)

        # Average the 4 frame features
        feature_mean = features.mean(dim=0)

        all_features.append(feature_mean.cpu())

        print("--------------------------------")
        print("Clip       :", clip_id)
        print("Actual     :", true_label)
        print("Input      :", tuple(frames.shape))
        print("Features   :", tuple(features.shape))

        print(
            "Feature mean:",
            feature_mean.mean().item()
        )

        print(
            "Feature std :",
            feature_mean.std().item()
        )

        print(
            "Feature min :",
            feature_mean.min().item()
        )

        print(
            "Feature max :",
            feature_mean.max().item()
        )


# Compare feature vectors between the 7 emotions
print("\n================================")
print("FEATURE DIFFERENCE TEST")
print("================================\n")

if len(all_features) >= 2:

    feature_matrix = torch.stack(all_features)

    print(
        "Feature matrix shape:",
        tuple(feature_matrix.shape)
    )

    # Pairwise Euclidean distances
    distances = torch.cdist(
        feature_matrix,
        feature_matrix,
        p=2
    )

    print("\nPairwise feature distances:\n")

    for i in range(len(samples)):

        for j in range(i + 1, len(samples)):

            emotion_i = samples[i]["label"]
            emotion_j = samples[j]["label"]

            print(
                f"{emotion_i:10s} <-> "
                f"{emotion_j:10s} : "
                f"{distances[i, j].item():.4f}"
            )

    # Average distance between different samples
    mask = ~torch.eye(
        len(feature_matrix),
        dtype=torch.bool
    )

    avg_distance = distances[mask].mean().item()

    print(
        "\nAverage pairwise distance:",
        f"{avg_distance:.4f}"
    )

else:

    print("Not enough feature vectors for comparison.")

print("\nDiagnostic complete.")