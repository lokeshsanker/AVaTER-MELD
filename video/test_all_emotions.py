import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from video.model import VideoEmotionModel
from config import NUM_FRAMES, FACE_SIZE, EMOTIONS, DEVICE

MANIFEST = "data/manifests/dev.csv"
CACHE_ROOT = "data/face_cache/dev"
CHECKPOINT = "checkpoints/video_model.pt"

df = pd.read_csv(MANIFEST)

# Pick ONE video from each emotion
samples = []

for emotion in EMOTIONS:
    subset = df[df["label"] == emotion]

    if len(subset) == 0:
        print("WARNING: no samples for", emotion)
        continue

    row = subset.iloc[0]
    samples.append(row)


print("\n================================")
print("7-EMOTION VIDEO TEST")
print("================================\n")

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

        # Make sure exactly NUM_FRAMES are used
        if len(frames) > NUM_FRAMES:
            indices = np.linspace(
                0,
                len(frames) - 1,
                NUM_FRAMES
            ).astype(int)

            frames = frames[indices]

        frames = torch.from_numpy(frames).float() / 255.0

        # H,W,C -> C,H,W
        frames = frames.permute(0, 3, 1, 2)

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(1, 3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(1, 3, 1, 1)

        frames = (frames - mean) / std

        frames = frames.unsqueeze(0).to(DEVICE)

        logits, _ = model(frames)

        probs = F.softmax(logits, dim=1)[0]

        pred_idx = torch.argmax(probs).item()
        pred_label = EMOTIONS[pred_idx]

        print("--------------------------------")
        print(f"Clip       : {clip_id}")
        print(f"Actual     : {true_label}")
        print(f"Predicted  : {pred_label}")

        print("\nProbabilities:")

        for emotion, probability in zip(
            EMOTIONS,
            probs.cpu().numpy()
        ):
            print(
                f"  {emotion:10s}: "
                f"{probability:.4f}"
            )

        print()