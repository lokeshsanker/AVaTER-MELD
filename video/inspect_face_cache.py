import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MANIFEST = "data/manifests/train.csv"
CACHE_ROOT = "data/face_cache/train"

EMOTIONS = [
    "neutral", "joy", "sadness",
    "anger", "fear", "disgust", "surprise"
]

SAMPLES_PER_EMOTION = 3

df = pd.read_csv(MANIFEST)

fig, axes = plt.subplots(
    len(EMOTIONS) * SAMPLES_PER_EMOTION,
    4,
    figsize=(12, 4 * len(EMOTIONS) * SAMPLES_PER_EMOTION)
)

row = 0

for emotion in EMOTIONS:
    subset = df[df["label"] == emotion]

    samples = subset.sample(
        min(SAMPLES_PER_EMOTION, len(subset)),
        random_state=42
    )

    for _, sample in samples.iterrows():

        video_path = str(sample["video_path"])
        video_name = os.path.splitext(
            os.path.basename(video_path)
        )[0]

        cache_path = os.path.join(
            CACHE_ROOT,
            video_name + ".npy"
        )

        if not os.path.exists(cache_path):
            print("MISSING:", cache_path)
            continue

        frames = np.load(cache_path)

        print(
            emotion,
            video_name,
            frames.shape,
            frames.dtype
        )

        for j in range(4):
            axes[row, j].imshow(frames[j])
            axes[row, j].axis("off")

            if j == 0:
                axes[row, j].set_title(
                    f"{emotion} | {video_name}"
                )
            else:
                axes[row, j].set_title(
                    f"Frame {j+1}"
                )

        row += 1

plt.tight_layout()

output = "face_cache_diagnostic.png"
plt.savefig(output, dpi=150)

print()
print("Saved:", output)

plt.show()