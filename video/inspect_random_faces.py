import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import LABEL2IDX

MANIFEST = "data/manifests/train.csv"
CACHE_DIR = "data/face_cache/train"

random.seed(42)

df = pd.read_csv(MANIFEST)

# Take 2 clips from each emotion
selected = []

for label in LABEL2IDX.keys():
    class_df = df[df["label"] == label]

    samples = class_df.sample(
        n=2,
        random_state=42
    )

    selected.append(samples)

selected = pd.concat(selected, ignore_index=True)

print("Selected clips:")
print(selected[["clip_id", "label"]].to_string(index=False))


# ---------------------------------------------------------
# Create one figure per clip
# ---------------------------------------------------------

for _, row in selected.iterrows():

    clip_id = row["clip_id"]
    label = row["label"]

    path = os.path.join(
        CACHE_DIR,
        clip_id + ".npy"
    )

    faces = np.load(path)

    n = faces.shape[0]

    indices = np.linspace(
        0,
        n - 1,
        4
    ).astype(int)

    faces = faces[indices]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12, 3)
    )

    fig.suptitle(
        f"{clip_id} | TRUE: {label}",
        fontsize=14
    )

    for i, (ax, face) in enumerate(zip(axes, faces)):

        ax.imshow(face)

        ax.set_title(
            f"Frame {i+1}"
        )

        ax.axis("off")

    plt.tight_layout()

    output = os.path.join(
        "video",
        f"inspect_{clip_id}.png"
    )

    plt.savefig(
        output,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", output)

print("\nDone.")
