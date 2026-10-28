import os
import numpy as np
import pandas as pd

from config import LABEL2IDX

MANIFEST = "data/manifests/train.csv"

df = pd.read_csv(MANIFEST)

# Use the same 350 validation samples from our previous experiment
parts = []

for label in LABEL2IDX.keys():
    class_df = df[df["label"] == label].reset_index(drop=True)
    parts.append(class_df.iloc[100:150])

val_df = pd.concat(parts, ignore_index=True)

cache_dir = "data/face_cache/train"

print("Validation clips:", len(val_df))

bad_shape = 0
bad_frames = 0
total_frames = 0

frame_means = []
frame_stds = []

for _, row in val_df.iterrows():

    path = os.path.join(
        cache_dir,
        row["clip_id"] + ".npy"
    )

    if not os.path.exists(path):
        print("MISSING:", path)
        continue

    faces = np.load(path)

    if faces.ndim != 4 or faces.shape[-1] != 3:
        bad_shape += 1
        continue

    n = faces.shape[0]

    if n < 4:
        bad_frames += 1

    total_frames += n

    # Sample the same 4 frames used by VideoEmotionDataset
    indices = np.linspace(
        0,
        n - 1,
        4
    ).astype(int)

    sampled = faces[indices].astype(np.float32)

    # Per-frame statistics
    for frame in sampled:
        frame_means.append(frame.mean())
        frame_stds.append(frame.std())


print()
print("Bad shape:", bad_shape)
print("Clips with <4 cached frames:", bad_frames)
print("Average cached frames per clip:", total_frames / len(val_df))
print("Sampled frame mean:", np.mean(frame_means))
print("Sampled frame std :", np.mean(frame_stds))

print()
print("Frame mean percentiles:")
print(np.percentile(frame_means, [0, 1, 5, 25, 50, 75, 95, 99, 100]))

print()
print("Frame std percentiles:")
print(np.percentile(frame_stds, [0, 1, 5, 25, 50, 75, 95, 99, 100]))
