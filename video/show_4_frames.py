"""
Visualize the 4 frames used for FER-ViT feature extraction.

Each video utterance:
    Video
      ↓
    Sample 4 frames
      ↓
    4 facial frames
      ↓
    FER-ViT
      ↓
    4 × 768-D features
      ↓
    Mean Pooling
      ↓
    1 × 768-D video feature
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys


# ---------------------------------------------------------
# Get video path
# ---------------------------------------------------------

video_path = sys.argv[1]


# ---------------------------------------------------------
# Open video
# ---------------------------------------------------------

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{video_path}"
    )


total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

fps = cap.get(
    cv2.CAP_PROP_FPS
)

print("Video:", video_path)
print("Total frames:", total_frames)
print("FPS:", fps)


# ---------------------------------------------------------
# Select 4 frames evenly across the video
# ---------------------------------------------------------

indices = np.linspace(
    0,
    max(total_frames - 1, 0),
    4,
    dtype=int
)

print("Selected frame indices:", indices)


frames = []


# ---------------------------------------------------------
# Extract the 4 frames
# ---------------------------------------------------------

for frame_idx in indices:

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(frame_idx)
    )

    success, frame = cap.read()

    if not success:
        continue

    # OpenCV uses BGR.
    # Matplotlib expects RGB.
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frames.append(
        (frame_idx, frame_rgb)
    )


cap.release()


# ---------------------------------------------------------
# Display the 4 frames
# ---------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8)
)

axes = axes.flatten()


for i, ax in enumerate(axes):

    if i < len(frames):

        frame_idx, frame = frames[i]

        ax.imshow(frame)

        ax.set_title(
            f"Frame {i + 1}  |  Index {frame_idx}"
        )

    ax.axis("off")


plt.suptitle(
    "4 Sampled Frames → FER-ViT",
    fontsize=16
)

plt.tight_layout()

plt.show()