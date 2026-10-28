import os
import sys
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import NUM_FRAMES, FACE_SIZE
from data.dataset import load_manifest
from video.face_utils import extract_aligned_faces

SETS = {
    "train": "data/manifests/train.csv",
    "dev": "data/manifests/dev.csv",
    "test": "data/manifests/test.csv",
}

for split, manifest in SETS.items():
    df = load_manifest(manifest)
    out_dir = f"data/face_cache/{split}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nCaching {split}: {len(df)} videos")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        out_path = os.path.join(out_dir, row["clip_id"] + ".npy")

        if os.path.exists(out_path):
            continue

        try:
            faces = extract_aligned_faces(
                row["video_path"],
                NUM_FRAMES,
                FACE_SIZE
            )
            np.save(out_path, faces)
        except Exception as e:
            print(f"\nFAILED {row['clip_id']}: {e}")

    print(f"Finished {split}")
