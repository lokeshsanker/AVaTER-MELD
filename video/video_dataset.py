import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from config import FACE_SIZE, LABEL2IDX, NUM_FRAMES


_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


class VideoEmotionDataset(Dataset):
    def __init__(self, df, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.augment = augment

        # Automatically determine cache directory from manifest paths
        manifest_paths = self.df["video_path"].astype(str).tolist()

        if manifest_paths and "\\train_splits\\" in manifest_paths[0]:
            self.cache_dir = os.path.join("data", "face_cache", "train")
        elif manifest_paths and "\\dev_splits_complete\\" in manifest_paths[0]:
            self.cache_dir = os.path.join("data", "face_cache", "dev")
        elif manifest_paths and "\\output_repeated_splits_test\\" in manifest_paths[0]:
            self.cache_dir = os.path.join("data", "face_cache", "test")
        else:
            raise ValueError(
                "Could not determine cache directory from video_path"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cache_path = os.path.join(
            self.cache_dir,
            row["clip_id"] + ".npy"
        )

        faces = np.load(cache_path)

        # Uniformly sample NUM_FRAMES from the cached frames.
        # Example: 16 cached frames -> 4 frames approximately [0, 5, 10, 15].
        if faces.shape[0] > NUM_FRAMES:
            indices = np.linspace(
                0,
                faces.shape[0] - 1,
                NUM_FRAMES
            ).astype(int)
            faces = faces[indices]

        # (T,H,W,3) uint8 -> (T,3,H,W) float
        frames = (
            torch.from_numpy(faces)
            .float()
            .permute(0, 3, 1, 2)
            / 255.0
        )

        frames = torch.stack([
            _normalize(frame) for frame in frames
        ])

        label = LABEL2IDX[row["label"]]

        return frames, label, row["clip_id"]