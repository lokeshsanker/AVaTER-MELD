"""
Extract Video V2 frame-sequence features.

Pipeline:
Face frames -> MobileNetV2 -> 1280-D feature per frame

We preserve the temporal sequence instead of mean-pooling it.

Output:
    data/video_v2_feature_cache_train.pt
    data/video_v2_feature_cache_dev.pt
    data/video_v2_feature_cache_test.pt

Expected feature shape:
    Train: [9988, 4, 1280]
    Dev:   [1108, 4, 1280]
    Test:  [2610, 4, 1280]
"""

import os
import sys

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.models as tv_models

from config import (
    DEVICE,
    FACE_SIZE,
    NUM_FRAMES,
    LABEL2IDX,
)

from data.dataset import load_manifest
from video.video_dataset import VideoEmotionDataset


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 32

OUTPUT_FILES = {
    "train": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_train.pt"
    ),
    "dev": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_dev.pt"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_test.pt"
    ),
}


# ---------------------------------------------------------
# MobileNetV2 frame feature extractor
# ---------------------------------------------------------

class MobileNetV2FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = tv_models.mobilenet_v2(
            weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V2
        )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        # x:
        # [B, 3, 224, 224]

        features = self.features(x)

        features = self.pool(features)

        features = features.flatten(1)

        # [B, 1280]

        return features


# ---------------------------------------------------------
# Extract one split
# ---------------------------------------------------------

@torch.no_grad()
def extract_split(
    model,
    manifest_path,
    output_path,
    split_name
):

    print("\n" + "=" * 70)
    print(f"VIDEO V2 FEATURE EXTRACTION - {split_name.upper()}")
    print("=" * 70)

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    df = load_manifest(manifest_path)

    print(f"Manifest: {manifest_path}")
    print(f"Samples: {len(df)}")

    # -----------------------------------------------------
    # Dataset
    #
    # We use the existing dataset only for frame loading
    # and uniform sampling.
    # -----------------------------------------------------

    dataset = VideoEmotionDataset(
        df,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Storage
    # -----------------------------------------------------

    all_features = []
    all_labels = []
    all_clip_ids = []

    # -----------------------------------------------------
    # Extraction
    # -----------------------------------------------------

    model.eval()

    for frames, labels, clip_ids in tqdm(
        loader,
        desc=f"Extracting {split_name}"
    ):

        # frames:
        # [B, T, 3, 224, 224]

        frames = frames.to(
            DEVICE,
            non_blocking=True
        )

        batch_size = frames.shape[0]
        num_frames = frames.shape[1]

        # -------------------------------------------------
        # Flatten frames
        # -------------------------------------------------

        frames_flat = frames.view(
            batch_size * num_frames,
            frames.shape[2],
            frames.shape[3],
            frames.shape[4]
        )

        # -------------------------------------------------
        # MobileNetV2
        # -------------------------------------------------

        frame_features = model(
            frames_flat
        )

        # [B*T, 1280]

        # -------------------------------------------------
        # Restore temporal dimension
        # -------------------------------------------------

        frame_features = frame_features.view(
            batch_size,
            num_frames,
            -1
        )

        # [B, T, 1280]

        all_features.append(
            frame_features.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        all_clip_ids.extend(
            list(clip_ids)
        )

    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    print("\nExtracted shape:")
    print(f"  Features: {features.shape}")
    print(f"  Labels:   {labels.shape}")
    print(f"  Clips:    {len(all_clip_ids)}")

    assert features.shape[0] == len(df)
    assert labels.shape[0] == len(df)
    assert len(all_clip_ids) == len(df)

    assert features.shape[1] == NUM_FRAMES
    assert features.shape[2] == 1280

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    cache = {
        "features": features,
        "labels": labels,
        "clip_ids": all_clip_ids,
        "feature_dim": 1280,
        "num_frames": NUM_FRAMES,
        "num_samples": len(df),
        "model": "MobileNetV2-Video-V2",
        "manifest": manifest_path,
    }

    torch.save(
        cache,
        output_path
    )

    print(f"\nSaved:")
    print(f"  {output_path}")

    print("\nFinal cache information:")
    print(f"  Samples:      {features.shape[0]}")
    print(f"  Frames/sample:{features.shape[1]}")
    print(f"  Feature dim:  {features.shape[2]}")

    return features, labels


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("VIDEO V2 FEATURE EXTRACTION")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        print("GPU not available. Using CPU.")

    print(f"Number of frames: {NUM_FRAMES}")
    print(f"Batch size: {BATCH_SIZE}")

    # -----------------------------------------------------
    # Create model
    # -----------------------------------------------------

    print("\nLoading MobileNetV2...")

    model = MobileNetV2FeatureExtractor().to(
        DEVICE
    )

    model.eval()

    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------

    train_manifest = os.path.join(
        PROJECT_ROOT,
        "data",
        "manifests",
        "train.csv"
    )

    dev_manifest = os.path.join(
        PROJECT_ROOT,
        "data",
        "manifests",
        "dev.csv"
    )

    test_manifest = os.path.join(
        PROJECT_ROOT,
        "data",
        "manifests",
        "test.csv"
    )

    # -----------------------------------------------------
    # Extract TRAIN
    # -----------------------------------------------------

    extract_split(
        model=model,
        manifest_path=train_manifest,
        output_path=OUTPUT_FILES["train"],
        split_name="train"
    )

    # -----------------------------------------------------
    # Extract DEV
    # -----------------------------------------------------

    extract_split(
        model=model,
        manifest_path=dev_manifest,
        output_path=OUTPUT_FILES["dev"],
        split_name="dev"
    )

    # -----------------------------------------------------
    # Extract TEST
    # -----------------------------------------------------

    extract_split(
        model=model,
        manifest_path=test_manifest,
        output_path=OUTPUT_FILES["test"],
        split_name="test"
    )

    # -----------------------------------------------------
    # Finished
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("VIDEO V2 FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print("\nCreated caches:")

    for split, path in OUTPUT_FILES.items():

        print(
            f"  {split:5s}: {path}"
        )

    print("\nExpected shapes:")

    print("  Train: [9988, 4, 1280]")
    print("  Dev:   [1108, 4, 1280]")
    print("  Test:  [2610, 4, 1280]")

    print("\nNext step:")
    print(
        "Verify the three Video V2 caches before training."
    )


if __name__ == "__main__":
    main()