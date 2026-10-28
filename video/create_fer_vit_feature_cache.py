import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# CONFIG
# ============================================================

from config import DEVICE

from video.video_dataset import VideoEmotionDataset

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = (
    "mo-thecreator/"
    "vit-Facial-Expression-Recognition"
)

BATCH_SIZE = 4

FRAMES_PER_CLIP = 4

TRAIN_MANIFEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "manifests",
    "train.csv"
)

DEV_MANIFEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "manifests",
    "dev.csv"
)

TEST_MANIFEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "manifests",
    "test.csv"
)

TRAIN_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fer_vit_feature_cache_train.pt"
)

DEV_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fer_vit_feature_cache_dev.pt"
)

TEST_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "fer_vit_feature_cache_test.pt"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("FULL MELD FER-ViT VIDEO FEATURE CACHE")
print("=" * 60)

print("Device :", DEVICE)
print("Model  :", MODEL_NAME)
print()


# ============================================================
# LOAD FER-ViT
# ============================================================

print("Loading FER-ViT...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

vit = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
).to(DEVICE)

vit.eval()

for parameter in vit.parameters():
    parameter.requires_grad = False

HIDDEN_SIZE = vit.config.hidden_size

print(
    "FER-ViT hidden size:",
    HIDDEN_SIZE
)

print()


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(
    3, 1, 1
)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(
    3, 1, 1
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    manifest_path,
    output_path,
    dataset_name
):

    print("=" * 60)
    print(f"Processing {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(
        manifest_path
    )

    print(
        "Samples:",
        len(df)
    )

    dataset = VideoEmotionDataset(
        df,
        augment=False
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    all_features = []

    all_labels = []

    # --------------------------------------------------------
    # Process batches
    # --------------------------------------------------------

    for frames, labels, clip_ids in tqdm(
        loader,
        desc=f"Extracting {dataset_name}"
    ):

        frames = frames.to(
            DEVICE
        )

        B, T, C, H, W = frames.shape

        # ----------------------------------------------------
        # Flatten frames
        # ----------------------------------------------------

        flat_frames = frames.reshape(
            B * T,
            C,
            H,
            W
        )

        # ----------------------------------------------------
        # Undo ImageNet normalization
        # ----------------------------------------------------

        flat_frames = (
            flat_frames * std
            + mean
        )

        flat_frames = (
            flat_frames
            .clamp(0, 1)
            .mul(255)
            .byte()
        )

        # ----------------------------------------------------
        # Convert tensors to PIL images
        # ----------------------------------------------------

        images = []

        for image_tensor in flat_frames:

            image_array = (
                image_tensor
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            images.append(
                Image.fromarray(
                    image_array
                )
            )

        # ----------------------------------------------------
        # FER-ViT preprocessing
        # ----------------------------------------------------

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Get ViT outputs
        # ----------------------------------------------------

        outputs = vit.vit(
            **inputs
        )

        # ----------------------------------------------------
        # CLS feature
        #
        # [B*T, 768]
        # ----------------------------------------------------

        cls_features = (
            outputs
            .last_hidden_state[:, 0]
        )

        # ----------------------------------------------------
        # Restore frame sequence
        #
        # [B, T, 768]
        # ----------------------------------------------------

        sequence_features = (
            cls_features.reshape(
                B,
                T,
                HIDDEN_SIZE
            )
        )

        # ----------------------------------------------------
        # Mean pool 4 frames
        #
        # [B, 768]
        # ----------------------------------------------------

        video_features = (
            sequence_features.mean(
                dim=1
            )
        )

        all_features.append(
            video_features.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

    # ========================================================
    # Combine
    # ========================================================

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # ========================================================
    # Save
    # ========================================================

    torch.save(
        {
            "features": features,
            "labels": labels
        },
        output_path
    )

    print()
    print("Cache saved:")
    print(output_path)

    print(
        "Feature shape:",
        features.shape
    )

    print(
        "Label shape:",
        labels.shape
    )

    print()


# ============================================================
# TRAIN
# ============================================================

extract_features(
    TRAIN_MANIFEST,
    TRAIN_CACHE,
    "train.csv"
)


# ============================================================
# DEV
# ============================================================

extract_features(
    DEV_MANIFEST,
    DEV_CACHE,
    "dev.csv"
)


# ============================================================
# TEST
# ============================================================

extract_features(
    TEST_MANIFEST,
    TEST_CACHE,
    "test.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("FULL FER-ViT VIDEO FEATURE CACHE COMPLETED")
print("=" * 60)

print()
print("Created:")

print(
    "data\\fer_vit_feature_cache_train.pt"
)

print(
    "data\\fer_vit_feature_cache_dev.pt"
)

print(
    "data\\fer_vit_feature_cache_test.pt"
)