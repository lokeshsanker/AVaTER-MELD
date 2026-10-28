import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# Add project root to Python path
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# ============================================================
# Import project modules
# ============================================================

from audio.model import AudioEmotionModel
from audio.audio_dataset import AudioEmotionDataset
from config import AUDIO_CKPT


# ============================================================
# Configuration
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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
    "audio_feature_cache_train.pt"
)

DEV_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "audio_feature_cache_dev.pt"
)

TEST_CACHE = os.path.join(
    PROJECT_ROOT,
    "data",
    "audio_feature_cache_test.pt"
)


# ============================================================
# Settings
# ============================================================

BATCH_SIZE = 8


# ============================================================
# Load trained Audio model
# ============================================================

print("=" * 60)
print("TRAINED AUDIO FEATURE CACHE CREATION")
print("=" * 60)

print(f"Device     : {DEVICE}")
print(
    f"Checkpoint : "
    f"{os.path.join(PROJECT_ROOT, AUDIO_CKPT)}"
)
print()

print("Creating AudioEmotionModel...")

model = AudioEmotionModel(
    freeze_encoder=True
)

checkpoint_path = os.path.join(
    PROJECT_ROOT,
    AUDIO_CKPT
)

checkpoint = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

# ------------------------------------------------------------
# Handle checkpoint format
# ------------------------------------------------------------

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(
    state_dict,
    strict=True
)

model.to(DEVICE)
model.eval()

print("Trained AudioEmotionModel loaded successfully.")
print()


# ============================================================
# Feature extraction
# ============================================================

@torch.no_grad()
def create_feature_cache(
    manifest_path,
    output_path,
    dataset_name
):

    print("=" * 60)
    print(f"Processing {dataset_name}")
    print("=" * 60)

    df = pd.read_csv(manifest_path)

    print(f"Samples: {len(df)}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = AudioEmotionDataset(df)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    features = []

    # --------------------------------------------------------
    # Extract Wav2Vec2 features
    # --------------------------------------------------------

    for input_values, labels, clip_ids in tqdm(
        loader,
        desc=f"Extracting {dataset_name}"
    ):

        input_values = input_values.to(DEVICE)

        # ----------------------------------------------------
        # Wav2Vec2 encoder
        # ----------------------------------------------------

        outputs = model.encoder(
            input_values
        )

        hidden = outputs.last_hidden_state

        # ----------------------------------------------------
        # Mean pooling over time
        #
        # hidden shape:
        # [Batch, Time, 768]
        #
        # Result:
        # [Batch, 768]
        # ----------------------------------------------------

        feature = hidden.mean(dim=1)

        features.append(
            feature.cpu()
        )

    # --------------------------------------------------------
    # Combine batches
    # --------------------------------------------------------

    features = torch.cat(
        features,
        dim=0
    )

    # --------------------------------------------------------
    # Save cache
    # --------------------------------------------------------

    torch.save(
        features,
        output_path
    )

    print()
    print("Cache saved:")
    print(output_path)
    print(f"Feature shape: {features.shape}")
    print(f"Feature dimension: {features.shape[1]}")
    print()


# ============================================================
# TRAIN
# ============================================================

create_feature_cache(
    TRAIN_MANIFEST,
    TRAIN_CACHE,
    "train.csv"
)


# ============================================================
# DEV
# ============================================================

create_feature_cache(
    DEV_MANIFEST,
    DEV_CACHE,
    "dev.csv"
)


# ============================================================
# TEST
# ============================================================

create_feature_cache(
    TEST_MANIFEST,
    TEST_CACHE,
    "test.csv"
)


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("TRAINED AUDIO FEATURE CACHE COMPLETED")
print("=" * 60)

print()
print("Files created:")
print("data\\audio_feature_cache_train.pt")
print("data\\audio_feature_cache_dev.pt")
print("data\\audio_feature_cache_test.pt")