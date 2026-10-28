# ============================================================
# EXTRACT VIDEO V3 512-D FEATURES FOR MULTIMODAL FUSION
# ============================================================
#
# Input:
#   data/video_v2_feature_cache_train.pt
#   data/video_v2_feature_cache_dev.pt
#   data/video_v2_feature_cache_test.pt
#
# Model:
#   Video V3
#   4 x 1280 MobileNetV2 frame features
#       ↓
#   BiGRU (256 hidden, bidirectional)
#       ↓
#   Attention pooling
#       ↓
#   512-D video representation
#
# Output:
#   data/video_v3_feature_cache_train.pt
#   data/video_v3_feature_cache_dev.pt
#   data/video_v3_feature_cache_test.pt
#
# These features will be used by V7 and V8 fusion.
# ============================================================

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 64

INPUT_DIM = 1280
GRU_HIDDEN = 256

NUM_CLASSES = 7

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "video_v3",
    "video_v3_best.pt",
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)

# ------------------------------------------------------------
# Existing Video V2 feature caches
# ------------------------------------------------------------

INPUT_CACHES = {
    "train": os.path.join(
        DATA_DIR,
        "video_v2_feature_cache_train.pt",
    ),

    "dev": os.path.join(
        DATA_DIR,
        "video_v2_feature_cache_dev.pt",
    ),

    "test": os.path.join(
        DATA_DIR,
        "video_v2_feature_cache_test.pt",
    ),
}

# ------------------------------------------------------------
# New Video V3 feature caches
# ------------------------------------------------------------

OUTPUT_CACHES = {
    "train": os.path.join(
        DATA_DIR,
        "video_v3_feature_cache_train.pt",
    ),

    "dev": os.path.join(
        DATA_DIR,
        "video_v3_feature_cache_dev.pt",
    ),

    "test": os.path.join(
        DATA_DIR,
        "video_v3_feature_cache_test.pt",
    ),
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# VIDEO V3 MODEL
# ============================================================

class VideoV3FeatureExtractor(nn.Module):
    """
    Exact feature-extraction portion of the trained
    Video V3 model.

    Input:
        [B, 4, 1280]

    Output:
        [B, 512]

    Architecture:

        4 x 1280 frame features
                ↓
              BiGRU
                ↓
          4 x 512 sequence
                ↓
            Attention
                ↓
              512-D
    """

    def __init__(
        self,
        input_dim=INPUT_DIM,
        gru_hidden=GRU_HIDDEN,
    ):

        super().__init__()

        # ----------------------------------------------------
        # BiGRU
        # ----------------------------------------------------

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = nn.Linear(
            2 * gru_hidden,
            1,
        )

    def forward(self, x):

        # x:
        # [B, 4, 1280]

        # ----------------------------------------------------
        # BiGRU
        # ----------------------------------------------------

        seq_out, _ = self.gru(x)

        # seq_out:
        # [B, 4, 512]

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        attention_scores = self.attention(
            seq_out
        ).squeeze(-1)

        # [B, 4]

        attention_weights = torch.softmax(
            attention_scores,
            dim=1,
        )

        # ----------------------------------------------------
        # Weighted pooling
        # ----------------------------------------------------

        pooled = torch.bmm(
            attention_weights.unsqueeze(1),
            seq_out,
        ).squeeze(1)

        # pooled:
        # [B, 512]

        return pooled


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(device):

    print("\n" + "=" * 70)
    print("LOADING VIDEO V3 CHECKPOINT")
    print("=" * 70)

    print("\nCheckpoint:")
    print(CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\n❌ Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Get state dictionary
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove possible DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):

            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[new_key] = value

    # --------------------------------------------------------
    # Create exact feature extractor
    # --------------------------------------------------------

    extractor = VideoV3FeatureExtractor().to(
        device
    )

    # --------------------------------------------------------
    # Extract only the layers required for
    # BiGRU + Attention.
    #
    # These MUST match the trained checkpoint.
    # --------------------------------------------------------

    required_keys = [
        "gru.weight_ih_l0",
        "gru.weight_hh_l0",
        "gru.bias_ih_l0",
        "gru.bias_hh_l0",

        "gru.weight_ih_l0_reverse",
        "gru.weight_hh_l0_reverse",
        "gru.bias_ih_l0_reverse",
        "gru.bias_hh_l0_reverse",

        "attention.weight",
        "attention.bias",
    ]

    missing = [
        key
        for key in required_keys
        if key not in cleaned_state_dict
    ]

    if missing:

        raise RuntimeError(
            "\n❌ Video V3 checkpoint is missing "
            "required feature-extractor keys:\n"
            + "\n".join(
                f"   {key}"
                for key in missing
            )
        )

    extractor_state = {
        key: cleaned_state_dict[key]
        for key in required_keys
    }

    # --------------------------------------------------------
    # Load exactly
    # --------------------------------------------------------

    extractor.load_state_dict(
        extractor_state,
        strict=True,
    )

    extractor.eval()

    print(
        "\n✅ Video V3 BiGRU + Attention "
        "weights loaded exactly."
    )

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:

            print(
                f"Checkpoint epoch: "
                f"{checkpoint['epoch']}"
            )

        if "val_macro_f1" in checkpoint:

            print(
                f"Checkpoint Dev Macro-F1: "
                f"{checkpoint['val_macro_f1']:.4f}"
            )

        if "val_accuracy" in checkpoint:

            print(
                f"Checkpoint Dev Accuracy: "
                f"{checkpoint['val_accuracy']:.4f}"
            )

    print(
        "\nFeature extractor output dimension: 512"
    )

    return extractor


# ============================================================
# LOAD INPUT CACHE
# ============================================================

def load_cache(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n❌ Feature cache not found:\n{path}"
        )

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(cache, dict):

        raise TypeError(
            f"\n❌ Expected dictionary cache, "
            f"got {type(cache)}"
        )

    if "features" not in cache:

        raise KeyError(
            "\n❌ Cache does not contain "
            "'features'."
        )

    features = cache["features"]

    if not torch.is_tensor(features):

        features = torch.tensor(
            features,
            dtype=torch.float32,
        )

    features = features.float()

    labels = cache.get(
        "labels",
        None,
    )

    clip_ids = cache.get(
        "clip_ids",
        None,
    )

    return (
        features,
        labels,
        clip_ids,
    )


# ============================================================
# EXTRACT FEATURES FOR ONE SPLIT
# ============================================================

@torch.no_grad()
def extract_split(
    split,
    extractor,
    device,
):

    print("\n" + "=" * 70)
    print(
        f"EXTRACTING VIDEO V3 FEATURES - "
        f"{split.upper()}"
    )
    print("=" * 70)

    input_path = INPUT_CACHES[split]
    output_path = OUTPUT_CACHES[split]

    print("\nInput:")
    print(input_path)

    print("\nOutput:")
    print(output_path)

    # --------------------------------------------------------
    # Load existing MobileNetV2 features
    # --------------------------------------------------------

    features, labels, clip_ids = load_cache(
        input_path
    )

    print(
        f"\nInput feature shape: "
        f"{tuple(features.shape)}"
    )

    if features.ndim != 3:

        raise ValueError(
            f"\n❌ Expected [N, T, D] features, "
            f"got {tuple(features.shape)}"
        )

    num_samples, num_frames, feature_dim = (
        features.shape
    )

    if num_frames != 4:

        raise ValueError(
            f"\n❌ Expected 4 frames, "
            f"got {num_frames}"
        )

    if feature_dim != INPUT_DIM:

        raise ValueError(
            f"\n❌ Expected feature dimension "
            f"{INPUT_DIM}, got {feature_dim}"
        )

    # --------------------------------------------------------
    # Create DataLoader
    # --------------------------------------------------------

    dataset = TensorDataset(
        features
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    extracted_features = []

    total_batches = len(loader)

    for batch_idx, batch in enumerate(
        loader,
        start=1,
    ):

        batch_features = batch[0].to(
            device,
            non_blocking=True,
        )

        video_features = extractor(
            batch_features
        )

        # Move to CPU
        video_features = (
            video_features
            .cpu()
            .float()
        )

        extracted_features.append(
            video_features
        )

        # Progress
        if (
            batch_idx == 1
            or batch_idx % 10 == 0
            or batch_idx == total_batches
        ):

            processed = min(
                batch_idx * BATCH_SIZE,
                num_samples,
            )

            print(
                f"Batch {batch_idx:>3}/"
                f"{total_batches:<3} | "
                f"Processed: "
                f"{processed}/{num_samples}"
            )

    # --------------------------------------------------------
    # Concatenate
    # --------------------------------------------------------

    new_features = torch.cat(
        extracted_features,
        dim=0,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    expected_shape = (
        num_samples,
        512,
    )

    if tuple(new_features.shape) != expected_shape:

        raise RuntimeError(
            f"\n❌ Unexpected output shape:\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(new_features.shape)}"
        )

    # --------------------------------------------------------
    # Build cache
    # --------------------------------------------------------

    output_cache = {
        "features": new_features,
        "labels": labels,
        "clip_ids": clip_ids,

        "feature_dim": 512,

        "num_samples": num_samples,

        "num_frames": 4,

        "model": "Video-V3-BiGRU-Attention",

        "source_features": (
            "MobileNetV2-4x1280"
        ),

        "checkpoint": CHECKPOINT_PATH,

        "split": split,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    torch.save(
        output_cache,
        output_path,
    )

    # --------------------------------------------------------
    # Verify saved file
    # --------------------------------------------------------

    if not os.path.exists(output_path):

        raise RuntimeError(
            f"\n❌ Output file was not created:\n"
            f"{output_path}"
        )

    file_size_mb = (
        os.path.getsize(output_path)
        / (1024 * 1024)
    )

    print(
        f"\n✅ {split.upper()} extraction complete."
    )

    print(
        f"Output shape: "
        f"{tuple(new_features.shape)}"
    )

    print(
        f"File size: "
        f"{file_size_mb:.2f} MB"
    )

    print(
        f"Saved to:\n{output_path}"
    )

    return new_features


# ============================================================
# VERIFY ALL OUTPUT CACHES
# ============================================================

def verify_outputs():

    print("\n" + "=" * 70)
    print("VERIFYING VIDEO V3 FEATURE CACHES")
    print("=" * 70)

    expected_samples = {
        "train": 9988,
        "dev": 1108,
        "test": 2610,
    }

    all_pass = True

    for split in [
        "train",
        "dev",
        "test",
    ]:

        path = OUTPUT_CACHES[split]

        print(
            f"\nChecking {split.upper()}:"
        )

        if not os.path.exists(path):

            print(
                "❌ File missing:"
            )

            print(path)

            all_pass = False

            continue

        cache = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        features = cache["features"]
        labels = cache["labels"]
        clip_ids = cache["clip_ids"]

        expected_n = expected_samples[
            split
        ]

        expected_shape = (
            expected_n,
            512,
        )

        print(
            f"Features: "
            f"{tuple(features.shape)}"
        )

        print(
            f"Labels: "
            f"{len(labels)}"
        )

        print(
            f"Clip IDs: "
            f"{len(clip_ids)}"
        )

        # ----------------------------------------------------
        # Shape
        # ----------------------------------------------------

        shape_ok = (
            tuple(features.shape)
            == expected_shape
        )

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        labels_ok = (
            len(labels)
            == expected_n
        )

        # ----------------------------------------------------
        # Clip IDs
        # ----------------------------------------------------

        clip_ids_ok = (
            len(clip_ids)
            == expected_n
        )

        # ----------------------------------------------------
        # NaN check
        # ----------------------------------------------------

        nan_ok = not torch.isnan(
            features
        ).any().item()

        # ----------------------------------------------------
        # Inf check
        # ----------------------------------------------------

        inf_ok = not torch.isinf(
            features
        ).any().item()

        # ----------------------------------------------------
        # Final
        # ----------------------------------------------------

        if (
            shape_ok
            and labels_ok
            and clip_ids_ok
            and nan_ok
            and inf_ok
        ):

            print(
                "✅ PASS"
            )

        else:

            print(
                "❌ FAIL"
            )

            if not shape_ok:
                print(
                    f"   Shape expected: "
                    f"{expected_shape}"
                )

            if not labels_ok:
                print(
                    "   Label count mismatch"
                )

            if not clip_ids_ok:
                print(
                    "   Clip ID count mismatch"
                )

            if not nan_ok:
                print(
                    "   NaN values detected"
                )

            if not inf_ok:
                print(
                    "   Inf values detected"
                )

            all_pass = False

    print("\n" + "-" * 70)

    if all_pass:

        print(
            "✅ ALL VIDEO V3 FEATURE CACHES PASSED."
        )

    else:

        print(
            "❌ ONE OR MORE CACHE CHECKS FAILED."
        )

    return all_pass


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("\n")
    print("=" * 80)
    print("VIDEO V3 → 512-D FEATURE EXTRACTION")
    print("=" * 80)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDEVICE INFORMATION")
    print("-" * 80)

    print(
        f"PyTorch:        "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA version:   "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU:            "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory:     "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"Device:         "
        f"{device}"
    )

    # --------------------------------------------------------
    # Check input files
    # --------------------------------------------------------

    print("\nINPUT CACHE CHECK")
    print("-" * 80)

    for split, path in INPUT_CACHES.items():

        exists = os.path.exists(path)

        print(
            f"{split:<6}: "
            f"{'FOUND' if exists else 'MISSING'}"
        )

        if not exists:

            raise FileNotFoundError(
                f"\n❌ Missing input cache:\n{path}"
            )

    # --------------------------------------------------------
    # Load Video V3 extractor
    # --------------------------------------------------------

    extractor = load_model(
        device
    )

    # --------------------------------------------------------
    # Extract all splits
    # --------------------------------------------------------

    for split in [
        "train",
        "dev",
        "test",
    ]:

        extract_split(
            split=split,
            extractor=extractor,
            device=device,
        )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    success = verify_outputs()

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 80)

    if success:

        print(
            "VIDEO V3 FEATURE EXTRACTION COMPLETE"
        )

        print(
            "\nReady for V7 improved feature fusion."
        )

        print(
            "\nGenerated:"
        )

        for split in [
            "train",
            "dev",
            "test",
        ]:

            print(
                f"  {OUTPUT_CACHES[split]}"
            )

        print(
            "\nExpected dimensions:"
        )

        print(
            "  Train: 9988 × 512"
        )

        print(
            "  Dev:   1108 × 512"
        )

        print(
            "  Test:  2610 × 512"
        )

    else:

        print(
            "VIDEO V3 FEATURE EXTRACTION "
            "COMPLETED WITH ERRORS"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()