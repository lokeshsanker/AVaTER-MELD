# ============================================================
# V7 - IMPROVED FEATURE-LEVEL MULTIMODAL FUSION
# ============================================================
#
# Modalities:
#   Text V2  -> 768-D
#   Audio V4 -> 768-D
#   Video V3 -> 512-D
#
# Total fused dimension:
#   768 + 768 + 512 = 2048
#
# Architecture:
#   LayerNorm(2048)
#       ↓
#   Linear 2048 -> 512
#       ↓
#   GELU
#       ↓
#   Dropout(0.30)
#       ↓
#   Linear 512 -> 128
#       ↓
#   GELU
#       ↓
#   Dropout(0.20)
#       ↓
#   Linear 128 -> 7
#
# Training:
#   AdamW
#   LR = 3e-4
#   Weight decay = 1e-4
#   Label smoothing = 0.05
#   ReduceLROnPlateau
#   Early stopping
#   Gradient clipping
#   AMP on CUDA
#
# Checkpoint:
#   checkpoints/v7_improved_fusion/
#       v7_improved_fusion_best.pt
#
# IMPORTANT:
#   This script does NOT modify V6.
# ============================================================

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

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
# CONFIG
# ============================================================

SEED = 42

NUM_CLASSES = 7

EMOTION_LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

# ------------------------------------------------------------
# Training configuration
# ------------------------------------------------------------

BATCH_SIZE = 64

MAX_EPOCHS = 30

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05

EARLY_STOP_PATIENCE = 5

GRAD_CLIP = 1.0

LR_PATIENCE = 2

LR_FACTOR = 0.5

MIN_LR = 1e-6

# ------------------------------------------------------------
# Feature dimensions
# ------------------------------------------------------------

TEXT_DIM = 768

AUDIO_DIM = 768

VIDEO_DIM = 512

FUSED_DIM = (
    TEXT_DIM
    + AUDIO_DIM
    + VIDEO_DIM
)

# 2048
assert FUSED_DIM == 2048


# ============================================================
# FEATURE CACHE PATHS
# ============================================================

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
)

TEXT_CACHES = {
    "train": os.path.join(
        DATA_DIR,
        "text_v2_feature_cache_train.pt",
    ),

    "dev": os.path.join(
        DATA_DIR,
        "text_v2_feature_cache_dev.pt",
    ),

    "test": os.path.join(
        DATA_DIR,
        "text_v2_feature_cache_test.pt",
    ),
}

AUDIO_CACHES = {
    "train": os.path.join(
        DATA_DIR,
        "audio_v4_feature_cache_train.pt",
    ),

    "dev": os.path.join(
        DATA_DIR,
        "audio_v4_feature_cache_dev.pt",
    ),

    "test": os.path.join(
        DATA_DIR,
        "audio_v4_feature_cache_test.pt",
    ),
}

VIDEO_CACHES = {
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
# CHECKPOINT / RESULTS
# ============================================================

CHECKPOINT_DIR = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "v7_improved_fusion",
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "v7_improved_fusion_best.pt",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "v7_improved_fusion_results.txt",
)


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
# LOAD FEATURE CACHE
# ============================================================

def load_feature_cache(path, name):

    print(
        f"\nLoading {name}:"
    )

    print(path)

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\n❌ Missing feature cache:\n{path}"
        )

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(cache, dict):

        if "features" not in cache:

            raise KeyError(
                f"\n❌ {name} cache does not "
                f"contain 'features'."
            )

        features = cache["features"]

        labels = cache.get(
            "labels",
            None,
        )

        clip_ids = cache.get(
            "clip_ids",
            None,
        )

    elif torch.is_tensor(cache):

        features = cache

        labels = None

        clip_ids = None

    else:

        raise TypeError(
            f"\n❌ Unsupported cache type "
            f"for {name}: {type(cache)}"
        )

    if not torch.is_tensor(features):

        features = torch.tensor(
            features,
            dtype=torch.float32,
        )

    features = features.float()

    print(
        f"{name} feature shape: "
        f"{tuple(features.shape)}"
    )

    if labels is not None:

        if not torch.is_tensor(labels):

            labels = torch.tensor(
                labels,
                dtype=torch.long,
            )

        labels = labels.long()

        print(
            f"{name} labels: "
            f"{len(labels)}"
        )

    if clip_ids is not None:

        print(
            f"{name} clip IDs: "
            f"{len(clip_ids)}"
        )

    return (
        features,
        labels,
        clip_ids,
    )


# ============================================================
# VALIDATE MODALITY CACHE
# ============================================================

def validate_features(
    features,
    expected_dim,
    expected_samples,
    name,
):

    if features.ndim != 2:

        raise ValueError(
            f"\n❌ {name} must be 2-D.\n"
            f"Got: {tuple(features.shape)}"
        )

    expected_shape = (
        expected_samples,
        expected_dim,
    )

    if tuple(features.shape) != expected_shape:

        raise ValueError(
            f"\n❌ {name} shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {tuple(features.shape)}"
        )

    if torch.isnan(features).any():

        raise ValueError(
            f"\n❌ NaN values found in {name}."
        )

    if torch.isinf(features).any():

        raise ValueError(
            f"\n❌ Inf values found in {name}."
        )

    print(
        f"✅ {name} validation passed: "
        f"{tuple(features.shape)}"
    )


# ============================================================
# CHECK LABELS / CLIP ALIGNMENT
# ============================================================

def validate_alignment(
    text_labels,
    audio_labels,
    video_labels,
    text_clip_ids,
    audio_clip_ids,
    video_clip_ids,
    split,
):

    print(
        f"\nChecking {split.upper()} alignment..."
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if text_labels is None:

        raise ValueError(
            f"\n❌ Text {split} cache has no labels."
        )

    if audio_labels is None:

        raise ValueError(
            f"\n❌ Audio {split} cache has no labels."
        )

    if video_labels is None:

        raise ValueError(
            f"\n❌ Video {split} cache has no labels."
        )

    text_labels = text_labels.cpu()

    audio_labels = audio_labels.cpu()

    video_labels = video_labels.cpu()

    label_text_audio = torch.equal(
        text_labels,
        audio_labels,
    )

    label_text_video = torch.equal(
        text_labels,
        video_labels,
    )

    print(
        f"Text == Audio labels: "
        f"{label_text_audio}"
    )

    print(
        f"Text == Video labels: "
        f"{label_text_video}"
    )

    if not label_text_audio:

        raise ValueError(
            f"\n❌ Text/Audio label mismatch "
            f"in {split}."
        )

    if not label_text_video:

        raise ValueError(
            f"\n❌ Text/Video label mismatch "
            f"in {split}."
        )

    # --------------------------------------------------------
    # Clip IDs
    # --------------------------------------------------------

    if (
        text_clip_ids is not None
        and audio_clip_ids is not None
        and video_clip_ids is not None
    ):

        text_clip_ids = list(
            text_clip_ids
        )

        audio_clip_ids = list(
            audio_clip_ids
        )

        video_clip_ids = list(
            video_clip_ids
        )

        clip_text_audio = (
            text_clip_ids
            == audio_clip_ids
        )

        clip_text_video = (
            text_clip_ids
            == video_clip_ids
        )

        print(
            f"Text == Audio clip IDs: "
            f"{clip_text_audio}"
        )

        print(
            f"Text == Video clip IDs: "
            f"{clip_text_video}"
        )

        if not clip_text_audio:

            raise ValueError(
                f"\n❌ Text/Audio clip ID mismatch "
                f"in {split}."
            )

        if not clip_text_video:

            raise ValueError(
                f"\n❌ Text/Video clip ID mismatch "
                f"in {split}."
            )

    print(
        f"✅ {split.upper()} alignment passed."
    )


# ============================================================
# MODEL
# ============================================================

class V7ImprovedFusion(nn.Module):

    def __init__(
        self,
        input_dim=FUSED_DIM,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(
                input_dim
            ),

            nn.Linear(
                input_dim,
                512,
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                512,
                128,
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# LOAD ALL MODALITIES
# ============================================================

def load_all_data():

    print("\n" + "=" * 80)
    print("LOADING V7 IMPROVED MULTIMODAL FEATURES")
    print("=" * 80)

    splits = {
        "train": 9988,
        "dev": 1108,
        "test": 2610,
    }

    data = {}

    for split, expected_samples in splits.items():

        print("\n" + "-" * 80)

        print(
            f"LOADING {split.upper()} "
            f"({expected_samples} samples)"
        )

        # ----------------------------------------------------
        # Text V2
        # ----------------------------------------------------

        (
            text_features,
            text_labels,
            text_clip_ids,
        ) = load_feature_cache(
            TEXT_CACHES[split],
            f"Text V2 {split}",
        )

        # ----------------------------------------------------
        # Audio V4
        # ----------------------------------------------------

        (
            audio_features,
            audio_labels,
            audio_clip_ids,
        ) = load_feature_cache(
            AUDIO_CACHES[split],
            f"Audio V4 {split}",
        )

        # ----------------------------------------------------
        # Video V3
        # ----------------------------------------------------

        (
            video_features,
            video_labels,
            video_clip_ids,
        ) = load_feature_cache(
            VIDEO_CACHES[split],
            f"Video V3 {split}",
        )

        # ----------------------------------------------------
        # Validate dimensions
        # ----------------------------------------------------

        validate_features(
            text_features,
            TEXT_DIM,
            expected_samples,
            f"Text V2 {split}",
        )

        validate_features(
            audio_features,
            AUDIO_DIM,
            expected_samples,
            f"Audio V4 {split}",
        )

        validate_features(
            video_features,
            VIDEO_DIM,
            expected_samples,
            f"Video V3 {split}",
        )

        # ----------------------------------------------------
        # Validate alignment
        # ----------------------------------------------------

        validate_alignment(
            text_labels,
            audio_labels,
            video_labels,
            text_clip_ids,
            audio_clip_ids,
            video_clip_ids,
            split,
        )

        # ----------------------------------------------------
        # Final labels
        # ----------------------------------------------------

        labels = text_labels

        # ----------------------------------------------------
        # Concatenate
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                text_features,
                audio_features,
                video_features,
            ],
            dim=1,
        )

        expected_fused_shape = (
            expected_samples,
            FUSED_DIM,
        )

        if tuple(
            fused_features.shape
        ) != expected_fused_shape:

            raise ValueError(
                f"\n❌ Fused {split} shape mismatch.\n"
                f"Expected: {expected_fused_shape}\n"
                f"Got:      {tuple(fused_features.shape)}"
            )

        print(
            f"\n✅ {split.upper()} fused feature shape: "
            f"{tuple(fused_features.shape)}"
        )

        data[split] = {
            "features": fused_features,
            "labels": labels,
        }

    return data


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(
    labels,
    name,
):

    print(
        f"\n{name} class distribution:"
    )

    labels_np = labels.cpu().numpy()

    counts = np.bincount(
        labels_np,
        minlength=NUM_CLASSES,
    )

    for idx, label in enumerate(
        EMOTION_LABELS
    ):

        print(
            f"  {label:<10}: "
            f"{counts[idx]}"
        )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    use_amp,
):

    model.train()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels,
            )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        if use_amp:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1,
    )


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    all_labels = []

    all_predictions = []

    for features, labels in loader:

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels,
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_labels.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        total_loss
        / len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        np.array(all_labels),
        np.array(all_predictions),
    )


# ============================================================
# SAVE BEST CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    train_metrics,
    dev_metrics,
):

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True,
    )

    checkpoint = {

        "version": "V7",

        "model_name": (
            "V7 Improved Feature-Level "
            "Multimodal Fusion"
        ),

        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "train_loss":
            train_metrics[0],

        "train_accuracy":
            train_metrics[1],

        "train_macro_f1":
            train_metrics[2],

        "train_weighted_f1":
            train_metrics[3],

        "val_loss":
            dev_metrics[0],

        "val_accuracy":
            dev_metrics[1],

        "val_macro_f1":
            dev_metrics[2],

        "val_weighted_f1":
            dev_metrics[3],

        "text_dim":
            TEXT_DIM,

        "audio_dim":
            AUDIO_DIM,

        "video_dim":
            VIDEO_DIM,

        "fused_dim":
            FUSED_DIM,

        "emotion_labels":
            EMOTION_LABELS,

        "seed":
            SEED,
    }

    torch.save(
        checkpoint,
        CHECKPOINT_PATH,
    )

    print(
        f"\n💾 Best V7 checkpoint saved:"
    )

    print(
        CHECKPOINT_PATH
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    set_seed(SEED)

    print("\n")
    print("=" * 80)
    print("V7 - IMPROVED FEATURE-LEVEL MULTIMODAL FUSION")
    print("=" * 80)

    print("\nVERSION")
    print("-" * 80)

    print(
        "V7 = Text V2 + Audio V4 + Video V3"
    )

    print(
        f"\nText dimension:  {TEXT_DIM}"
    )

    print(
        f"Audio dimension: {AUDIO_DIM}"
    )

    print(
        f"Video dimension: {VIDEO_DIM}"
    )

    print(
        f"Fused dimension: {FUSED_DIM}"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDEVICE")
    print("-" * 80)

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_all_data()

    train_features = data[
        "train"
    ]["features"]

    train_labels = data[
        "train"
    ]["labels"]

    dev_features = data[
        "dev"
    ]["features"]

    dev_labels = data[
        "dev"
    ]["labels"]

    test_features = data[
        "test"
    ]["features"]

    test_labels = data[
        "test"
    ]["labels"]

    # --------------------------------------------------------
    # Print class distributions
    # --------------------------------------------------------

    print_class_distribution(
        train_labels,
        "TRAIN",
    )

    print_class_distribution(
        dev_labels,
        "DEV",
    )

    print_class_distribution(
        test_labels,
        "TEST",
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        train_features,
        train_labels,
    )

    dev_dataset = TensorDataset(
        dev_features,
        dev_labels,
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDATA")
    print("-" * 80)

    print(
        f"Train samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Dev samples:   "
        f"{len(dev_dataset)}"
    )

    print(
        f"Test samples:  "
        f"{len(test_dataset)}"
    )

    print(
        f"Batch size:    "
        f"{BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = V7ImprovedFusion().to(
        device
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nMODEL")
    print("-" * 80)

    print(
        "Architecture:"
    )

    print(
        "LayerNorm(2048)"
    )

    print(
        "Linear(2048 → 512)"
    )

    print(
        "GELU"
    )

    print(
        "Dropout(0.30)"
    )

    print(
        "Linear(512 → 128)"
    )

    print(
        "GELU"
    )

    print(
        "Dropout(0.20)"
    )

    print(
        "Linear(128 → 7)"
    )

    print(
        f"\nTotal parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    use_amp = torch.cuda.is_available()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print("\nTRAINING CONFIGURATION")
    print("-" * 80)

    print(
        f"Batch size:        {BATCH_SIZE}"
    )

    print(
        f"Max epochs:        {MAX_EPOCHS}"
    )

    print(
        f"Learning rate:     {LEARNING_RATE}"
    )

    print(
        f"Weight decay:      {WEIGHT_DECAY}"
    )

    print(
        f"Label smoothing:   {LABEL_SMOOTHING}"
    )

    print(
        f"Early stop:        {EARLY_STOP_PATIENCE}"
    )

    print(
        f"Gradient clip:     {GRAD_CLIP}"
    )

    print(
        f"AMP enabled:       {use_amp}"
    )

    print(
        f"LR scheduler:      ReduceLROnPlateau"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STARTING V7 TRAINING")
    print("=" * 80)

    best_macro_f1 = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            train_macro_f1,
            train_weighted_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
        )

        # ----------------------------------------------------
        # Dev
        # ----------------------------------------------------

        (
            dev_loss,
            dev_accuracy,
            dev_macro_f1,
            dev_weighted_f1,
            _,
            _,
        ) = evaluate(
            model,
            dev_loader,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(
            dev_macro_f1
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        history.append({

            "epoch": epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "train_macro_f1":
                train_macro_f1,

            "train_weighted_f1":
                train_weighted_f1,

            "dev_loss":
                dev_loss,

            "dev_accuracy":
                dev_accuracy,

            "dev_macro_f1":
                dev_macro_f1,

            "dev_weighted_f1":
                dev_weighted_f1,

            "lr":
                current_lr,
        })

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Acc:  "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted-F1: "
            f"{train_weighted_f1:.4f}"
        )

        print(
            f"Dev Loss: "
            f"{dev_loss:.4f}"
        )

        print(
            f"Dev Acc: "
            f"{dev_accuracy:.4f}"
        )

        print(
            f"Dev Macro-F1: "
            f"{dev_macro_f1:.4f}"
        )

        print(
            f"Dev Weighted-F1: "
            f"{dev_weighted_f1:.4f}"
        )

        print(
            f"Learning Rate: "
            f"{current_lr:.7f}"
        )

        # ----------------------------------------------------
        # Best model based on Macro-F1
        # ----------------------------------------------------

        if dev_macro_f1 > best_macro_f1:

            best_macro_f1 = (
                dev_macro_f1
            )

            best_epoch = epoch

            patience_counter = 0

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                (
                    train_loss,
                    train_accuracy,
                    train_macro_f1,
                    train_weighted_f1,
                ),
                (
                    dev_loss,
                    dev_accuracy,
                    dev_macro_f1,
                    dev_weighted_f1,
                ),
            )

            print(
                f"⭐ NEW BEST MODEL"
            )

            print(
                f"Best Dev Macro-F1: "
                f"{best_macro_f1:.4f}"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement: "
                f"{patience_counter}/"
                f"{EARLY_STOP_PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            patience_counter
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "\n⏹️ Early stopping triggered."
            )

            break

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("LOADING BEST V7 MODEL")
    print("=" * 80)

    best_checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.eval()

    print(
        f"\nBest epoch: "
        f"{best_checkpoint['epoch']}"
    )

    print(
        f"Best Dev Accuracy: "
        f"{best_checkpoint['val_accuracy']:.4f}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_checkpoint['val_macro_f1']:.4f}"
    )

    print(
        f"Best Dev Weighted-F1: "
        f"{best_checkpoint['val_weighted_f1']:.4f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 80)
    print("V7 FINAL TEST EVALUATION")
    print("=" * 80)

    (
        test_loss,
        test_accuracy,
        test_macro_f1,
        test_weighted_f1,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        f"\nTest Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Macro-F1: "
        f"{test_macro_f1 * 100:.2f}%"
    )

    print(
        f"Test Weighted-F1: "
        f"{test_weighted_f1 * 100:.2f}%"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )

    print("\n" + "=" * 80)
    print("V7 CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    print("\n" + "=" * 80)
    print("V7 CONFUSION MATRIX")
    print("=" * 80)

    print(
        "\nRows = Actual"
    )

    print(
        "Columns = Predicted\n"
    )

    print(
        f"{'':12}",
        end="",
    )

    for label in EMOTION_LABELS:

        print(
            f"{label:>10}",
            end="",
        )

    print()

    for i, label in enumerate(
        EMOTION_LABELS
    ):

        print(
            f"{label:<12}",
            end="",
        )

        for j in range(
            NUM_CLASSES
        ):

            print(
                f"{cm[i, j]:>10}",
                end="",
            )

        print()

    # ========================================================
    # PREDICTION DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 80)
    print("V7 PREDICTION DISTRIBUTION")
    print("=" * 80)

    prediction_counts = np.bincount(
        y_pred,
        minlength=NUM_CLASSES,
    )

    detected_classes = 0

    for idx, label in enumerate(
        EMOTION_LABELS
    ):

        count = prediction_counts[
            idx
        ]

        percentage = (
            count
            / len(y_pred)
            * 100
        )

        if count > 0:

            detected_classes += 1

        print(
            f"{label:<12}: "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"\nClasses detected: "
        f"{detected_classes}/"
        f"{NUM_CLASSES}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=" * 80
            + "\n"
        )

        f.write(
            "V7 - IMPROVED FEATURE-LEVEL "
            "MULTIMODAL FUSION\n"
        )

        f.write(
            "=" * 80
            + "\n\n"
        )

        f.write(
            "MODEL CONFIGURATION\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            "Text:  Text V2 - 768-D\n"
        )

        f.write(
            "Audio: Audio V4 - 768-D\n"
        )

        f.write(
            "Video: Video V3 - 512-D\n"
        )

        f.write(
            "Fusion: Concatenation\n"
        )

        f.write(
            "Fused dimension: 2048-D\n"
        )

        f.write(
            "Architecture: "
            "LayerNorm → 512 → 128 → 7\n\n"
        )

        f.write(
            "TRAINING RESULTS\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Best epoch: "
            f"{best_checkpoint['epoch']}\n"
        )

        f.write(
            f"Best Dev Accuracy: "
            f"{best_checkpoint['val_accuracy']:.6f}\n"
        )

        f.write(
            f"Best Dev Macro-F1: "
            f"{best_checkpoint['val_macro_f1']:.6f}\n"
        )

        f.write(
            f"Best Dev Weighted-F1: "
            f"{best_checkpoint['val_weighted_f1']:.6f}\n\n"
        )

        f.write(
            "TEST RESULTS\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            f"Test Loss: "
            f"{test_loss:.6f}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{test_accuracy:.6f}\n"
        )

        f.write(
            f"Test Macro-F1: "
            f"{test_macro_f1:.6f}\n"
        )

        f.write(
            f"Test Weighted-F1: "
            f"{test_weighted_f1:.6f}\n\n"
        )

        f.write(
            "CLASSIFICATION REPORT\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            report
        )

        f.write(
            "\nCONFUSION MATRIX\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        f.write(
            "Rows = Actual, "
            "Columns = Predicted\n\n"
        )

        for i, label in enumerate(
            EMOTION_LABELS
        ):

            f.write(
                f"{label:<12}"
            )

            for j in range(
                NUM_CLASSES
            ):

                f.write(
                    f"{cm[i, j]:>10}"
                )

            f.write("\n")

        f.write(
            "\nPREDICTION DISTRIBUTION\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        for idx, label in enumerate(
            EMOTION_LABELS
        ):

            count = prediction_counts[
                idx
            ]

            percentage = (
                count
                / len(y_pred)
                * 100
            )

            f.write(
                f"{label:<12}: "
                f"{count:>5} "
                f"({percentage:6.2f}%)\n"
            )

        f.write(
            f"\nClasses detected: "
            f"{detected_classes}/"
            f"{NUM_CLASSES}\n"
        )

        f.write(
            "\nSANITY CHECK\n"
        )

        f.write(
            "-" * 80
            + "\n"
        )

        if (
            len(y_true)
            == len(y_pred)
            == 2610
        ):

            f.write(
                "PASS - 2610 test predictions generated.\n"
            )

        else:

            f.write(
                f"WARNING - Expected 2610 "
                f"predictions, got "
                f"{len(y_pred)}.\n"
            )

        f.write(
            "\n" + "=" * 80
            + "\n"
        )

    # ========================================================
    # FINAL
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "V7 TRAINING AND TEST EVALUATION COMPLETE"
    )

    print("=" * 80)

    print(
        f"\nBest checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print(
        f"\nResults:"
    )

    print(
        RESULT_FILE
    )

    print(
        "\nV7 Summary:"
    )

    print(
        f"Accuracy:    "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1:    "
        f"{test_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1: "
        f"{test_weighted_f1 * 100:.2f}%"
    )

    print(
        f"Classes:     "
        f"{detected_classes}/7"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()