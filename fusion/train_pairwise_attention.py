# =============================================================================
# V5 FEATURE-LEVEL PAIRWISE CROSS-MODAL ATTENTION
# =============================================================================
#
# Project:
#   MULTIMODAL EMOTION RECOGNITION USING TEXT, AUDIO AND VIDEO
#
# Dataset:
#   MELD
#
# Run ID:
#   RUN_006
#
# Version:
#   V5
#
# PURPOSE
# -------
# V4 used simple feature concatenation:
#
#   Text 768
#   Audio 768
#   Video 768
#        |
#        v
#   Concatenate = 2304
#        |
#        v
#   MLP classifier
#
# V5 explicitly models relationships between modalities using
# feature-level bidirectional cross-modal attention.
#
# Pairwise interactions:
#
#   Text  -> Audio
#   Audio -> Text
#
#   Text  -> Video
#   Video -> Text
#
#   Audio -> Video
#   Video -> Audio
#
# Each modality starts as a 768-dimensional feature vector.
# Each vector is projected into multiple learned feature tokens.
#
# IMPORTANT:
# ----------
# This is FEATURE-LEVEL cross-modal attention.
# It is NOT temporal cross-modal attention because our cached
# representations are already pooled 768-dimensional vectors.
#
# Video cache:
#   4 FER-ViT frame representations
#       |
#       v
#   Mean pooling
#       |
#       v
#   768-dimensional video representation
#
# Final classifier predicts 7 MELD emotions:
#
#   0 = neutral
#   1 = joy
#   2 = sadness
#   3 = anger
#   4 = fear
#   5 = disgust
#   6 = surprise
#
# =============================================================================


import os
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

# IMPORTANT:
# Change this import ONLY if your logger file has a different module name.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.experiment_logger import log_run


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_ID = "RUN_006"
VERSION = "V5"

NUM_CLASSES = 7
FEATURE_DIM = 768

# Feature-token configuration
NUM_TOKENS = 6
TOKEN_DIM = 128

# Attention
NUM_HEADS = 4
ATTENTION_DROPOUT = 0.10

# Classifier
CLASSIFIER_DROPOUT = 0.20

# Training
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

EARLY_STOPPING_PATIENCE = 5
GRADIENT_CLIP = 1.0

SEED = 42

# Paths
TEXT_TRAIN = "data/text_feature_cache_train.pt"
TEXT_DEV = "data/text_feature_cache_dev.pt"
TEXT_TEST = "data/text_feature_cache_test.pt"

AUDIO_TRAIN = "data/audio_feature_cache_train.pt"
AUDIO_DEV = "data/audio_feature_cache_dev.pt"
AUDIO_TEST = "data/audio_feature_cache_test.pt"

VIDEO_TRAIN = "data/fer_vit_feature_cache_train.pt"
VIDEO_DEV = "data/fer_vit_feature_cache_dev.pt"
VIDEO_TEST = "data/fer_vit_feature_cache_test.pt"

CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"

BEST_CHECKPOINT = os.path.join(
    CHECKPOINT_DIR,
    "v5_pairwise_attention_best.pt"
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "v5_pairwise_attention_results.txt"
)


# =============================================================================
# EMOTION LABELS
# =============================================================================

EMOTION_LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

LABEL_TO_ID = {
    "neutral": 0,
    "joy": 1,
    "sadness": 2,
    "anger": 3,
    "fear": 4,
    "disgust": 5,
    "surprise": 6,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DEVICE CHECK
# =============================================================================

def get_device():

    print("=" * 80)
    print("V5 FEATURE-LEVEL PAIRWISE CROSS-MODAL ATTENTION")
    print("=" * 80)

    print(f"Run ID        : {RUN_ID}")
    print(f"Version       : {VERSION}")

    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version  : {torch.version.cuda}")

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory    : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        device = torch.device("cpu")

        print("GPU           : NOT AVAILABLE")

        print()
        print("WARNING:")
        print("CUDA is not available.")
        print("This V5 experiment is intended to run on the NVIDIA GPU.")
        print("Do NOT intentionally train this experiment on CPU.")
        print()

    print(f"Device        : {device}")

    print("=" * 80)

    return device


# =============================================================================
# CACHE LOADING
# =============================================================================

def load_cache(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nCache file not found:\n{path}\n"
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )


# =============================================================================
# FEATURE EXTRACTION FROM CACHE
# =============================================================================

def get_features(cache, name):

    # Text and video caches are dictionaries.
    if isinstance(cache, dict):

        if "features" not in cache:

            raise KeyError(
                f"{name}: dictionary cache does not contain 'features'."
            )

        features = cache["features"]

    # Audio cache is a plain tensor.
    elif torch.is_tensor(cache):

        features = cache

    else:

        raise TypeError(
            f"{name}: unsupported cache type: {type(cache)}"
        )

    if not torch.is_tensor(features):

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

    features = features.float()

    print(
        f"{name}: "
        f"shape={tuple(features.shape)}, "
        f"dtype={features.dtype}"
    )

    if features.ndim != 2:

        raise ValueError(
            f"{name}: expected 2D tensor [samples, features], "
            f"got {features.shape}"
        )

    if features.shape[1] != FEATURE_DIM:

        raise ValueError(
            f"{name}: expected feature dimension "
            f"{FEATURE_DIM}, got {features.shape[1]}"
        )

    return features


# =============================================================================
# LABEL EXTRACTION
# =============================================================================

def get_labels(cache, name):

    if not isinstance(cache, dict):

        return None

    return cache.get("labels", None)


# =============================================================================
# LABEL NORMALIZATION
# =============================================================================

def normalize_labels(labels, name):

    """
    Convert labels into numeric class IDs.

    Text cache:
        labels are emotion strings such as:
        neutral, joy, sadness, anger, fear, disgust, surprise

    Video cache:
        labels are already numeric tensors.

    Final mapping:
        0 = neutral
        1 = joy
        2 = sadness
        3 = anger
        4 = fear
        5 = disgust
        6 = surprise
    """

    if labels is None:

        return None

    # -------------------------------------------------------------------------
    # Tensor labels
    # -------------------------------------------------------------------------

    if torch.is_tensor(labels):

        labels = labels.detach().cpu()

        if labels.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):

            labels = labels.long()

        else:

            raise TypeError(
                f"{name}: tensor labels have unsupported dtype "
                f"{labels.dtype}"
            )

    # -------------------------------------------------------------------------
    # Python list labels
    # -------------------------------------------------------------------------

    elif isinstance(labels, list):

        if len(labels) == 0:

            raise ValueError(
                f"{name}: label list is empty."
            )

        first = labels[0]

        # Text cache contains strings.
        if isinstance(first, str):

            converted = []

            for label in labels:

                label = label.strip().lower()

                if label not in LABEL_TO_ID:

                    raise ValueError(
                        f"{name}: unknown emotion label: {label}"
                    )

                converted.append(
                    LABEL_TO_ID[label]
                )

            labels = torch.tensor(
                converted,
                dtype=torch.long
            )

        # Already numeric
        else:

            labels = torch.tensor(
                labels,
                dtype=torch.long
            )

    else:

        raise TypeError(
            f"{name}: unsupported label type: {type(labels)}"
        )

    # Make sure labels are 1D.
    if labels.ndim != 1:

        labels = labels.reshape(-1)

    # Validate class range.
    if labels.numel() > 0:

        min_label = labels.min().item()
        max_label = labels.max().item()

        if min_label < 0 or max_label >= NUM_CLASSES:

            raise ValueError(
                f"{name}: labels outside valid range "
                f"0-{NUM_CLASSES - 1}: "
                f"min={min_label}, max={max_label}"
            )

        print(
            f"{name} labels: "
            f"shape={tuple(labels.shape)}, "
            f"dtype={labels.dtype}, "
            f"min={min_label}, "
            f"max={max_label}"
        )

    return labels


# =============================================================================
# LABEL DISTRIBUTION
# =============================================================================

def print_distribution(labels, name):

    print()
    print(f"{name} class distribution")
    print("-" * 50)

    counts = torch.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    for idx, emotion in enumerate(EMOTION_LABELS):

        print(
            f"{emotion:<10}: {counts[idx].item()}"
        )


# =============================================================================
# LABEL ALIGNMENT CHECK
# =============================================================================

def verify_label_alignment(
    text_labels,
    video_labels,
    split_name
):

    if text_labels is None:

        raise ValueError(
            f"{split_name}: text labels are missing."
        )

    if video_labels is None:

        raise ValueError(
            f"{split_name}: video labels are missing."
        )

    if len(text_labels) != len(video_labels):

        raise ValueError(
            f"{split_name}: label length mismatch. "
            f"Text={len(text_labels)}, "
            f"Video={len(video_labels)}"
        )

    matches = torch.equal(
        text_labels,
        video_labels
    )

    print(
        f"{split_name} label alignment: "
        f"{matches}"
    )

    if not matches:

        mismatch_indices = torch.where(
            text_labels != video_labels
        )[0]

        print(
            f"Number of mismatches: "
            f"{len(mismatch_indices)}"
        )

        if len(mismatch_indices) > 0:

            first = mismatch_indices[:10]

            print(
                "First mismatches:"
            )

            for idx in first:

                i = idx.item()

                print(
                    f"  index={i}, "
                    f"text={text_labels[i].item()}, "
                    f"video={video_labels[i].item()}"
                )

        raise ValueError(
            f"{split_name}: text/video labels do not match."
        )


# =============================================================================
# MODEL
# =============================================================================

class FeatureTokenProjection(nn.Module):

    """
    Converts one 768-D modality feature into several learned feature tokens.

    Input:
        [batch, 768]

    Output:
        [batch, NUM_TOKENS, TOKEN_DIM]

    Example:
        [64, 768]
            |
            v
        [64, 6, 128]
    """

    def __init__(
        self,
        input_dim=768,
        num_tokens=6,
        token_dim=128
    ):

        super().__init__()

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.projection = nn.Sequential(

            nn.LayerNorm(input_dim),

            nn.Linear(
                input_dim,
                num_tokens * token_dim
            ),

            nn.GELU()
        )

    def forward(self, x):

        x = self.projection(x)

        x = x.view(
            x.size(0),
            self.num_tokens,
            self.token_dim
        )

        return x


# =============================================================================
# BIDIRECTIONAL CROSS-MODAL ATTENTION
# =============================================================================

class CrossModalAttention(nn.Module):

    """
    Bidirectional cross-modal attention.

    Query modality attends to Key/Value modality.

    Example:

        Text -> Audio

        Query = Text tokens
        Key   = Audio tokens
        Value = Audio tokens

    Output:
        Text tokens enriched with Audio information.
    """

    def __init__(
        self,
        token_dim=128,
        num_heads=4,
        dropout=0.10
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(token_dim)

        self.feed_forward = nn.Sequential(

            nn.Linear(
                token_dim,
                token_dim * 2
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                token_dim * 2,
                token_dim
            )
        )

        self.norm2 = nn.LayerNorm(token_dim)

    def forward(
        self,
        query_tokens,
        context_tokens
    ):

        attended, _ = self.attention(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False
        )

        x = self.norm1(
            query_tokens + attended
        )

        ff = self.feed_forward(x)

        x = self.norm2(
            x + ff
        )

        return x


# =============================================================================
# V5 MODEL
# =============================================================================

class V5PairwiseCrossModalAttention(nn.Module):

    def __init__(self):

        super().__init__()

        # ---------------------------------------------------------------------
        # Project each modality
        # ---------------------------------------------------------------------

        self.text_projection = FeatureTokenProjection(
            FEATURE_DIM,
            NUM_TOKENS,
            TOKEN_DIM
        )

        self.audio_projection = FeatureTokenProjection(
            FEATURE_DIM,
            NUM_TOKENS,
            TOKEN_DIM
        )

        self.video_projection = FeatureTokenProjection(
            FEATURE_DIM,
            NUM_TOKENS,
            TOKEN_DIM
        )

        # ---------------------------------------------------------------------
        # Six directional pairwise interactions
        # ---------------------------------------------------------------------

        # Text -> Audio
        self.text_to_audio = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # Audio -> Text
        self.audio_to_text = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # Text -> Video
        self.text_to_video = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # Video -> Text
        self.video_to_text = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # Audio -> Video
        self.audio_to_video = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # Video -> Audio
        self.video_to_audio = CrossModalAttention(
            TOKEN_DIM,
            NUM_HEADS,
            ATTENTION_DROPOUT
        )

        # ---------------------------------------------------------------------
        # Final classifier
        #
        # 6 interaction outputs
        # each = 128
        #
        # Total = 6 * 128 = 768
        # ---------------------------------------------------------------------

        fusion_dim = 6 * TOKEN_DIM

        self.classifier = nn.Sequential(

            nn.LayerNorm(fusion_dim),

            nn.Linear(
                fusion_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                CLASSIFIER_DROPOUT
            ),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                CLASSIFIER_DROPOUT
            ),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )

    # -------------------------------------------------------------------------
    # Mean pool tokens
    # -------------------------------------------------------------------------

    @staticmethod
    def pool_tokens(x):

        return x.mean(dim=1)

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        text,
        audio,
        video
    ):

        # -------------------------------------------------------------
        # Convert modality vectors into feature tokens
        # -------------------------------------------------------------

        text_tokens = self.text_projection(text)

        audio_tokens = self.audio_projection(audio)

        video_tokens = self.video_projection(video)

        # -------------------------------------------------------------
        # Six directional interactions
        # -------------------------------------------------------------

        text_audio = self.text_to_audio(
            text_tokens,
            audio_tokens
        )

        audio_text = self.audio_to_text(
            audio_tokens,
            text_tokens
        )

        text_video = self.text_to_video(
            text_tokens,
            video_tokens
        )

        video_text = self.video_to_text(
            video_tokens,
            text_tokens
        )

        audio_video = self.audio_to_video(
            audio_tokens,
            video_tokens
        )

        video_audio = self.video_to_audio(
            video_tokens,
            audio_tokens
        )

        # -------------------------------------------------------------
        # Pool each interaction
        # -------------------------------------------------------------

        text_audio = self.pool_tokens(
            text_audio
        )

        audio_text = self.pool_tokens(
            audio_text
        )

        text_video = self.pool_tokens(
            text_video
        )

        video_text = self.pool_tokens(
            video_text
        )

        audio_video = self.pool_tokens(
            audio_video
        )

        video_audio = self.pool_tokens(
            video_audio
        )

        # -------------------------------------------------------------
        # Combine six interaction representations
        #
        # 6 x 128 = 768
        # -------------------------------------------------------------

        fusion = torch.cat(
            [
                text_audio,
                audio_text,
                text_video,
                video_text,
                audio_video,
                video_audio,
            ],
            dim=1
        )

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        logits = self.classifier(
            fusion
        )

        return logits


# =============================================================================
# PARAMETER COUNT
# =============================================================================

def print_model_parameters(model):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print("Model parameters")
    print("-" * 50)
    print(f"Total parameters    : {total:,}")
    print(f"Trainable parameters: {trainable:,}")


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler,
    use_amp
):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for (
        text,
        audio,
        video,
        labels
    ) in loader:

        text = text.to(
            device,
            non_blocking=True
        )

        audio = audio.to(
            device,
            non_blocking=True
        )

        video = video.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp
        ):

            logits = model(
                text,
                audio,
                video
            )

            loss = criterion(
                logits,
                labels
            )

        if use_amp:

            scaler.scale(loss).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP
            )

            optimizer.step()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    use_amp
):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for (
        text,
        audio,
        video,
        labels
    ) in loader:

        text = text.to(
            device,
            non_blocking=True
        )

        audio = audio.to(
            device,
            non_blocking=True
        )

        video = video.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=use_amp
        ):

            logits = model(
                text,
                audio,
                video
            )

            loss = criterion(
                logits,
                labels
            )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": all_labels,
        "predictions": all_predictions,
    }


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    best_epoch,
    dev_results,
    test_results,
    v4_accuracy,
    v4_macro_f1,
    v4_weighted_f1
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "=" * 80 + "\n"
        )

        f.write(
            "V5 FEATURE-LEVEL PAIRWISE CROSS-MODAL ATTENTION RESULTS\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            f"Run ID: {RUN_ID}\n"
        )

        f.write(
            f"Version: {VERSION}\n\n"
        )

        f.write(
            "MODEL\n"
        )

        f.write(
            "------\n"
        )

        f.write(
            "Text: DistilBERT 768-D\n"
        )

        f.write(
            "Audio: Wav2Vec2 768-D\n"
        )

        f.write(
            "Video: FER-ViT 768-D after frame mean pooling\n"
        )

        f.write(
            "Fusion: Feature-level pairwise cross-modal attention\n"
        )

        f.write(
            "Directions: Text->Audio, Audio->Text, "
            "Text->Video, Video->Text, "
            "Audio->Video, Video->Audio\n\n"
        )

        f.write(
            f"Best Epoch: {best_epoch}\n\n"
        )

        # ---------------------------------------------------------------------
        # V5 Dev
        # ---------------------------------------------------------------------

        f.write(
            "V5 DEV RESULTS\n"
        )

        f.write(
            "---------------\n"
        )

        f.write(
            f"Accuracy    : "
            f"{dev_results['accuracy']:.4f}\n"
        )

        f.write(
            f"Macro-F1    : "
            f"{dev_results['macro_f1']:.4f}\n"
        )

        f.write(
            f"Weighted-F1 : "
            f"{dev_results['weighted_f1']:.4f}\n\n"
        )

        # ---------------------------------------------------------------------
        # V5 Test
        # ---------------------------------------------------------------------

        f.write(
            "V5 TEST RESULTS\n"
        )

        f.write(
            "----------------\n"
        )

        f.write(
            f"Accuracy    : "
            f"{test_results['accuracy']:.4f}\n"
        )

        f.write(
            f"Macro-F1    : "
            f"{test_results['macro_f1']:.4f}\n"
        )

        f.write(
            f"Weighted-F1 : "
            f"{test_results['weighted_f1']:.4f}\n\n"
        )

        # ---------------------------------------------------------------------
        # V4 comparison
        # ---------------------------------------------------------------------

        f.write(
            "V4 BASELINE\n"
        )

        f.write(
            "-----------\n"
        )

        f.write(
            f"Accuracy    : "
            f"{v4_accuracy:.4f}\n"
        )

        f.write(
            f"Macro-F1    : "
            f"{v4_macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted-F1 : "
            f"{v4_weighted_f1:.4f}\n\n"
        )

        # ---------------------------------------------------------------------
        # Difference
        # ---------------------------------------------------------------------

        f.write(
            "V5 - V4 DIFFERENCE\n"
        )

        f.write(
            "------------------\n"
        )

        f.write(
            f"Accuracy difference    : "
            f"{test_results['accuracy'] - v4_accuracy:+.4f}\n"
        )

        f.write(
            f"Macro-F1 difference    : "
            f"{test_results['macro_f1'] - v4_macro_f1:+.4f}\n"
        )

        f.write(
            f"Weighted-F1 difference : "
            f"{test_results['weighted_f1'] - v4_weighted_f1:+.4f}\n\n"
        )

        # ---------------------------------------------------------------------
        # Classification report
        # ---------------------------------------------------------------------

        f.write(
            "V5 TEST CLASSIFICATION REPORT\n"
        )

        f.write(
            "------------------------------\n"
        )

        f.write(
            classification_report(
                test_results["labels"],
                test_results["predictions"],
                target_names=EMOTION_LABELS,
                zero_division=0
            )
        )

        f.write("\n\n")

        # ---------------------------------------------------------------------
        # Confusion matrix
        # ---------------------------------------------------------------------

        f.write(
            "V5 TEST CONFUSION MATRIX\n"
        )

        f.write(
            "------------------------\n"
        )

        cm = confusion_matrix(
            test_results["labels"],
            test_results["predictions"],
            labels=list(
                range(NUM_CLASSES)
            )
        )

        f.write(
            np.array2string(cm)
        )

        f.write("\n")


# =============================================================================
# MAIN
# =============================================================================

def main():

    set_seed(SEED)

    device = get_device()

    # -------------------------------------------------------------------------
    # IMPORTANT GPU SAFETY CHECK
    # -------------------------------------------------------------------------

    if device.type != "cuda":

        raise RuntimeError(
            "\nCUDA GPU is not available.\n\n"
            "V5 training has been stopped intentionally.\n"
            "Please fix the PyTorch CUDA installation first.\n\n"
            "Run:\n"
            "python -c \"import torch; "
            "print('PyTorch:',torch.__version__); "
            "print('CUDA available:',torch.cuda.is_available()); "
            "print('CUDA version:',torch.version.cuda); "
            "print('GPU:',torch.cuda.get_device_name(0) "
            "if torch.cuda.is_available() else 'NO GPU')\""
        )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # =========================================================================
    # LOAD CACHES
    # =========================================================================

    print()
    print("Loading cached features...")
    print("-" * 80)

    text_train_cache = load_cache(
        TEXT_TRAIN
    )

    text_dev_cache = load_cache(
        TEXT_DEV
    )

    text_test_cache = load_cache(
        TEXT_TEST
    )

    audio_train_cache = load_cache(
        AUDIO_TRAIN
    )

    audio_dev_cache = load_cache(
        AUDIO_DEV
    )

    audio_test_cache = load_cache(
        AUDIO_TEST
    )

    video_train_cache = load_cache(
        VIDEO_TRAIN
    )

    video_dev_cache = load_cache(
        VIDEO_DEV
    )

    video_test_cache = load_cache(
        VIDEO_TEST
    )

    # =========================================================================
    # EXTRACT FEATURES
    # =========================================================================

    text_train = get_features(
        text_train_cache,
        "Text train"
    )

    text_dev = get_features(
        text_dev_cache,
        "Text dev"
    )

    text_test = get_features(
        text_test_cache,
        "Text test"
    )

    audio_train = get_features(
        audio_train_cache,
        "Audio train"
    )

    audio_dev = get_features(
        audio_dev_cache,
        "Audio dev"
    )

    audio_test = get_features(
        audio_test_cache,
        "Audio test"
    )

    video_train = get_features(
        video_train_cache,
        "Video train"
    )

    video_dev = get_features(
        video_dev_cache,
        "Video dev"
    )

    video_test = get_features(
        video_test_cache,
        "Video test"
    )

    # =========================================================================
    # EXTRACT + NORMALIZE LABELS
    # =========================================================================

    text_labels_train = normalize_labels(
        get_labels(
            text_train_cache,
            "Text train"
        ),
        "Text train"
    )

    text_labels_dev = normalize_labels(
        get_labels(
            text_dev_cache,
            "Text dev"
        ),
        "Text dev"
    )

    text_labels_test = normalize_labels(
        get_labels(
            text_test_cache,
            "Text test"
        ),
        "Text test"
    )

    video_labels_train = normalize_labels(
        get_labels(
            video_train_cache,
            "Video train"
        ),
        "Video train"
    )

    video_labels_dev = normalize_labels(
        get_labels(
            video_dev_cache,
            "Video dev"
        ),
        "Video dev"
    )

    video_labels_test = normalize_labels(
        get_labels(
            video_test_cache,
            "Video test"
        ),
        "Video test"
    )

    # =========================================================================
    # VERIFY LABEL ALIGNMENT
    # =========================================================================

    print()
    print("Checking label alignment...")
    print("-" * 80)

    verify_label_alignment(
        text_labels_train,
        video_labels_train,
        "TRAIN"
    )

    verify_label_alignment(
        text_labels_dev,
        video_labels_dev,
        "DEV"
    )

    verify_label_alignment(
        text_labels_test,
        video_labels_test,
        "TEST"
    )

    # We use text labels after verifying that they exactly match video labels.
    train_labels = text_labels_train
    dev_labels = text_labels_dev
    test_labels = text_labels_test

    # =========================================================================
    # PRINT DISTRIBUTIONS
    # =========================================================================

    print_distribution(
        train_labels,
        "TRAIN"
    )

    print_distribution(
        dev_labels,
        "DEV"
    )

    print_distribution(
        test_labels,
        "TEST"
    )

    # =========================================================================
    # VERIFY SAMPLE COUNTS
    # =========================================================================

    print()
    print("Checking sample counts...")
    print("-" * 80)

    train_count = len(train_labels)
    dev_count = len(dev_labels)
    test_count = len(test_labels)

    print(
        f"Train samples: {train_count}"
    )

    print(
        f"Dev samples  : {dev_count}"
    )

    print(
        f"Test samples : {test_count}"
    )

    if not (
        len(text_train) == len(audio_train) == len(video_train) == train_count
    ):

        raise ValueError(
            "TRAIN sample count mismatch."
        )

    if not (
        len(text_dev) == len(audio_dev) == len(video_dev) == dev_count
    ):

        raise ValueError(
            "DEV sample count mismatch."
        )

    if not (
        len(text_test) == len(audio_test) == len(video_test) == test_count
    ):

        raise ValueError(
            "TEST sample count mismatch."
        )

    # =========================================================================
    # DATASETS
    # =========================================================================

    train_dataset = TensorDataset(
        text_train,
        audio_train,
        video_train,
        train_labels
    )

    dev_dataset = TensorDataset(
        text_dev,
        audio_dev,
        video_dev,
        dev_labels
    )

    test_dataset = TensorDataset(
        text_test,
        audio_test,
        video_test,
        test_labels
    )

    # =========================================================================
    # DATALOADERS
    # =========================================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    print()
    print("Creating V5 model...")
    print("-" * 80)

    model = V5PairwiseCrossModalAttention()

    model = model.to(device)

    print_model_parameters(
        model
    )

    # =========================================================================
    # LOSS
    # =========================================================================

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # =========================================================================
    # LR SCHEDULER
    # =========================================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # =========================================================================
    # AMP
    # =========================================================================

    use_amp = (
        device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    print()
    print("Training configuration")
    print("-" * 80)

    print(
        f"Batch size        : {BATCH_SIZE}"
    )

    print(
        f"Max epochs        : {MAX_EPOCHS}"
    )

    print(
        f"Learning rate     : {LEARNING_RATE}"
    )

    print(
        f"Weight decay      : {WEIGHT_DECAY}"
    )

    print(
        f"Label smoothing   : {LABEL_SMOOTHING}"
    )

    print(
        f"Attention heads   : {NUM_HEADS}"
    )

    print(
        f"Feature tokens    : {NUM_TOKENS}"
    )

    print(
        f"Token dimension   : {TOKEN_DIM}"
    )

    print(
        f"AMP enabled       : {use_amp}"
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    print()
    print("=" * 80)
    print("STARTING V5 TRAINING")
    print("=" * 80)

    best_dev_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        train_loss, train_acc, train_macro_f1, train_weighted_f1 = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scaler=scaler,
                use_amp=use_amp
            )
        )

        dev_results = evaluate(
            model=model,
            loader=dev_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp
        )

        scheduler.step(
            dev_results["macro_f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print()
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"  Train Loss     : {train_loss:.4f}"
        )

        print(
            f"  Train Accuracy : {train_acc:.4f}"
        )

        print(
            f"  Train Macro-F1 : {train_macro_f1:.4f}"
        )

        print(
            f"  Train Weighted : {train_weighted_f1:.4f}"
        )

        print(
            f"  Dev Loss       : {dev_results['loss']:.4f}"
        )

        print(
            f"  Dev Accuracy   : {dev_results['accuracy']:.4f}"
        )

        print(
            f"  Dev Macro-F1   : {dev_results['macro_f1']:.4f}"
        )

        print(
            f"  Dev Weighted   : {dev_results['weighted_f1']:.4f}"
        )

        print(
            f"  Learning Rate  : {current_lr:.8f}"
        )

        # ---------------------------------------------------------------------
        # Save best model using DEV MACRO-F1
        # ---------------------------------------------------------------------

        if dev_results["macro_f1"] > best_dev_macro_f1:

            best_dev_macro_f1 = (
                dev_results["macro_f1"]
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "run_id": RUN_ID,
                    "version": VERSION,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_macro_f1": best_dev_macro_f1,
                    "config": {
                        "feature_dim": FEATURE_DIM,
                        "num_tokens": NUM_TOKENS,
                        "token_dim": TOKEN_DIM,
                        "num_heads": NUM_HEADS,
                        "batch_size": BATCH_SIZE,
                        "max_epochs": MAX_EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                        "seed": SEED,
                    }
                },
                BEST_CHECKPOINT
            )

            print(
                f"  *** BEST MODEL SAVED "
                f"(Dev Macro-F1={best_dev_macro_f1:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement. "
                f"Patience {patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # ---------------------------------------------------------------------
        # Early stopping
        # ---------------------------------------------------------------------

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING BEST V5 CHECKPOINT")
    print("=" * 80)

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best epoch       : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{checkpoint['best_dev_macro_f1']:.4f}"
    )

    # =========================================================================
    # FINAL DEV EVALUATION
    # =========================================================================

    final_dev_results = evaluate(
        model=model,
        loader=dev_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp
    )

    # =========================================================================
    # FINAL TEST EVALUATION
    # =========================================================================

    print()
    print("=" * 80)
    print("FINAL V5 TEST EVALUATION")
    print("=" * 80)

    final_test_results = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp
    )

    print()
    print(
        f"Test Accuracy    : "
        f"{final_test_results['accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1    : "
        f"{final_test_results['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted-F1 : "
        f"{final_test_results['weighted_f1']:.4f}"
    )

    # =========================================================================
    # CLASSIFICATION REPORT
    # =========================================================================

    print()
    print("V5 TEST CLASSIFICATION REPORT")
    print("-" * 80)

    print(
        classification_report(
            final_test_results["labels"],
            final_test_results["predictions"],
            target_names=EMOTION_LABELS,
            zero_division=0
        )
    )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    print()
    print("V5 TEST CONFUSION MATRIX")
    print("-" * 80)

    cm = confusion_matrix(
        final_test_results["labels"],
        final_test_results["predictions"],
        labels=list(
            range(NUM_CLASSES)
        )
    )

    print(cm)

    # =========================================================================
    # V4 BASELINE
    # =========================================================================

    V4_ACCURACY = 0.4782
    V4_MACRO_F1 = 0.1910
    V4_WEIGHTED_F1 = 0.4018

    print()
    print("=" * 80)
    print("V4 VS V5 COMPARISON")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<20}"
        f"{'V4':>12}"
        f"{'V5':>12}"
        f"{'Difference':>15}"
    )

    print("-" * 60)

    print(
        f"{'Accuracy':<20}"
        f"{V4_ACCURACY:>12.4f}"
        f"{final_test_results['accuracy']:>12.4f}"
        f"{final_test_results['accuracy'] - V4_ACCURACY:>+15.4f}"
    )

    print(
        f"{'Macro-F1':<20}"
        f"{V4_MACRO_F1:>12.4f}"
        f"{final_test_results['macro_f1']:>12.4f}"
        f"{final_test_results['macro_f1'] - V4_MACRO_F1:>+15.4f}"
    )

    print(
        f"{'Weighted-F1':<20}"
        f"{V4_WEIGHTED_F1:>12.4f}"
        f"{final_test_results['weighted_f1']:>12.4f}"
        f"{final_test_results['weighted_f1'] - V4_WEIGHTED_F1:>+15.4f}"
    )

    # =========================================================================
    # SAVE TEXT RESULTS
    # =========================================================================

    save_results(
        best_epoch=checkpoint["epoch"],
        dev_results=final_dev_results,
        test_results=final_test_results,
        v4_accuracy=V4_ACCURACY,
        v4_macro_f1=V4_MACRO_F1,
        v4_weighted_f1=V4_WEIGHTED_F1
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        f"  {RESULTS_FILE}"
    )

    print(
        f"Best checkpoint saved to:"
    )

    print(
        f"  {BEST_CHECKPOINT}"
    )

    # =========================================================================
    # EXPERIMENT LOGGER
    # =========================================================================

    print()
    print("=" * 80)
    print("LOGGING EXPERIMENT")
    print("=" * 80)

    try:

        log_run(
            modality="Multimodal",
            model="V5 Feature-Level Pairwise Cross-Modal Attention",
            dataset="MELD",
            train_samples=str(train_count),
            validation_samples=str(dev_count),
            test_samples=str(test_count),
            best_epoch=str(
                checkpoint["epoch"]
            ),
            accuracy=f"{final_test_results['accuracy']:.4f}",
            macro_f1=f"{final_test_results['macro_f1']:.4f}",
            weighted_f1=f"{final_test_results['weighted_f1']:.4f}",
            checkpoint=BEST_CHECKPOINT,
            status="Complete",
            notes=(
                f"Run ID={RUN_ID}; Version={VERSION}; "
                f"Feature-level bidirectional pairwise cross-modal attention; "
                f"directions=Text->Audio, Audio->Text, "
                f"Text->Video, Video->Text, "
                f"Audio->Video, Video->Audio; "
                f"V4 baseline accuracy={V4_ACCURACY:.4f}, "
                f"macro_f1={V4_MACRO_F1:.4f}, "
                f"weighted_f1={V4_WEIGHTED_F1:.4f}"
            )
        )

        print(
            "Experiment logged successfully."
        )

    except Exception as e:

        print()
        print(
            "WARNING: Experiment logger failed."
        )

        print(
            f"Logger error: {e}"
        )

        print(
            "The V5 checkpoint and results file were already saved."
        )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 80)
    print("V5 TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Run ID             : {RUN_ID}"
    )

    print(
        f"Best Epoch         : {checkpoint['epoch']}"
    )

    print(
        f"Dev Macro-F1       : "
        f"{final_dev_results['macro_f1']:.4f}"
    )

    print(
        f"Test Accuracy      : "
        f"{final_test_results['accuracy']:.4f}"
    )

    print(
        f"Test Macro-F1      : "
        f"{final_test_results['macro_f1']:.4f}"
    )

    print(
        f"Test Weighted-F1   : "
        f"{final_test_results['weighted_f1']:.4f}"
    )

    print()
    print(
        "Checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        "Results:"
    )

    print(
        RESULTS_FILE
    )

    print("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()