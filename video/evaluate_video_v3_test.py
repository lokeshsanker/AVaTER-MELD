# ============================================================
# VIDEO V3 - TEST SET EVALUATION
# ============================================================
# Evaluates the best Video V3 checkpoint on the TEST set.
#
# Input:
#   data/video_v2_feature_cache_test.pt
#   checkpoints/video_v3/video_v3_best.pt
#
# Feature shape:
#   [N, 4, 1280]
#
# Model:
#   MobileNetV2 features -> BiGRU -> Attention Pooling -> Classifier
#
# Output:
#   - Test Accuracy
#   - Macro-F1
#   - Weighted-F1
#   - Per-class precision / recall / F1
#   - Confusion matrix
#   - Prediction distribution
#   - Class detection
#   - Results saved to results/video_v3_test_results.txt
# ============================================================

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ------------------------------------------------------------
# PROJECT ROOT
# ------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    NUM_CLASSES,
    IDX2LABEL,
)

EMOTION_LABELS = [
    IDX2LABEL[i]
    for i in range(NUM_CLASSES)
]

# Exact test feature-cache path
VIDEO_V2_FEATURE_CACHE_TEST = os.path.join(
    PROJECT_ROOT,
    "data",
    "video_v2_feature_cache_test.pt",
)
# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "video_v3",
    "video_v3_best.pt",
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "video_v3_test_results.txt",
)

# Video V3 architecture
INPUT_DIM = 1280
GRU_HIDDEN = 256


# ------------------------------------------------------------
# REPRODUCIBILITY
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    Attention pooling over the GRU temporal sequence.
    """

    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, seq):
        """
        seq:
            [B, T, H]

        returns:
            pooled:
                [B, H]

            weights:
                [B, T]
        """

        scores = self.attn(seq).squeeze(-1)

        weights = torch.softmax(scores, dim=1)

        pooled = torch.bmm(
            weights.unsqueeze(1),
            seq
        ).squeeze(1)

        return pooled, weights


class VideoEmotionModel(nn.Module):
    """
    Exact Video V3 architecture reconstructed from
    the trained checkpoint.

    Input:
        [B, 4, 1280]

    1280-dimensional MobileNetV2 frame features
        ↓
    BiGRU
        ↓
    Attention
        ↓
    LayerNorm
        ↓
    Linear 512 → 128
        ↓
    ReLU
        ↓
    Dropout
        ↓
    Linear 128 → 7
    """

    def __init__(
        self,
        input_dim=1280,
        gru_hidden=256,
        num_classes=NUM_CLASSES,
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

        # BiGRU output dimension:
        # 256 × 2 = 512
        gru_output_dim = 2 * gru_hidden

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------
        self.attention = nn.Linear(
            gru_output_dim,
            1,
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_output_dim),
            nn.Linear(gru_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
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
        # Attention
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
        # Weighted temporal pooling
        # ----------------------------------------------------
        pooled = torch.bmm(
            attention_weights.unsqueeze(1),
            seq_out,
        ).squeeze(1)

        # pooled:
        # [B, 512]

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------
        logits = self.classifier(
            pooled
        )

        return logits, attention_weights

# ------------------------------------------------------------
# LOAD FEATURES
# ------------------------------------------------------------
def load_test_features():

    print("\n" + "=" * 70)
    print("LOADING VIDEO V3 TEST FEATURES")
    print("=" * 70)

    print(f"\nFeature cache:")
    print(VIDEO_V2_FEATURE_CACHE_TEST)

    if not os.path.exists(VIDEO_V2_FEATURE_CACHE_TEST):
        raise FileNotFoundError(
            f"\n❌ Test feature cache not found:\n"
            f"{VIDEO_V2_FEATURE_CACHE_TEST}"
        )

    cache = torch.load(
        VIDEO_V2_FEATURE_CACHE_TEST,
        map_location="cpu",
        weights_only=False,
    )

    print(f"\nCache type: {type(cache)}")

    # --------------------------------------------------------
    # Handle dictionary cache
    # --------------------------------------------------------
    if isinstance(cache, dict):

        if "features" not in cache:
            raise KeyError(
                "❌ Cache dictionary does not contain 'features'."
            )

        features = cache["features"]

        labels = cache.get("labels", None)

        clip_ids = cache.get("clip_ids", None)

    # --------------------------------------------------------
    # Handle tensor cache
    # --------------------------------------------------------
    elif torch.is_tensor(cache):

        features = cache
        labels = None
        clip_ids = None

    else:

        raise TypeError(
            f"❌ Unsupported cache type: {type(cache)}"
        )

    # --------------------------------------------------------
    # Convert features
    # --------------------------------------------------------
    if not torch.is_tensor(features):

        features = torch.tensor(
            features,
            dtype=torch.float32,
        )

    features = features.float()

    print(f"\nFeature shape: {tuple(features.shape)}")
    print(f"Feature dtype: {features.dtype}")

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------
    if features.ndim != 3:

        raise ValueError(
            f"\n❌ Expected features with 3 dimensions "
            f"[N, T, D], got {tuple(features.shape)}"
        )

    n_samples, num_frames, feature_dim = features.shape

    print(f"Samples:       {n_samples}")
    print(f"Frames/sample: {num_frames}")
    print(f"Feature dim:   {feature_dim}")

    if num_frames != 4:

        print(
            f"\n⚠️ Warning: expected 4 frames, "
            f"found {num_frames}"
        )

    if feature_dim != INPUT_DIM:

        raise ValueError(
            f"\n❌ Expected feature dimension {INPUT_DIM}, "
            f"got {feature_dim}"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------
    if labels is None:

        raise ValueError(
            "\n❌ Test cache does not contain labels.\n"
            "Video V3 test evaluation requires labels."
        )

    if not torch.is_tensor(labels):

        labels = torch.tensor(
            labels,
            dtype=torch.long,
        )

    labels = labels.long()

    print(f"Labels shape: {tuple(labels.shape)}")

    if len(labels) != len(features):

        raise ValueError(
            f"\n❌ Feature/label mismatch:\n"
            f"Features = {len(features)}\n"
            f"Labels   = {len(labels)}"
        )

    # --------------------------------------------------------
    # Clip IDs
    # --------------------------------------------------------
    if clip_ids is not None:

        print(f"Clip IDs:      {len(clip_ids)}")

    return features, labels, clip_ids


# ------------------------------------------------------------
# LOAD CHECKPOINT
# ------------------------------------------------------------
def load_model(device):

    print("\n" + "=" * 70)
    print("LOADING VIDEO V3 CHECKPOINT")
    print("=" * 70)

    print(f"\nCheckpoint:")
    print(CHECKPOINT_PATH)

    if not os.path.exists(CHECKPOINT_PATH):

        raise FileNotFoundError(
            f"\n❌ Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = VideoEmotionModel().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Support different checkpoint formats
    # --------------------------------------------------------
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            # Sometimes the checkpoint itself is state_dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

        # --------------------------------------------------------
    # Remove possible "module." prefix
    # --------------------------------------------------------
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned_state_dict[new_key] = value

    # --------------------------------------------------------
    # Load model checkpoint EXACTLY
    # --------------------------------------------------------
    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.eval()

    print("\n✅ Video V3 checkpoint loaded EXACTLY.")

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

    return model

# ------------------------------------------------------------
# EVALUATE
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):

    all_labels = []
    all_predictions = []

    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (features, labels) in enumerate(loader):

        features = features.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits, _ = model(features)

        loss = criterion(
            logits,
            labels,
        )

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        total_loss += loss.item() * labels.size(0)

        all_labels.extend(
            labels.cpu().numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    avg_loss = total_loss / len(loader.dataset)

    return (
        np.array(all_labels),
        np.array(all_predictions),
        avg_loss,
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------
def save_results(
    y_true,
    y_pred,
    test_loss,
    accuracy,
    macro_f1,
    weighted_f1,
    report,
    cm,
):

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("=" * 80 + "\n")
        f.write("VIDEO V3 - TEST SET EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL\n")
        f.write("-" * 80 + "\n")
        f.write("Video V3\n")
        f.write("MobileNetV2 cached features + BiGRU + Attention Pooling\n")
        f.write("Input shape: [N, 4, 1280]\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n\n")

        f.write("TEST DATA\n")
        f.write("-" * 80 + "\n")
        f.write(f"Samples: {len(y_true)}\n")
        f.write("Frames per sample: 4\n")
        f.write("Feature dimension: 1280\n\n")

        f.write("OVERALL RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Test Loss:       {test_loss:.4f}\n")
        f.write(f"Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Macro-F1:        {macro_f1:.4f} ({macro_f1 * 100:.2f}%)\n")
        f.write(
            f"Weighted-F1:     {weighted_f1:.4f} "
            f"({weighted_f1 * 100:.2f}%)\n"
        )

        f.write("\n\nCLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")

        f.write(
            classification_report(
                y_true,
                y_pred,
                labels=list(range(NUM_CLASSES)),
                target_names=EMOTION_LABELS,
                digits=4,
                zero_division=0,
            )
        )

        f.write("\n\nCONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")

        f.write(
            "Rows = Actual\n"
            "Columns = Predicted\n\n"
        )

        f.write("              ")

        for label in EMOTION_LABELS:
            f.write(f"{label:>10}")

        f.write("\n")

        for i, label in enumerate(EMOTION_LABELS):

            f.write(f"{label:<12}")

            for j in range(NUM_CLASSES):

                f.write(
                    f"{cm[i, j]:>10}"
                )

            f.write("\n")

        f.write("\n\nPREDICTION DISTRIBUTION\n")
        f.write("-" * 80 + "\n")

        prediction_counts = np.bincount(
            y_pred,
            minlength=NUM_CLASSES,
        )

        for idx, label in enumerate(EMOTION_LABELS):

            percentage = (
                prediction_counts[idx]
                / len(y_pred)
                * 100
            )

            f.write(
                f"{label:<12}: "
                f"{prediction_counts[idx]:>5} "
                f"({percentage:6.2f}%)\n"
            )

        f.write("\n\nACTUAL TEST DISTRIBUTION\n")
        f.write("-" * 80 + "\n")

        actual_counts = np.bincount(
            y_true,
            minlength=NUM_CLASSES,
        )

        for idx, label in enumerate(EMOTION_LABELS):

            percentage = (
                actual_counts[idx]
                / len(y_true)
                * 100
            )

            f.write(
                f"{label:<12}: "
                f"{actual_counts[idx]:>5} "
                f"({percentage:6.2f}%)\n"
            )

        f.write("\n\nCLASS DETECTION\n")
        f.write("-" * 80 + "\n")

        detected = 0

        for idx, label in enumerate(EMOTION_LABELS):

            count = prediction_counts[idx]

            if count > 0:
                detected += 1
                status = "DETECTED"
            else:
                status = "NOT DETECTED"

            f.write(
                f"{label:<12}: "
                f"{status:<14} "
                f"predictions={count}\n"
            )

        f.write(
            f"\nClasses detected: "
            f"{detected}/{NUM_CLASSES}\n"
        )

        f.write("\n\nSANITY CHECK\n")
        f.write("-" * 80 + "\n")

        if len(y_true) == len(y_pred) == 2610:
            f.write(
                "PASS - 2610 predictions generated.\n"
            )
        else:
            f.write(
                f"WARNING - Expected 2610 predictions, "
                f"got {len(y_pred)}.\n"
            )

        f.write("\n" + "=" * 80 + "\n")

    print(
        f"\n✅ Results saved to:\n{RESULT_FILE}"
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    set_seed(SEED)

    print("\n")
    print("=" * 80)
    print("VIDEO V3 - TEST EVALUATION")
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

    print(f"PyTorch:       {torch.__version__}")
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"CUDA version:  "
            f"{torch.version.cuda}"
        )

        print(
            f"GPU:           "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory:    "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print(f"Device:        {device}")

    # --------------------------------------------------------
    # Load test features
    # --------------------------------------------------------
    features, labels, clip_ids = load_test_features()

    # --------------------------------------------------------
    # Create dataset
    # --------------------------------------------------------
    dataset = TensorDataset(
        features,
        labels,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\nDataLoader")
    print("-" * 80)
    print(f"Dataset samples: {len(dataset)}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Number batches:  {len(loader)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{total_params:,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("STARTING VIDEO V3 TEST EVALUATION")
    print("=" * 80)

    y_true, y_pred, test_loss = evaluate(
        model,
        loader,
        device,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("VIDEO V3 TEST RESULTS")
    print("=" * 80)

    print(
        f"\nTest Loss:       {test_loss:.4f}"
    )

    print(
        f"Accuracy:        "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1:        "
        f"{macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1:     "
        f"{weighted_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        "\nRows = Actual"
        "\nColumns = Predicted\n"
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

    for i, label in enumerate(EMOTION_LABELS):

        print(
            f"{label:<12}",
            end="",
        )

        for j in range(NUM_CLASSES):

            print(
                f"{cm[i, j]:>10}",
                end="",
            )

        print()

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("PREDICTION DISTRIBUTION")
    print("=" * 80)

    prediction_counts = np.bincount(
        y_pred,
        minlength=NUM_CLASSES,
    )

    detected = 0

    for idx, label in enumerate(EMOTION_LABELS):

        count = prediction_counts[idx]

        percentage = (
            count / len(y_pred) * 100
        )

        if count > 0:
            detected += 1

        print(
            f"{label:<12}: "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"\nClasses detected: "
        f"{detected}/{NUM_CLASSES}"
    )

    # --------------------------------------------------------
    # Actual distribution
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("ACTUAL TEST DISTRIBUTION")
    print("=" * 80)

    actual_counts = np.bincount(
        y_true,
        minlength=NUM_CLASSES,
    )

    for idx, label in enumerate(EMOTION_LABELS):

        count = actual_counts[idx]

        percentage = (
            count / len(y_true) * 100
        )

        print(
            f"{label:<12}: "
            f"{count:>5} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Sanity check
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("SANITY CHECK")
    print("=" * 80)

    if len(y_true) == len(y_pred) == 2610:

        print(
            "✅ PASS - 2610 test predictions generated."
        )

    else:

        print(
            f"⚠️ WARNING - Expected 2610 predictions, "
            f"got {len(y_pred)}."
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    save_results(
        y_true=y_true,
        y_pred=y_pred,
        test_loss=test_loss,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        report=report,
        cm=cm,
    )

    print("\n" + "=" * 80)
    print("VIDEO V3 TEST EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()