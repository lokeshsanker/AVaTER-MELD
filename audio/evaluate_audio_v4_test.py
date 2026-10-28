import os
import sys
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
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

from config import DEVICE, NUM_CLASSES, IDX2LABEL


# ============================================================
# CONFIG
# ============================================================

TEST_CACHE = (
    "data/audio_v4_feature_cache_test.pt"
)

CHECKPOINT_PATH = (
    "checkpoints/audio_v4/audio_v4_best.pt"
)

BATCH_SIZE = 64


# ============================================================
# DATASET
# ============================================================

class FeatureDataset(Dataset):

    def __init__(self, cache_path):

        cache = torch.load(
            cache_path,
            map_location="cpu"
        )

        self.features = cache[
            "features"
        ].float()

        self.labels = cache[
            "labels"
        ].long()

        self.clip_ids = cache[
            "clip_ids"
        ]

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.features[idx],
            self.labels[idx],
            self.clip_ids[idx]
        )


# ============================================================
# MODEL
# ============================================================

class AudioV4Classifier(nn.Module):

    def __init__(
        self,
        feature_dim=768,
        num_classes=7
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(
                feature_dim
            ),

            nn.Linear(
                feature_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO V4 - TEST EVALUATION")
    print("=" * 70)

    print(
        f"PyTorch       : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device        : "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # LOAD TEST CACHE
    # --------------------------------------------------------

    print("\nLoading Audio V4 test features...")

    test_dataset = FeatureDataset(
        TEST_CACHE
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print(
        f"Test features: "
        f"{test_dataset.features.shape}"
    )

    print(
        f"Test samples : "
        f"{len(test_dataset)}"
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = AudioV4Classifier(
        feature_dim=768,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    print(
        "\nLoading checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state
    )

    model.eval()

    print(
        "Checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    y_true = []
    y_pred = []

    print(
        "\nRunning TEST evaluation..."
    )

    with torch.no_grad():

        for features, labels, clip_ids in test_loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                features
            )

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                labels.numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    labels = list(
        range(NUM_CLASSES)
    )

    names = [
        IDX2LABEL[i]
        for i in labels
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels,
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels,
        zero_division=0
    )

    # --------------------------------------------------------
    # PREDICTION COUNTS
    # --------------------------------------------------------

    predicted_counts = {}

    detected_classes = 0

    for i in labels:

        emotion = IDX2LABEL[i]

        count = int(
            np.sum(
                y_pred == i
            )
        )

        predicted_counts[
            emotion
        ] = count

        if count > 0:

            detected_classes += 1

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "AUDIO V4 TEST RESULTS"
    )

    print("=" * 70)

    print(
        f"Accuracy   : "
        f"{accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1   : "
        f"{macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1: "
        f"{weighted_f1:.4f} "
        f"({weighted_f1 * 100:.2f}%)"
    )

    print(
        f"Classes detected: "
        f"{detected_classes}/7"
    )

    # --------------------------------------------------------
    # CLASS DETECTION
    # --------------------------------------------------------

    print(
        "\nClass Detection:"
    )

    for emotion in names:

        count = predicted_counts[
            emotion
        ]

        if count > 0:

            print(
                f"{emotion:<10}: "
                f"Detected ({count})"
            )

        else:

            print(
                f"{emotion:<10}: "
                f"Not detected (0)"
            )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print(
        "\nClassification Report:"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print(
        "Confusion Matrix:"
    )

    print(
        "             "
        + " ".join(
            f"{name[:8]:>9}"
            for name in names
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{names[i]:>10} "
            + " ".join(
                f"{value:9d}"
                for value in row
            )
        )

    # --------------------------------------------------------
    # PREDICTION DISTRIBUTION
    # --------------------------------------------------------

    print(
        "\nPredicted class distribution:"
    )

    for emotion in names:

        print(
            f"{emotion:>10}: "
            f"{predicted_counts[emotion]:4d}"
        )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print()

    if len(y_pred) == len(test_dataset):

        print(
            f"Sanity check: PASS "
            f"({len(y_pred)} predictions)"
        )

    else:

        print(
            "Sanity check: FAIL"
        )

    print("=" * 70)


if __name__ == "__main__":

    main()