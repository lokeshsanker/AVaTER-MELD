import sys
import os

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# IMPORTS
# =========================================================

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm

from config import (
    DEVICE,
    TEXT_BATCH_SIZE,
    IDX2LABEL,
    NUM_CLASSES,
)

from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


# =========================================================
# CONFIGURATION
# =========================================================

MANIFEST = "data/manifests/test.csv"

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "text_v2",
    "pytorch_model.bin"
)


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 70)
    print("TEXT V2 - FINAL TEST EVALUATION")
    print("=" * 70)

    print(f"Device     : {DEVICE}")
    print(f"Manifest   : {MANIFEST}")
    print(f"Checkpoint : {CHECKPOINT}")


    # -----------------------------------------------------
    # Check checkpoint
    # -----------------------------------------------------

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Text V2 checkpoint not found:\n{CHECKPOINT}"
        )


    # -----------------------------------------------------
    # Load test manifest
    # -----------------------------------------------------

    print("\nLoading test manifest...")

    df = load_manifest(MANIFEST)

    print(
        f"Test samples: {len(df)}"
    )


    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    dataset = TextEmotionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    # -----------------------------------------------------
    # Load Text V2 model
    # -----------------------------------------------------

    print("\nLoading Text V2 model...")

    model = TextEmotionModel().to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    print("Text V2 checkpoint loaded successfully.")


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_true = []
    y_pred = []

    print("\nRunning test evaluation...")

    with torch.no_grad():

        for input_ids, attention_mask, labels, _ in tqdm(
            loader
        ):

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            logits = model(
                input_ids,
                attention_mask
            )

            predictions = logits.argmax(
                dim=-1
            ).cpu().tolist()

            y_pred.extend(predictions)

            y_true.extend(
                labels.tolist()
            )


    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------

    labels = list(range(NUM_CLASSES))

    names = [
        IDX2LABEL[i]
        for i in labels
    ]


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0
    )


    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        zero_division=0
    )


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


    # -----------------------------------------------------
    # Prediction counts
    # -----------------------------------------------------

    predicted_counts = {}

    for class_index, emotion in IDX2LABEL.items():

        predicted_counts[emotion] = y_pred.count(
            class_index
        )


    detected_classes = sum(
        count > 0
        for count in predicted_counts.values()
    )


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n")
    print("=" * 70)
    print("TEXT V2 - FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Macro-F1    : {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1 : {weighted_f1:.4f} "
        f"({weighted_f1 * 100:.2f}%)"
    )


    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    print("\nClassification Report:")
    print(report)


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    print("Confusion Matrix:")

    print(
        f"{'':12s}" +
        "".join(
            f"{name[:9]:>10s}"
            for name in names
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{names[i]:12s}" +
            "".join(
                f"{value:10d}"
                for value in row
            )
        )


    # -----------------------------------------------------
    # 7-class detection
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("7-CLASS DETECTION")
    print("=" * 70)

    for emotion in names:

        count = predicted_counts[emotion]

        if count > 0:

            print(
                f"{emotion:<10}: Detected ({count})"
            )

        else:

            print(
                f"{emotion:<10}: NOT detected (0)"
            )


    print(
        f"\nDetected classes: "
        f"{detected_classes}/{NUM_CLASSES}"
    )


    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    print("\nPrediction Distribution:")

    total_predictions = len(y_pred)

    for emotion in names:

        count = predicted_counts[emotion]

        percentage = (
            count / total_predictions * 100
        )

        print(
            f"{emotion:<10}: "
            f"{count:4d} "
            f"({percentage:6.2f}%)"
        )


    # -----------------------------------------------------
    # Sanity check
    # -----------------------------------------------------

    print("\nSanity Check:")

    print(
        f"Expected test samples : {len(df)}"
    )

    print(
        f"Predictions generated : {len(y_pred)}"
    )

    if len(df) == len(y_pred):

        print("Status: PASS")

    else:

        print("Status: ERROR - sample count mismatch")


    print("\n" + "=" * 70)
    print("TEXT V2 TEST EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()