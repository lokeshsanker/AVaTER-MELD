import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import numpy as np

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# CONFIG
# =========================================================

from config import (
    DEVICE,
    AUDIO_BATCH_SIZE,
    IDX2LABEL
)


# =========================================================
# DATASET / MODEL
# =========================================================

from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel


# =========================================================
# EXCEL LOGGER
# =========================================================

from utils.experiment_logger import (
    log_run,
    log_class_results,
    log_configuration,
    log_confusion_matrix,
    log_class_detection
)


# =========================================================
# PATHS
# =========================================================

MANIFEST = "data/manifests/dev.csv"
CHECKPOINT = "checkpoints/audio_model.pt"


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Load dev manifest
    # -----------------------------------------------------

    print("Loading dev manifest...")

    df = load_manifest(MANIFEST)

    print(f"Validation samples: {len(df)}")


    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = AudioEmotionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # -----------------------------------------------------
    # Load audio model
    # -----------------------------------------------------

    print("Loading audio model...")

    model = AudioEmotionModel(
        freeze_encoder=True
    ).to(DEVICE)

    if not os.path.exists(CHECKPOINT):

        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    state = torch.load(
        CHECKPOINT,
        map_location=DEVICE
    )

    model.load_state_dict(state)

    model.eval()


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_true = []
    y_pred = []

    print("Evaluating...")

    with torch.no_grad():

        for input_values, labels, ids in loader:

            input_values = input_values.to(DEVICE)

            logits = model(input_values)

            preds = logits.argmax(
                dim=-1
            ).cpu().numpy()

            y_true.extend(
                labels.numpy().tolist()
            )

            y_pred.extend(
                preds.tolist()
            )


    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    # -----------------------------------------------------
    # Emotion labels
    # -----------------------------------------------------

    labels = sorted(
        IDX2LABEL.keys()
    )

    names = [
        IDX2LABEL[i]
        for i in labels
    ]


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    predicted_counts = {}

    for class_index in labels:

        emotion = IDX2LABEL[class_index]

        predicted_counts[emotion] = int(
            np.sum(y_pred == class_index)
        )


    print("\n" + "=" * 60)
    print("CLASS DETECTION")
    print("=" * 60)

    for emotion in names:

        count = predicted_counts.get(
            emotion,
            0
        )

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


    # =====================================================
    # METRICS
    # =====================================================

    acc = accuracy_score(
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


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        digits=4,
        output_dict=True,
        zero_division=0
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n" + "=" * 60)
    print("AUDIO MODEL - DEV RESULTS")
    print("=" * 60)

    print(
        f"Accuracy   : {acc:.4f} "
        f"({acc * 100:.2f}%)"
    )

    print(
        f"Macro-F1   : {macro_f1:.4f} "
        f"({macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f} "
        f"({weighted_f1 * 100:.2f}%)"
    )


    # =====================================================
    # PRINT CLASSIFICATION REPORT
    # =====================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            digits=4,
            zero_division=0
        )
    )


    # =====================================================
    # PRINT CONFUSION MATRIX
    # =====================================================

    print("Confusion Matrix:")

    print(
        "             "
        + " ".join(
            f"{n[:8]:>9}"
            for n in names
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{names[i]:>10} "
            + " ".join(
                f"{v:9d}"
                for v in row
            )
        )


    # =====================================================
    # PRINT PREDICTED DISTRIBUTION
    # =====================================================

    print("\nPredicted class distribution:")

    for class_index in labels:

        emotion = IDX2LABEL[class_index]

        count = predicted_counts[emotion]

        print(
            f"{emotion:>10}: {count:4d}"
        )

    print("=" * 60)


    # =====================================================
    # SAVE TO EXCEL
    # =====================================================

    print("\nSaving audio experiment to Excel...")


    # -----------------------------------------------------
    # Create Run
    # -----------------------------------------------------

    run_id = log_run(

        modality="Audio",

        model="Wav2Vec2",

        dataset="MELD",

        train_samples=9988,

        validation_samples=len(df),

        test_samples=2610,

        best_epoch=3,

        accuracy=acc,

        macro_f1=macro_f1,

        weighted_f1=weighted_f1,

        checkpoint=CHECKPOINT,

        status="Complete",

        notes=(
            "Baseline frozen Wav2Vec2 audio "
            "emotion classifier. "
            "Class Detection records actual "
            "predicted classes."
        )

    )


    # -----------------------------------------------------
    # Class-wise results
    # -----------------------------------------------------

    class_results = []

    for emotion in names:

        class_results.append({

            "emotion": emotion,

            "precision": report[emotion][
                "precision"
            ],

            "recall": report[emotion][
                "recall"
            ],

            "f1": report[emotion][
                "f1-score"
            ],

            "support": report[emotion][
                "support"
            ]

        })


    log_class_results(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2",

        class_results=class_results

    )


    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    configuration = {

        "version": "V2",

        "sample_rate": 16000,

        "max_audio_seconds": 3.5,

        "batch_size": AUDIO_BATCH_SIZE,

        "encoder": "Wav2Vec2-base-960h",

        "encoder_frozen": True,

        "number_of_classes": 7,

        "emotion_classes": ", ".join(names),

        "device": str(DEVICE),

        "checkpoint": CHECKPOINT,

        "validation_manifest": MANIFEST

    }


    log_configuration(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2",

        configuration=configuration

    )


    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    log_confusion_matrix(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2",

        matrix=cm,

        emotion_labels=names

    )


    # -----------------------------------------------------
    # Class detection
    # -----------------------------------------------------

    log_class_detection(

        run_id=run_id,

        version="V2",

        modality="Audio",

        model="Wav2Vec2",

        emotion_labels=names,

        predicted_counts=predicted_counts

    )


    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)
    print("AUDIO EXPERIMENT SUCCESSFULLY RECORDED")
    print("=" * 60)

    print(
        f"Run ID       : {run_id}"
    )

    print(
        "Version      : V2"
    )

    print(
        "Modality     : Audio"
    )

    print(
        "Model        : Wav2Vec2"
    )

    print(
        f"Accuracy     : {acc * 100:.2f}%"
    )

    print(
        f"Macro-F1     : {macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1  : {weighted_f1 * 100:.2f}%"
    )

    print(
        "Excel Report : "
        "results\\experiment_report.xlsx"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()