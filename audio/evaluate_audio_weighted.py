import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import (
    DEVICE,
    AUDIO_BATCH_SIZE,
    IDX2LABEL,
    AUDIO_SAMPLE_RATE,
    WAV2VEC2_MODEL_NAME,
    NUM_CLASSES
)

from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel

from utils.experiment_logger import (
    log_run,
    log_class_results,
    log_configuration,
    log_confusion_matrix
)


MANIFEST = "data/manifests/dev.csv"
CHECKPOINT = "checkpoints/audio_model_weighted.pt"


def main():

    print("Loading dev manifest...")
    df = load_manifest(MANIFEST)

    dataset = AudioEmotionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Loading audio model...")
    model = AudioEmotionModel(freeze_encoder=True).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    y_true = []
    y_pred = []
    clip_ids = []

    print("Evaluating...")

    with torch.no_grad():

        for input_values, labels, ids in loader:

            input_values = input_values.to(DEVICE)

            logits = model(input_values)

            preds = logits.argmax(dim=-1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            clip_ids.extend(ids)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = sorted(IDX2LABEL.keys())
    names = [IDX2LABEL[i] for i in labels]

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    acc = accuracy_score(y_true, y_pred)

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

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # ---------------------------------------------------------
    # DISPLAY RESULTS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("AUDIO MODEL - DEV RESULTS")
    print("=" * 60)

    print(f"Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro-F1   : {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Weighted-F1: {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")

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

    print("Confusion Matrix:")

    print(
        "             "
        + " ".join(f"{n[:8]:>9}" for n in names)
    )

    for i, row in enumerate(cm):

        print(
            f"{names[i]:>10} "
            + " ".join(f"{v:9d}" for v in row)
        )

    print("\nPredicted class distribution:")

    unique, counts = np.unique(
        y_pred,
        return_counts=True
    )

    for i in labels:

        count = (
            counts[unique.tolist().index(i)]
            if i in unique
            else 0
        )

        print(
            f"{IDX2LABEL[i]:>10}: {count:4d}"
        )

    print("=" * 60)

    # ---------------------------------------------------------
    # EXCEL LOGGING
    # ---------------------------------------------------------

    print("\nSaving weighted audio experiment to Excel...")

    run_id = log_run(
        modality="Audio",
        model="Wav2Vec2 + Class Weights",
        dataset="MELD",
        train_samples=9988,
        validation_samples=len(df),
        test_samples=2610,
        best_epoch=1,
        accuracy=acc,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        checkpoint=CHECKPOINT,
        status="Completed",
        notes=(
            "Frozen Wav2Vec2 encoder with inverse-frequency "
            "class weights. Evaluated on MELD dev set."
        )
    )

    # ---------------------------------------------------------
    # CLASS-WISE RESULTS
    # ---------------------------------------------------------

    class_results = []

    for emotion in names:

        class_results.append({
            "emotion": emotion,
            "precision": report[emotion]["precision"],
            "recall": report[emotion]["recall"],
            "f1": report[emotion]["f1-score"],
            "support": report[emotion]["support"]
        })

    log_class_results(
        run_id,
        modality="Audio",
        model="Wav2Vec2 + Class Weights",
        class_results=class_results
    )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------

    configuration = {
        "sample_rate": AUDIO_SAMPLE_RATE,
        "audio_batch_size": AUDIO_BATCH_SIZE,
        "wav2vec2_model": WAV2VEC2_MODEL_NAME,
        "encoder_frozen": True,
        "number_of_classes": NUM_CLASSES,
        "device": str(DEVICE),
        "checkpoint": CHECKPOINT,
        "validation_manifest": MANIFEST,
        "training_samples": 9988,
        "validation_samples": len(df),
        "test_samples": 2610,
        "best_epoch": 1,
        "class_weighting": "Inverse-frequency"
    }

    log_configuration(
        run_id,
        modality="Audio",
        model="Wav2Vec2 + Class Weights",
        configuration=configuration
    )

    # ---------------------------------------------------------
    # CONFUSION MATRIX
    # ---------------------------------------------------------

    log_confusion_matrix(
        run_id,
        modality="Audio",
        model="Wav2Vec2 + Class Weights",
        matrix=cm,
        emotion_labels=names
    )

    print(f"\n[Excel] Experiment logged: {run_id}")
    print("[Excel] Audio weighted experiment successfully recorded.")
    print("[Excel] File: results\\experiment_report.xlsx")


if __name__ == "__main__":
    main()
