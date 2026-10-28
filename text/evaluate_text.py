import sys
import os

# Allow imports from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import pandas as pd

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm


# =========================================================
# CONFIG
# =========================================================

from config import (
    DEVICE,
    TEXT_BATCH_SIZE,
    IDX2LABEL,
    NUM_CLASSES,
    TEXT_CKPT,
    TEXT_MODEL_NAME,
    MAX_TEXT_LEN,
    LR,
    EPOCHS,
    EARLY_STOP_PATIENCE
)


# =========================================================
# DATASET / MODEL
# =========================================================

from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


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
# MANIFEST
# =========================================================

MANIFEST = "data/manifests/dev.csv"


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    print("Loading dev manifest...")

    df = load_manifest(MANIFEST)

    print(f"Validation samples: {len(df)}")


    # -----------------------------------------------------
    # Create DataLoader
    # -----------------------------------------------------

    loader = DataLoader(
        TextEmotionDataset(df),
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    print("Loading text model...")

    model = TextEmotionModel().to(DEVICE)

    checkpoint = os.path.join(
        TEXT_CKPT,
        "pytorch_model.bin"
    )

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    state_dict = torch.load(
        checkpoint,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    y_true = []
    y_pred = []

    print("Evaluating...")

    with torch.no_grad():

        for input_ids, attention_mask, labels, _ in tqdm(loader):

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            logits = model(
                input_ids,
                attention_mask
            )

            preds = logits.argmax(
                dim=-1
            ).cpu().numpy()

            y_pred.extend(
                preds.tolist()
            )

            y_true.extend(
                labels.numpy().tolist()
            )


    # -----------------------------------------------------
    # Emotion labels
    # -----------------------------------------------------

    labels = list(range(NUM_CLASSES))

    names = [
        IDX2LABEL[i]
        for i in labels
    ]


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    predicted_counts = {}

    for class_index, emotion in IDX2LABEL.items():

        predicted_counts[emotion] = (
            y_pred.count(class_index)
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


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
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
    print("TEXT MODEL - DEV RESULTS")
    print("=" * 60)

    print(
        f"Accuracy   : {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
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
            zero_division=0
        )
    )


    # =====================================================
    # PREPARE CLASS RESULTS FOR EXCEL
    # =====================================================

    class_results = []

    for emotion in names:

        result = report[emotion]

        class_results.append({

            "emotion": emotion,

            "precision": result["precision"],

            "recall": result["recall"],

            "f1": result["f1"],

            "support": result["support"]

        })


    # =====================================================
    # CONFIGURATION INFORMATION
    # =====================================================

    configuration = {

        "version": "V1",

        "modality": "Text",

        "model_name": TEXT_MODEL_NAME,

        "max_text_length": MAX_TEXT_LEN,

        "batch_size": TEXT_BATCH_SIZE,

        "learning_rate": LR,

        "epochs_configured": EPOCHS,

        "early_stop_patience": EARLY_STOP_PATIENCE,

        "device": str(DEVICE),

        "checkpoint": checkpoint,

        "validation_manifest": MANIFEST,

        "number_of_classes": NUM_CLASSES,

        "emotion_classes": ", ".join(names)

    }


    # =====================================================
    # CREATE EXCEL RUN
    # =====================================================

    print("\nLogging experiment to Excel...")

    run_id = log_run(

        modality="Text",

        model="DistilBERT",

        dataset="MELD",

        train_samples=9988,

        validation_samples=len(df),

        test_samples=2610,

        best_epoch=2,

        accuracy=accuracy,

        macro_f1=macro_f1,

        weighted_f1=weighted_f1,

        checkpoint=checkpoint,

        status="Complete",

        notes=(
            "Text modality evaluation on MELD dev set. "
            "Class Detection records actual predicted classes."
        )

    )


    # =====================================================
    # LOG CLASS RESULTS
    # =====================================================

    log_class_results(

        run_id=run_id,

        modality="Text",

        model="DistilBERT",

        class_results=class_results

    )


    # =====================================================
    # LOG CONFIGURATION
    # =====================================================

    log_configuration(

        run_id=run_id,

        modality="Text",

        model="DistilBERT",

        configuration=configuration

    )


    # =====================================================
    # LOG CONFUSION MATRIX
    # =====================================================

    log_confusion_matrix(

        run_id=run_id,

        modality="Text",

        model="DistilBERT",

        matrix=cm,

        emotion_labels=names

    )


    # =====================================================
    # LOG CLASS DETECTION
    # =====================================================

    log_class_detection(

        run_id=run_id,

        version="V1",

        modality="Text",

        model="DistilBERT",

        emotion_labels=names,

        predicted_counts=predicted_counts

    )


    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 60)
    print("TEXT EXPERIMENT SUCCESSFULLY RECORDED")
    print("=" * 60)

    print(
        f"Run ID        : {run_id}"
    )

    print(
        "Version       : V1"
    )

    print(
        "Modality      : Text"
    )

    print(
        "Model         : DistilBERT"
    )

    print(
        f"Accuracy      : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro-F1      : {macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1   : {weighted_f1 * 100:.2f}%"
    )

    print(
        "Excel Report  : "
        "results\\experiment_report.xlsx"
    )


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()