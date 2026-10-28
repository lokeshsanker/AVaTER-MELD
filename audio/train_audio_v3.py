import os
import sys
import random
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn
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
    NUM_CLASSES,
    IDX2LABEL
)

from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel

from utils.experiment_logger import (
    log_run,
    log_epoch,
    log_class_results,
    log_configuration,
    log_confusion_matrix,
    log_class_detection
)


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_MANIFEST = "data/manifests/train.csv"
DEV_MANIFEST = "data/manifests/dev.csv"

CHECKPOINT_DIR = "checkpoints/audio_v3"
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "audio_v3_best.pt"
)

SEED = 42

MAX_EPOCHS = 20
PATIENCE = 4

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 1.0

USE_AMP = False


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# CLASS WEIGHTS
# =========================================================

def calculate_class_weights(labels):

    labels = np.asarray(labels)

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    print("\nTraining class distribution:")
    print("=" * 60)

    for i in range(NUM_CLASSES):

        emotion = IDX2LABEL[i]

        print(
            f"{emotion:<10}: {counts[i]}"
        )

    # -----------------------------------------------------
    # Balanced inverse-frequency weights
    #
    # weight_i = N / (num_classes * class_count_i)
    # -----------------------------------------------------

    total_samples = len(labels)

    weights = (
        total_samples
        /
        (
            NUM_CLASSES
            *
            np.maximum(counts, 1)
        )
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    print("\nClass weights:")
    print("=" * 60)

    for i in range(NUM_CLASSES):

        emotion = IDX2LABEL[i]

        print(
            f"{emotion:<10}: "
            f"{weights[i].item():.4f}"
        )

    print("=" * 60)

    return weights


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for input_values, labels, ids in loader:

            input_values = input_values.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                input_values
            )

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            predictions = logits.argmax(
                dim=-1
            )

            y_true.extend(
                labels.cpu().numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    avg_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        np.array(y_true),
        np.array(y_pred)
    )


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    for batch_idx, (
        input_values,
        labels,
        ids
    ) in enumerate(loader):

        input_values = input_values.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        if USE_AMP:

            with torch.cuda.amp.autocast():

                logits = model(
                    input_values
                )

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            logits = model(
                input_values
            )

            loss = criterion(
                logits,
                labels
            )

            # -------------------------------------------------
            # Safety check for NaN / Inf
            # -------------------------------------------------

            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss detected: "
                    f"{loss.item()}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        predictions = logits.argmax(
            dim=-1
        )

        y_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    avg_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print("AUDIO V3 - CLASS-WEIGHTED WAV2VEC2")
    print("=" * 70)

    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device        : {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory    : "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    print("=" * 70)


    # =====================================================
    # LOAD MANIFESTS
    # =====================================================

    print("\nLoading manifests...")

    train_df = load_manifest(
        TRAIN_MANIFEST
    )

    dev_df = load_manifest(
        DEV_MANIFEST
    )

    print(
        f"Training samples  : {len(train_df)}"
    )

    print(
        f"Validation samples: {len(dev_df)}"
    )


    # =====================================================
    # DATASETS
    # =====================================================

    train_dataset = AudioEmotionDataset(
        train_df
    )

    dev_dataset = AudioEmotionDataset(
        dev_df
    )


    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    train_labels = train_df["label"].tolist()

    # -----------------------------------------------------
    # Handle string labels if present
    # -----------------------------------------------------

    if len(train_labels) > 0 and isinstance(
        train_labels[0],
        str
    ):

        label_to_index = {
            label: index
            for index, label
            in IDX2LABEL.items()
        }

        train_labels = [
            label_to_index[label]
            for label in train_labels
        ]

    else:

        train_labels = [
            int(label)
            for label in train_labels
        ]

    class_weights = calculate_class_weights(
        train_labels
    ).to(DEVICE)


    # =====================================================
    # MODEL
    # =====================================================

    print("\nLoading Wav2Vec2 model...")

    model = AudioEmotionModel(
        freeze_encoder=True
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )


    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # =====================================================
    # AMP
    # =====================================================

    scaler = torch.cuda.amp.GradScaler(
        enabled=USE_AMP
    )


    # =====================================================
    # TRAINING CONFIG
    # =====================================================

    print("\nTraining configuration:")
    print("=" * 70)

    print(
        f"Batch size        : {AUDIO_BATCH_SIZE}"
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
        f"Gradient clipping : {GRAD_CLIP}"
    )

    print(
        f"Class-weighted CE : True"
    )

    print(
        f"Encoder frozen    : True"
    )

    print(
        f"AMP               : {USE_AMP}"
    )

    print("=" * 70)


    # =====================================================
    # TRAIN
    # =====================================================

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    print("\n" + "=" * 70)
    print("STARTING AUDIO V3 TRAINING")
    print("=" * 70)

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        # -------------------------------------------------
        # Training
        # -------------------------------------------------

        (
            train_loss,
            train_acc,
            train_macro_f1,
            train_weighted_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        (
            dev_loss,
            dev_acc,
            dev_macro_f1,
            dev_weighted_f1,
            _,
            _
        ) = evaluate(
            model,
            dev_loader,
            criterion
        )

        print(
            f"Train Loss       : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : {train_acc:.4f}"
        )

        print(
            f"Train Macro-F1   : {train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted-F1: {train_weighted_f1:.4f}"
        )

        print(
            f"Dev Loss         : {dev_loss:.4f}"
        )

        print(
            f"Dev Accuracy     : {dev_acc:.4f}"
        )

        print(
            f"Dev Macro-F1     : {dev_macro_f1:.4f}"
        )

        print(
            f"Dev Weighted-F1  : {dev_weighted_f1:.4f}"
        )


        # -------------------------------------------------
        # Save history
        # -------------------------------------------------

        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_acc,

            "train_macro_f1": train_macro_f1,

            "train_weighted_f1": train_weighted_f1,

            "dev_loss": dev_loss,

            "dev_accuracy": dev_acc,

            "dev_macro_f1": dev_macro_f1,

            "dev_weighted_f1": dev_weighted_f1

        })


        # -------------------------------------------------
        # Best checkpoint by Macro-F1
        # -------------------------------------------------

        if dev_macro_f1 > best_macro_f1:

            best_macro_f1 = dev_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"✓ Best model saved "
                f"(Dev Macro-F1 = "
                f"{dev_macro_f1:.4f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # -------------------------------------------------
        # Early stopping
        # -------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break


    # =====================================================
    # LOAD BEST MODEL
    # =====================================================

    print("\n" + "=" * 70)
    print("LOADING BEST AUDIO V3 CHECKPOINT")
    print("=" * 70)

    print(
        f"Best epoch     : {best_epoch}"
    )

    print(
        f"Best Dev Macro-F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint     : "
        f"{CHECKPOINT_PATH}"
    )

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state
    )

    model.eval()


    # =====================================================
    # FINAL DEV EVALUATION
    # =====================================================

    (
        final_loss,
        final_acc,
        final_macro_f1,
        final_weighted_f1,
        y_true,
        y_pred
    ) = evaluate(
        model,
        dev_loader,
        criterion
    )


    # =====================================================
    # CLASS LABELS
    # =====================================================

    labels = list(
        range(NUM_CLASSES)
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

        emotion = IDX2LABEL[
            class_index
        ]

        predicted_counts[emotion] = int(
            np.sum(
                y_pred == class_index
            )
        )


    print("\n" + "=" * 70)
    print("AUDIO V3 FINAL DEV RESULTS")
    print("=" * 70)

    print(
        f"Accuracy   : "
        f"{final_acc:.4f} "
        f"({final_acc * 100:.2f}%)"
    )

    print(
        f"Macro-F1   : "
        f"{final_macro_f1:.4f} "
        f"({final_macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1: "
        f"{final_weighted_f1:.4f} "
        f"({final_weighted_f1 * 100:.2f}%)"
    )


    # =====================================================
    # CLASS DETECTION PRINT
    # =====================================================

    print("\nClass Detection:")

    detected_classes = 0

    for emotion in names:

        count = predicted_counts[
            emotion
        ]

        if count > 0:

            detected_classes += 1

            print(
                f"{emotion:<10}: "
                f"Detected ({count})"
            )

        else:

            print(
                f"{emotion:<10}: "
                f"Not detected (0)"
            )


    print(
        f"\nClasses detected: "
        f"{detected_classes}/{NUM_CLASSES}"
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
        zero_division=0
    )

    print("\nClassification Report:")
    print(report)


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

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
    # PREDICTION DISTRIBUTION
    # =====================================================

    print("\nPredicted class distribution:")

    for class_index in labels:

        emotion = IDX2LABEL[
            class_index
        ]

        print(
            f"{emotion:>10}: "
            f"{predicted_counts[emotion]:4d}"
        )


    # =====================================================
    # EXCEL LOGGING
    # =====================================================

    print(
        "\nSaving Audio V3 experiment to Excel..."
    )

    run_id = log_run(

        modality="Audio",

        model="Wav2Vec2 + "
              "Class-Weighted Loss",

        dataset="MELD",

        train_samples=len(
            train_df
        ),

        validation_samples=len(
            dev_df
        ),

        test_samples=2610,

        best_epoch=best_epoch,

        accuracy=final_acc,

        macro_f1=final_macro_f1,

        weighted_f1=final_weighted_f1,

        checkpoint=CHECKPOINT_PATH,

        status="Complete",

        notes=(
            "Audio V3. "
            "Frozen Wav2Vec2 encoder with "
            "attention pooling and "
            "class-weighted CrossEntropyLoss. "
            "Checkpoint selected by "
            "validation Macro-F1."
        )

    )


    # =====================================================
    # EPOCH LOGGING
    # =====================================================

    for item in history:

        log_epoch(

            run_id=run_id,

            epoch=item["epoch"],

            train_loss=item["train_loss"],

            train_accuracy=item["train_accuracy"],

            train_macro_f1=item["train_macro_f1"],

            train_weighted_f1=item["train_weighted_f1"],

            validation_loss=item["dev_loss"],

            validation_accuracy=item["dev_accuracy"],

            validation_macro_f1=item["dev_macro_f1"],

            validation_weighted_f1=item["dev_weighted_f1"]

        )


    # =====================================================
    # CLASS-WISE RESULTS
    # =====================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0
    )

    class_results = []

    for emotion in names:

        class_results.append({

            "emotion": emotion,

            "precision": report_dict[
                emotion
            ]["precision"],

            "recall": report_dict[
                emotion
            ]["recall"],

            "f1": report_dict[
                emotion
            ]["f1-score"],

            "support": report_dict[
                emotion
            ]["support"]

        })


    log_class_results(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2 + "
              "Class-Weighted Loss",

        class_results=class_results

    )


    # =====================================================
    # CONFIGURATION LOGGING
    # =====================================================

    configuration = {

        "version": "V3",

        "sample_rate": 16000,

        "max_audio_seconds": 3.5,

        "batch_size": AUDIO_BATCH_SIZE,

        "encoder":
            "facebook/wav2vec2-base-960h",

        "encoder_frozen": True,

        "attention_pooling": True,

        "loss":
            "Class-weighted CrossEntropyLoss",

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "gradient_clipping": GRAD_CLIP,

        "max_epochs": MAX_EPOCHS,

        "early_stopping_patience":
            PATIENCE,

        "number_of_classes":
            NUM_CLASSES,

        "emotion_classes":
            ", ".join(names),

        "device": str(DEVICE),

        "checkpoint":
            CHECKPOINT_PATH,

        "train_manifest":
            TRAIN_MANIFEST,

        "validation_manifest":
            DEV_MANIFEST

    }


    log_configuration(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2 + "
              "Class-Weighted Loss",

        configuration=configuration

    )


    # =====================================================
    # CONFUSION MATRIX LOGGING
    # =====================================================

    log_confusion_matrix(

        run_id=run_id,

        modality="Audio",

        model="Wav2Vec2 + "
              "Class-Weighted Loss",

        matrix=cm,

        emotion_labels=names

    )


    # =====================================================
    # CLASS DETECTION LOGGING
    # =====================================================

    log_class_detection(

        run_id=run_id,

        version="V3",

        modality="Audio",

        model="Wav2Vec2 + "
              "Class-Weighted Loss",

        emotion_labels=names,

        predicted_counts=predicted_counts

    )


    # =====================================================
    # FINAL
    # =====================================================

    print("\n" + "=" * 70)
    print("AUDIO V3 EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Run ID        : {run_id}"
    )

    print(
        f"Best Epoch    : {best_epoch}"
    )

    print(
        f"Accuracy      : "
        f"{final_acc * 100:.2f}%"
    )

    print(
        f"Macro-F1      : "
        f"{final_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1   : "
        f"{final_weighted_f1 * 100:.2f}%"
    )

    print(
        f"Classes       : "
        f"{detected_classes}/{NUM_CLASSES}"
    )

    print(
        f"Checkpoint    : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "Excel Report  : "
        "results\\experiment_report.xlsx"
    )

    print("=" * 70)


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()