import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import EMOTIONS, NUM_CLASSES, DEVICE
from utils.experiment_logger import (
    log_run,
    log_epoch,
    log_class_results,
    log_configuration,
    log_confusion_matrix,
    log_class_detection
)


# =========================================================
# V6 CONFIGURATION
# =========================================================

SEED = 42

BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 5
GRAD_CLIP = 1.0

RUN_VERSION = "V6"
MODEL_NAME = "TextV2-Audio-Video-Fusion"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "v6_textv2_fusion_best.pt"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "v6_textv2_fusion_results.txt"
)


# =========================================================
# REPRODUCIBILITY
# =========================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# CACHE PATHS
# =========================================================

CACHE_PATHS = {

    "train": {
        "text": PROJECT_ROOT / "data" / "text_v2_feature_cache_train.pt",
        "audio": PROJECT_ROOT / "data" / "audio_feature_cache_train.pt",
        "video": PROJECT_ROOT / "data" / "fer_vit_feature_cache_train.pt",
    },

    "dev": {
        "text": PROJECT_ROOT / "data" / "text_v2_feature_cache_dev.pt",
        "audio": PROJECT_ROOT / "data" / "audio_feature_cache_dev.pt",
        "video": PROJECT_ROOT / "data" / "fer_vit_feature_cache_dev.pt",
    },

    "test": {
        "text": PROJECT_ROOT / "data" / "text_v2_feature_cache_test.pt",
        "audio": PROJECT_ROOT / "data" / "audio_feature_cache_test.pt",
        "video": PROJECT_ROOT / "data" / "fer_vit_feature_cache_test.pt",
    }
}


# =========================================================
# CACHE LOADING
# =========================================================

def load_text_cache(path):

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    features = cache["features"].float()
    labels = cache["labels"].long()
    clip_ids = cache["clip_ids"]

    return features, labels, clip_ids


def load_audio_cache(path):

    features = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    ).float()

    return features


def load_video_cache(path):

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    features = cache["features"].float()
    labels = cache["labels"]

    if labels.dtype != torch.long:

        if labels.dtype.is_floating_point:
            labels = labels.long()

        else:
            labels = torch.tensor(
                [
                    EMOTIONS.index(str(x))
                    if str(x) in EMOTIONS
                    else int(x)
                    for x in labels
                ],
                dtype=torch.long
            )

    return features, labels


def load_split(split):

    print("\n" + "=" * 70)
    print(f"LOADING {split.upper()} FEATURES")
    print("=" * 70)

    text_path = CACHE_PATHS[split]["text"]
    audio_path = CACHE_PATHS[split]["audio"]
    video_path = CACHE_PATHS[split]["video"]

    for path in [text_path, audio_path, video_path]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required cache not found:\n{path}"
            )

    text_features, text_labels, clip_ids = load_text_cache(
        text_path
    )

    audio_features = load_audio_cache(
        audio_path
    )

    video_features, video_labels = load_video_cache(
        video_path
    )

    print("Text  :", tuple(text_features.shape))
    print("Audio :", tuple(audio_features.shape))
    print("Video :", tuple(video_features.shape))
    print("Labels:", tuple(text_labels.shape))

    # -----------------------------------------------------
    # Sample count
    # -----------------------------------------------------

    counts = [
        len(text_features),
        len(audio_features),
        len(video_features),
        len(text_labels),
        len(video_labels)
    ]

    if len(set(counts)) != 1:

        raise ValueError(
            f"Sample count mismatch in {split}: {counts}"
        )

    # -----------------------------------------------------
    # Label alignment
    # -----------------------------------------------------

    if not torch.equal(text_labels, video_labels):

        raise ValueError(
            f"Text/video label mismatch in {split}"
        )

    print("Label alignment: PASS")

    # -----------------------------------------------------
    # Concatenate modalities
    # -----------------------------------------------------

    features = torch.cat(
        [
            text_features,
            audio_features,
            video_features
        ],
        dim=1
    )

    print(
        "Fused feature shape:",
        tuple(features.shape)
    )

    return features, text_labels, clip_ids


# =========================================================
# MODEL
# =========================================================

class V6FusionModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(2304),

            nn.Linear(2304, 512),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(features)

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            predictions = (
                logits.argmax(dim=1)
                .cpu()
                .tolist()
            )

            y_pred.extend(predictions)
            y_true.extend(labels.cpu().tolist())

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
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "true": y_true,
        "pred": y_pred
    }


# =========================================================
# TRAINING
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    y_true = []
    y_pred = []

    for features, labels in loader:

        features = features.to(
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

        logits = model(features)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_samples += batch_size

        y_true.extend(
            labels.detach().cpu().tolist()
        )

        y_pred.extend(
            logits.argmax(dim=1)
            .detach()
            .cpu()
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
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="weighted",
        zero_division=0
    )

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("V6 TEXT V2 + AUDIO + VIDEO FUSION")
    print("=" * 70)

    print("Version       :", RUN_VERSION)
    print("Model         :", MODEL_NAME)
    print("Device        :", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU           :",
            torch.cuda.get_device_name(0)
        )

    print("Batch size    :", BATCH_SIZE)
    print("Max epochs    :", MAX_EPOCHS)
    print("Learning rate :", LEARNING_RATE)
    print("Weight decay  :", WEIGHT_DECAY)
    print("Label smooth. :", LABEL_SMOOTHING)

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    train_features, train_labels, train_ids = load_split(
        "train"
    )

    dev_features, dev_labels, dev_ids = load_split(
        "dev"
    )

    test_features, test_labels, test_ids = load_split(
        "test"
    )

    # -----------------------------------------------------
    # Final alignment checks
    # -----------------------------------------------------

    if not torch.equal(
        train_labels,
        torch.tensor(
            train_labels.tolist(),
            dtype=torch.long
        )
    ):
        raise ValueError("Train label integrity failed")

    print("\nSample counts:")

    print(
        "Train:",
        len(train_labels)
    )

    print(
        "Dev  :",
        len(dev_labels)
    )

    print(
        "Test :",
        len(test_labels)
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_dataset = TensorDataset(
        train_features,
        train_labels
    )

    dev_dataset = TensorDataset(
        dev_features,
        dev_labels
    )

    test_dataset = TensorDataset(
        test_features,
        test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = V6FusionModel().to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "\nTotal parameters:",
        total_parameters
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    history = []

    print("\n" + "=" * 70)
    print("STARTING V6 TRAINING")
    print("=" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):

        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        dev_result = evaluate(
            model,
            dev_loader,
            criterion
        )

        scheduler.step(
            dev_result["macro_f1"]
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"Train Loss: {train_result['loss']:.4f} | "
            f"Train Acc: {train_result['accuracy']:.4f} | "
            f"Train Macro-F1: {train_result['macro_f1']:.4f} | "
            f"Train Weighted-F1: {train_result['weighted_f1']:.4f}"
        )

        print(
            f"Dev Loss: {dev_result['loss']:.4f} | "
            f"Dev Acc: {dev_result['accuracy']:.4f} | "
            f"Dev Macro-F1: {dev_result['macro_f1']:.4f} | "
            f"Dev Weighted-F1: {dev_result['weighted_f1']:.4f}"
        )

        print(
            f"Learning Rate: {current_lr:.8f}"
        )

        history.append({
            "epoch": epoch,
            "train": train_result,
            "dev": dev_result,
            "lr": current_lr
        })

        # -------------------------------------------------
        # Best model by DEV MACRO-F1
        # -------------------------------------------------

        if dev_result["macro_f1"] > best_macro_f1:

            best_macro_f1 = dev_result["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "dev_accuracy": dev_result["accuracy"],
                    "dev_macro_f1": dev_result["macro_f1"],
                    "dev_weighted_f1": dev_result["weighted_f1"],
                    "config": {
                        "version": RUN_VERSION,
                        "model": MODEL_NAME,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                    }
                },
                CHECKPOINT
            )

            print(
                f"  -> Saved best checkpoint "
                f"(Dev Macro-F1={best_macro_f1:.4f})"
            )

        else:

            patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:

                print(
                    "\nEarly stopping triggered."
                )

                break

    # -----------------------------------------------------
    # Load best checkpoint
    # -----------------------------------------------------

    print(
        "\nLoading best V6 checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # -----------------------------------------------------
    # Final Dev
    # -----------------------------------------------------

    final_dev = evaluate(
        model,
        dev_loader,
        criterion
    )

    # -----------------------------------------------------
    # Final Test
    # -----------------------------------------------------

    final_test = evaluate(
        model,
        test_loader,
        criterion
    )

    # -----------------------------------------------------
    # Test metrics
    # -----------------------------------------------------

    test_true = final_test["true"]
    test_pred = final_test["pred"]

    cm = confusion_matrix(
        test_true,
        test_pred,
        labels=list(range(NUM_CLASSES))
    )

    report = classification_report(
        test_true,
        test_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("V6 FINAL DEV RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : {final_dev['accuracy']:.4f} "
        f"({final_dev['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Macro-F1    : {final_dev['macro_f1']:.4f} "
        f"({final_dev['macro_f1'] * 100:.2f}%)"
    )

    print(
        f"Weighted-F1 : {final_dev['weighted_f1']:.4f} "
        f"({final_dev['weighted_f1'] * 100:.2f}%)"
    )

    print("\n" + "=" * 70)
    print("V6 FINAL TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy    : {final_test['accuracy']:.4f} "
        f"({final_test['accuracy'] * 100:.2f}%)"
    )

    print(
        f"Macro-F1    : {final_test['macro_f1']:.4f} "
        f"({final_test['macro_f1'] * 100:.2f}%)"
    )

    print(
        f"Weighted-F1 : {final_test['weighted_f1']:.4f} "
        f"({final_test['weighted_f1'] * 100:.2f}%)"
    )

    print("\nTest Classification Report:")

    print(
        classification_report(
            test_true,
            test_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=EMOTIONS,
            zero_division=0
        )
    )

    # -----------------------------------------------------
    # Class detection
    # -----------------------------------------------------

    predicted_counts = {}

    print("=" * 70)
    print("V6 TEST 7-CLASS DETECTION")
    print("=" * 70)

    detected_classes = 0

    for i, emotion in enumerate(EMOTIONS):

        count = test_pred.count(i)

        predicted_counts[emotion] = count

        if count > 0:

            detected_classes += 1

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
    # Save text result
    # -----------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("V6 TEXT V2 + AUDIO + VIDEO FUSION\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Best Epoch: {best_epoch}\n"
        )

        f.write(
            f"Dev Accuracy: {final_dev['accuracy']:.6f}\n"
        )

        f.write(
            f"Dev Macro-F1: {final_dev['macro_f1']:.6f}\n"
        )

        f.write(
            f"Dev Weighted-F1: "
            f"{final_dev['weighted_f1']:.6f}\n\n"
        )

        f.write(
            f"Test Accuracy: {final_test['accuracy']:.6f}\n"
        )

        f.write(
            f"Test Macro-F1: {final_test['macro_f1']:.6f}\n"
        )

        f.write(
            f"Test Weighted-F1: "
            f"{final_test['weighted_f1']:.6f}\n\n"
        )

        f.write(
            classification_report(
                test_true,
                test_pred,
                labels=list(range(NUM_CLASSES)),
                target_names=EMOTIONS,
                zero_division=0
            )
        )

        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    # -----------------------------------------------------
    # Excel logging
    # -----------------------------------------------------

    run_id = log_run(
        modality="Multimodal",
        model=MODEL_NAME,
        dataset="MELD",
        train_samples=len(train_labels),
        validation_samples=len(dev_labels),
        test_samples=len(test_labels),
        best_epoch=best_epoch,
        accuracy=final_test["accuracy"],
        macro_f1=final_test["macro_f1"],
        weighted_f1=final_test["weighted_f1"],
        checkpoint=str(CHECKPOINT),
        status="Complete",
        notes=(
            "V6 experiment. Text V2 features + existing "
            "Audio Wav2Vec2 features + existing Video FER-ViT "
            "features. Feature-level concatenation. "
            "Best checkpoint selected using Dev Macro-F1."
        )
    )

    # -----------------------------------------------------
    # Epoch logging
    # -----------------------------------------------------

    for item in history:

        log_epoch(
            run_id=run_id,
            modality="Multimodal",
            model=MODEL_NAME,
            epoch=item["epoch"],
            train_loss=item["train"]["loss"],
            train_accuracy=item["train"]["accuracy"],
            val_loss=item["dev"]["loss"],
            val_accuracy=item["dev"]["accuracy"],
            macro_f1=item["dev"]["macro_f1"]
        )

    # -----------------------------------------------------
    # Class results
    # -----------------------------------------------------

    class_results = []

    for emotion in EMOTIONS:

        result = report[emotion]

        class_results.append({
            "emotion": emotion,
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1-score"],
            "support": result["support"]
        })

    log_class_results(
        run_id=run_id,
        modality="Multimodal",
        model=MODEL_NAME,
        class_results=class_results
    )

    # -----------------------------------------------------
    # Configuration
    # -----------------------------------------------------

    configuration = {
        "version": RUN_VERSION,
        "model": MODEL_NAME,
        "text_features": "Text V2 DistilBERT",
        "audio_features": "Wav2Vec2",
        "video_features": "FER-ViT",
        "text_dim": 768,
        "audio_dim": 768,
        "video_dim": 768,
        "fused_dim": 2304,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing": LABEL_SMOOTHING,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "seed": SEED,
        "device": str(DEVICE),
        "best_epoch": best_epoch,
        "checkpoint": str(CHECKPOINT)
    }

    log_configuration(
        run_id=run_id,
        modality="Multimodal",
        model=MODEL_NAME,
        configuration=configuration
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    log_confusion_matrix(
        run_id=run_id,
        modality="Multimodal",
        model=MODEL_NAME,
        matrix=cm,
        emotion_labels=EMOTIONS
    )

    # -----------------------------------------------------
    # Class detection
    # -----------------------------------------------------

    log_class_detection(
        run_id=run_id,
        version=RUN_VERSION,
        modality="Multimodal",
        model=MODEL_NAME,
        emotion_labels=EMOTIONS,
        predicted_counts=predicted_counts
    )

    # -----------------------------------------------------
    # Final
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("V6 TRAINING COMPLETED")
    print("=" * 70)

    print("Run ID       :", run_id)
    print("Best epoch   :", best_epoch)

    print(
        f"Test Accuracy: "
        f"{final_test['accuracy'] * 100:.2f}%"
    )

    print(
        f"Test Macro-F1: "
        f"{final_test['macro_f1'] * 100:.2f}%"
    )

    print(
        f"Test Weighted-F1: "
        f"{final_test['weighted_f1'] * 100:.2f}%"
    )

    print(
        f"Classes detected: "
        f"{detected_classes}/7"
    )

    print(
        "Checkpoint:",
        CHECKPOINT
    )

    print(
        "Results:",
        RESULTS_FILE
    )

    print(
        "Excel:",
        PROJECT_ROOT
        / "results"
        / "experiment_report.xlsx"
    )


if __name__ == "__main__":
    main()