
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import AVaTERMELDV2

DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULT_DIR = ROOT / "results" / "avater_meld_v2"
CHECKPOINT_PATH = RESULT_DIR / "best_model.pt"

LABELS = [
    "neutral", "joy", "sadness", "anger",
    "fear", "disgust", "surprise"
]
NUM_CLASSES = len(LABELS)
BATCH_SIZE = 64

FEATURE_FILES = {
    split: {
        "text": DATA_DIR / "bert_features" /
                f"bert_{split}_features.npz",
        "audio": DATA_DIR / "yamnet_features" /
                 f"yamnet_{split}_features.npz",
        "video": DATA_DIR / "deepface_features" /
                 f"deepface_{split}_features.npz",
    }
    for split in ("train", "dev", "test")
}


def load_split(split):
    manifest_path = MANIFEST_DIR / f"{split}.csv"
    manifest = pd.read_csv(manifest_path)

    manifest_ids = manifest["clip_id"].astype(str).to_numpy()
    manifest_labels = manifest["label"].astype(str).to_numpy()

    features = {}

    for modality, path in FEATURE_FILES[split].items():
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file: {path}")

        with np.load(path, allow_pickle=True) as data:
            x = np.asarray(data["features"], dtype=np.float32)
            ids = data["clip_ids"].astype(str)
            labels = data["labels"].astype(str)

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in {path}")

        if not np.array_equal(ids, manifest_ids):
            raise ValueError(
                f"{split}/{modality}: clip IDs do not match manifest"
            )

        if not np.array_equal(labels, manifest_labels):
            raise ValueError(
                f"{split}/{modality}: labels do not match manifest"
            )

        features[modality] = x

    y = np.asarray(
        [LABELS.index(label) for label in manifest_labels],
        dtype=np.int64
    )

    return features, y, manifest_ids


def standardize(train, test):
    result = {}

    for modality in ("text", "audio", "video"):
        mean = train[modality].mean(axis=0, keepdims=True)
        std = np.maximum(
            train[modality].std(axis=0, keepdims=True), 1e-6
        )

        result[modality] = np.clip(
            (test[modality] - mean) / std, -10, 10
        )

    return result


def main():
    print("=" * 60)
    print("AVaTER-MELD V2 — SAVED CHECKPOINT EVALUATION")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Load data and verify alignment.
    train_raw, y_train, _ = load_split("train")
    test_raw, y_test, test_ids = load_split("test")
    print("Feature IDs and labels match the manifests.")

    test_features = standardize(train_raw, test_raw)

    test_dataset = TensorDataset(
        torch.from_numpy(test_features["text"]),
        torch.from_numpy(test_features["audio"]),
        torch.from_numpy(test_features["video"]),
        torch.from_numpy(y_test),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    # Recreate the same training class weights.
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    weights = len(y_train) / (
        NUM_CLASSES * np.maximum(counts, 1)
    )
    class_weights = torch.tensor(
        weights, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=0.05
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=device, weights_only=False
    )

    model = AVaTERMELDV2(
        text_dim=768,
        audio_dim=1024,
        video_dim=7,
        hidden_dim=checkpoint["hidden_dim"],
        num_heads=checkpoint["num_heads"],
        num_classes=NUM_CLASSES,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Checkpoint dev macro-F1:",
          f'{checkpoint["dev_macro_f1"]:.6f}')
    print("Test samples:", len(y_test))

    losses = []
    predictions = []

    with torch.inference_mode():
        for text, audio, video, labels in test_loader:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            logits = model(text, audio, video)
            loss = criterion(logits, labels)

            losses.append(loss.item() * labels.size(0))
            predictions.extend(
                logits.argmax(dim=1).cpu().tolist()
            )

    y_pred = np.asarray(predictions)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred, average="macro",
        labels=list(range(NUM_CLASSES)), zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(
        y_test, y_pred, labels=list(range(NUM_CLASSES))
    )

    print("\nINDEPENDENT EVALUATION")
    print("-" * 40)
    print(f"Test loss:        {sum(losses) / len(y_test):.6f}")
    print(f"Test accuracy:    {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Test macro-F1:    {macro_f1:.6f} ({macro_f1 * 100:.2f}%)")
    print(f"Test weighted-F1: {weighted_f1:.6f} ({weighted_f1 * 100:.2f}%)")

    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=LABELS, zero_division=0, digits=4
    ))

    # Compare with the existing saved report without modifying it.
    old_path = RESULT_DIR / "test_results.json"
    if old_path.exists():
        with open(old_path, "r", encoding="utf-8") as file:
            old = json.load(file)

        print("\nCOMPARISON WITH SAVED RESULTS")
        print("-" * 40)
        for key, new_value in (
            ("test_accuracy", accuracy),
            ("test_macro_f1", macro_f1),
            ("test_weighted_f1", weighted_f1),
        ):
            old_value = old.get(key)
            if old_value is not None:
                print(
                    f"{key}: saved={old_value:.8f}, "
                    f"recomputed={new_value:.8f}, "
                    f"match={np.isclose(old_value, new_value, atol=1e-6)}"
                )

    # Print prediction count and confusion matrix; no files are written.
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))
    print("\nEvaluation complete. Existing files were not modified.")


if __name__ == "__main__":
    main()
