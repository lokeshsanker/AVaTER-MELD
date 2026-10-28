
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fusion" / "avater_meld_v2"))

from model import AVaTERMELDV2

DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULT_DIR = ROOT / "results" / "avater_meld_v3"
CHECKPOINT_PATH = RESULT_DIR / "best_model.pt"
SAVED_RESULTS_PATH = RESULT_DIR / "test_results.json"

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
    manifest = pd.read_csv(MANIFEST_DIR / f"{split}.csv")
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
            raise ValueError(f"Non-finite features in {path}")
        if not np.array_equal(ids, manifest_ids):
            raise ValueError(f"{split}/{modality}: clip IDs mismatch")
        if not np.array_equal(labels, manifest_labels):
            raise ValueError(f"{split}/{modality}: label mismatch")

        features[modality] = x

    unknown = set(manifest_labels) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    y = np.asarray(
        [LABELS.index(label) for label in manifest_labels],
        dtype=np.int64
    )
    return features, y, manifest_ids


def standardize(train, other):
    result = {}

    for modality in ("text", "audio", "video"):
        mean = train[modality].mean(axis=0, keepdims=True)
        std = np.maximum(
            train[modality].std(axis=0, keepdims=True), 1e-6
        )
        result[modality] = np.clip(
            (other[modality] - mean) / std, -10, 10
        )

    return result


def main():
    print("=" * 65)
    print("AVaTER-MELD V3 — INDEPENDENT CHECKPOINT EVALUATION")
    print("=" * 65)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint missing: {CHECKPOINT_PATH}")
    if not SAVED_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Saved results missing: {SAVED_RESULTS_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_raw, y_train, _ = load_split("train")
    test_raw, y_test, test_ids = load_split("test")
    print("Feature IDs and labels match the manifests.")

    test_features = standardize(train_raw, test_raw)

    test_dataset = TensorDataset(
        torch.from_numpy(test_features["text"].astype(np.float32)),
        torch.from_numpy(test_features["audio"].astype(np.float32)),
        torch.from_numpy(test_features["video"].astype(np.float32)),
        torch.from_numpy(y_test),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    if checkpoint.get("labels") != LABELS:
        raise ValueError("Checkpoint emotion labels do not match.")

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

    # Reproduce V3's softened inverse-frequency class weights.
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weight_power = float(checkpoint["class_weight_power"])
    weights = (
        len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    ) ** class_weight_power
    weights = weights / weights.mean()

    class_weights = torch.tensor(
        weights, dtype=torch.float32, device=device
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    total_loss = 0.0
    y_pred = []

    with torch.inference_mode():
        for text, audio, video, labels in test_loader:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            logits = model(text, audio, video)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            y_pred.extend(logits.argmax(1).cpu().tolist())

    y_pred = np.asarray(y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred, labels=list(range(NUM_CLASSES)),
        average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    with open(SAVED_RESULTS_PATH, "r", encoding="utf-8") as file:
        saved = json.load(file)

    print("\nCHECKPOINT")
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Saved best dev macro-F1:",
        f'{checkpoint["dev_macro_f1"]:.6f}'
    )
    print("Test samples:", len(y_test))

    print("\nINDEPENDENT TEST METRICS")
    print(f"Test loss:        {total_loss / len(y_test):.6f}")
    print(f"Test accuracy:    {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print(f"Test macro-F1:    {macro_f1:.6f} ({macro_f1 * 100:.2f}%)")
    print(f"Test weighted-F1: {weighted_f1:.6f} ({weighted_f1 * 100:.2f}%)")

    print("\nCOMPARISON WITH SAVED RESULTS")
    for key, value in (
        ("test_accuracy", accuracy),
        ("test_macro_f1", macro_f1),
        ("test_weighted_f1", weighted_f1),
    ):
        old = saved.get(key)
        if old is not None:
            print(
                f"{key}: saved={old:.8f}, "
                f"recomputed={value:.8f}, "
                f"match={np.isclose(old, value, atol=1e-6)}"
            )

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        zero_division=0,
        digits=4,
    ))

    matrix = confusion_matrix(
        y_test, y_pred, labels=list(range(NUM_CLASSES))
    )
    print("CONFUSION MATRIX (rows=true, columns=predicted)")
    print(pd.DataFrame(matrix, index=LABELS, columns=LABELS))
    print("\nEvaluation finished. No files were modified.")


if __name__ == "__main__":
    main()
