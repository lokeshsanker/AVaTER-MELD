
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from torch.utils.data import DataLoader, TensorDataset

from model import AVaTERMELDV2


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULT_DIR = ROOT / "results" / "avater_meld_v2"

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 6
HIDDEN_DIM = 128
NUM_HEADS = 4
NUM_CLASSES = 7
NUM_WORKERS = 0

LABELS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise"
]

FEATURE_FILES = {
    "train": {
        "text": DATA_DIR / "bert_features" / "bert_train_features.npz",
        "audio": DATA_DIR / "yamnet_features" / "yamnet_train_features.npz",
        "video": DATA_DIR / "deepface_features" / "deepface_train_features.npz"
    },
    "dev": {
        "text": DATA_DIR / "bert_features" / "bert_dev_features.npz",
        "audio": DATA_DIR / "yamnet_features" / "yamnet_dev_features.npz",
        "video": DATA_DIR / "deepface_features" / "deepface_dev_features.npz"
    },
    "test": {
        "text": DATA_DIR / "bert_features" / "bert_test_features.npz",
        "audio": DATA_DIR / "yamnet_features" / "yamnet_test_features.npz",
        "video": DATA_DIR / "deepface_features" / "deepface_test_features.npz"
    }
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_feature_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Feature file missing: {path}")

    with np.load(path, allow_pickle=True) as data:
        required = {"features", "clip_ids", "labels"}
        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"{path.name} is missing keys: {sorted(missing)}"
            )

        features = np.asarray(data["features"], dtype=np.float32)
        clip_ids = data["clip_ids"].astype(str)
        labels = data["labels"].astype(str)

    if not np.isfinite(features).all():
        raise ValueError(f"Non-finite features found in {path}")

    return features, clip_ids, labels


def load_split(split):
    manifest_path = MANIFEST_DIR / f"{split}.csv"
    manifest = pd.read_csv(manifest_path)

    required = {"clip_id", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Invalid manifest: {manifest_path}")

    manifest_ids = manifest["clip_id"].astype(str).to_numpy()
    manifest_labels = manifest["label"].astype(str).to_numpy()

    loaded = {}

    for modality, path in FEATURE_FILES[split].items():
        features, clip_ids, labels = load_feature_file(path)

        if not np.array_equal(clip_ids, manifest_ids):
            raise ValueError(
                f"{split} {modality}: clip IDs do not match manifest order"
            )

        if not np.array_equal(labels, manifest_labels):
            raise ValueError(
                f"{split} {modality}: labels do not match manifest"
            )

        loaded[modality] = features

    for label in manifest_labels:
        if label not in LABELS:
            raise ValueError(f"Unknown emotion label: {label}")

    y = np.asarray(
        [LABELS.index(label) for label in manifest_labels],
        dtype=np.int64
    )

    return loaded, y, manifest_ids


def standardize_splits(train, dev, test):
    normalized = {}

    for modality in ("text", "audio", "video"):
        mean = train[modality].mean(axis=0, keepdims=True)
        std = train[modality].std(axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)

        normalized[modality] = {
            "train": np.clip((train[modality] - mean) / std, -10, 10),
            "dev": np.clip((dev[modality] - mean) / std, -10, 10),
            "test": np.clip((test[modality] - mean) / std, -10, 10)
        }

    return normalized


def make_loader(features, labels, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features["text"].astype(np.float32)),
        torch.from_numpy(features["audio"].astype(np.float32)),
        torch.from_numpy(features["video"].astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64))
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )


def run_epoch(model, loader, criterion, optimizer, device, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_predictions = []
    all_targets = []

    for text, audio, video, labels in loader:
        text = text.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video = video.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):
                logits = model(text, audio, video)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    return (
        total_loss / len(loader.dataset),
        accuracy_score(all_targets, all_predictions),
        f1_score(
            all_targets,
            all_predictions,
            average="macro",
            labels=list(range(NUM_CLASSES)),
            zero_division=0
        ),
        np.asarray(all_targets),
        np.asarray(all_predictions)
    )


def main():
    set_seed(SEED)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("AVaTER-MELD V2: FEATURE-LEVEL CROSS-MODAL ATTENTION")
    print("=" * 65)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB"
        )

    print("Loading and validating feature caches...")

    train_raw, y_train, train_ids = load_split("train")
    dev_raw, y_dev, dev_ids = load_split("dev")
    test_raw, y_test, test_ids = load_split("test")

    print("Feature dimensions:")
    for modality in ("text", "audio", "video"):
        print(
            f"  {modality}:",
            train_raw[modality].shape,
            dev_raw[modality].shape,
            test_raw[modality].shape
        )

    print("All split IDs and labels match their manifests.")

    normalized = standardize_splits(train_raw, dev_raw, test_raw)

    train_features = {
        m: normalized[m]["train"] for m in ("text", "audio", "video")
    }
    dev_features = {
        m: normalized[m]["dev"] for m in ("text", "audio", "video")
    }
    test_features = {
        m: normalized[m]["test"] for m in ("text", "audio", "video")
    }

    train_loader = make_loader(train_features, y_train, True)
    dev_loader = make_loader(dev_features, y_dev, False)
    test_loader = make_loader(test_features, y_test, False)

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = len(y_train) / (
        NUM_CLASSES * np.maximum(counts, 1)
    )
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    model = AVaTERMELDV2(
        text_dim=768,
        audio_dim=1024,
        video_dim=7,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    checkpoint_path = RESULT_DIR / "best_model.pt"
    history_path = RESULT_DIR / "training_history.json"

    print("\nStarting training...")
    print(
        f"Batch size={BATCH_SIZE} | Max epochs={MAX_EPOCHS} | "
        f"LR={LEARNING_RATE} | Patience={PATIENCE}"
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )

        dev_loss, dev_acc, dev_f1, _, _ = run_epoch(
            model, dev_loader, criterion, None, device
        )

        scheduler.step(dev_f1)
        elapsed = time.time() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "dev_loss": dev_loss,
            "dev_accuracy": dev_acc,
            "dev_macro_f1": dev_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"train_macroF1={train_f1:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_acc={dev_acc:.4f} | "
            f"dev_macroF1={dev_f1:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"time={elapsed:.1f}s"
        )

        if dev_f1 > best_macro_f1:
            best_macro_f1 = dev_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_macro_f1": dev_f1,
                "labels": LABELS,
                "hidden_dim": HIDDEN_DIM,
                "num_heads": NUM_HEADS,
                "seed": SEED
            }, checkpoint_path)

            print("  Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1

        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation macro-F1 did not improve.")
            break

    print("\nLoading best validation checkpoint from epoch", best_epoch)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_macro_f1, y_true, y_pred = run_epoch(
        model, test_loader, criterion, None, device
    )

    test_weighted_f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(NUM_CLASSES))
    )

    results = {
        "experiment": "AVaTER-MELD V2",
        "description": (
            "AVaTER-inspired pairwise cross-modal attention over "
            "feature-group tokens; not an exact reproduction."
        ),
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_macro_f1,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
        "test_weighted_f1": test_weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "labels": LABELS,
        "train_samples": len(y_train),
        "dev_samples": len(y_dev),
        "test_samples": len(y_test)
    }

    with open(
        RESULT_DIR / "test_results.json", "w", encoding="utf-8"
    ) as file:
        json.dump(results, file, indent=2)

    pd.DataFrame(
        matrix, index=LABELS, columns=LABELS
    ).to_csv(RESULT_DIR / "confusion_matrix.csv")

    pd.DataFrame({
        "clip_id": test_ids,
        "true_label": [LABELS[i] for i in y_true],
        "predicted_label": [LABELS[i] for i in y_pred]
    }).to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)
    print("Best epoch:", best_epoch)
    print(f"Best dev macro-F1: {best_macro_f1:.4f}")
    print(f"Test loss:         {test_loss:.4f}")
    print(f"Test accuracy:     {test_acc:.4f}")
    print(f"Test macro-F1:     {test_macro_f1:.4f}")
    print(f"Test weighted-F1:  {test_weighted_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=LABELS,
        digits=4,
        zero_division=0
    ))
    print("Results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()
