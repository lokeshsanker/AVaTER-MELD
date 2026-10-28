
import json
import random
import sys
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
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fusion" / "avater_meld_v2"))

from model import AVaTERMELDV2

DATA_DIR = ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
RESULT_DIR = ROOT / "results" / "avater_meld_v3"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = [
    "neutral", "joy", "sadness", "anger",
    "fear", "disgust", "surprise"
]
NUM_CLASSES = len(LABELS)
SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 8
HIDDEN_DIM = 128
NUM_HEADS = 4

# V3 controlled regularization settings.
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.05
CLASS_WEIGHT_POWER = 0.5
GRAD_CLIP = 1.0

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split):
    manifest = pd.read_csv(MANIFEST_DIR / f"{split}.csv")
    manifest_ids = manifest["clip_id"].astype(str).to_numpy()
    manifest_labels = manifest["label"].astype(str).to_numpy()

    features = {}
    for modality, path in FEATURE_FILES[split].items():
        if not path.exists():
            raise FileNotFoundError(f"Missing feature cache: {path}")

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


def normalize_from_train(train, other):
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


def make_loader(features, labels, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features["text"].astype(np.float32)),
        torch.from_numpy(features["audio"].astype(np.float32)),
        torch.from_numpy(features["video"].astype(np.float32)),
        torch.from_numpy(labels.astype(np.int64)),
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(model, loader, criterion, device, optimizer=None,
              scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_y = []
    all_pred = []

    for text, audio, video, labels in loader:
        text = text.to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        video = video.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(text, audio, video)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), GRAD_CLIP
                    )
                    optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_y.extend(labels.detach().cpu().tolist())
        all_pred.extend(logits.argmax(1).detach().cpu().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(
            all_y, all_pred,
            labels=list(range(NUM_CLASSES)),
            average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            all_y, all_pred, average="weighted", zero_division=0
        ),
    }


def main():
    set_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("AVaTER-MELD V3 — REGULARIZED TRAINING")
    print("=" * 65)
    print("Device:", device)
    print("Learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)
    print("Class-weight power:", CLASS_WEIGHT_POWER)

    # Load all splits and verify alignment.
    train_raw, y_train, _ = load_split("train")
    dev_raw, y_dev, _ = load_split("dev")
    test_raw, y_test, test_ids = load_split("test")
    print("Feature IDs and labels match all manifests.")

    # Use train-only statistics for normalization.
    train_features = normalize_from_train(train_raw, train_raw)
    dev_features = normalize_from_train(train_raw, dev_raw)
    test_features = normalize_from_train(train_raw, test_raw)

    train_loader = make_loader(train_features, y_train, True)
    dev_loader = make_loader(dev_features, y_dev, False)
    test_loader = make_loader(test_features, y_test, False)

    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    # Softer inverse-frequency weights than V2.
    class_weights = (
        len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    ) ** CLASS_WEIGHT_POWER
    class_weights = class_weights / class_weights.mean()
    class_weights = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    model = AVaTERMELDV2(
        text_dim=768,
        audio_dim=1024,
        video_dim=7,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_classes=NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda")
    )

    checkpoint_path = RESULT_DIR / "best_model.pt"
    history_path = RESULT_DIR / "training_history.json"
    history = []
    best_dev_f1 = -1.0
    best_epoch = 0
    waiting = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()

        train_metrics = run_epoch(
            model, train_loader, criterion, device,
            optimizer, scaler
        )
        dev_metrics = run_epoch(
            model, dev_loader, criterion, device
        )
        scheduler.step(dev_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"dev_acc={dev_metrics['accuracy']:.4f} | "
            f"train_macroF1={train_metrics['macro_f1']:.4f} | "
            f"dev_macroF1={dev_metrics['macro_f1']:.4f} | "
            f"dev_loss={dev_metrics['loss']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            waiting = 0
            torch.save({
                "experiment": "AVaTER-MELD V3",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_macro_f1": best_dev_f1,
                "labels": LABELS,
                "hidden_dim": HIDDEN_DIM,
                "num_heads": NUM_HEADS,
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "class_weight_power": CLASS_WEIGHT_POWER,
            }, checkpoint_path)
            print("  Saved best V3 checkpoint.")
        else:
            waiting += 1

        history_path.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        if waiting >= PATIENCE:
            print("Early stopping: development macro-F1 did not improve.")
            break

    # Evaluate only the selected best-development checkpoint.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(
        model, test_loader, criterion, device
    )

    # Obtain predictions for the final report.
    model.eval()
    y_pred = []
    with torch.inference_mode():
        for text, audio, video, _ in test_loader:
            logits = model(
                text.to(device), audio.to(device), video.to(device)
            )
            y_pred.extend(logits.argmax(1).cpu().tolist())

    y_pred = np.asarray(y_pred)
    matrix = confusion_matrix(
        y_test, y_pred, labels=list(range(NUM_CLASSES))
    )

    results = {
        "experiment": "AVaTER-MELD V3",
        "description": (
            "V2 architecture with stronger regularization and "
            "softer inverse-frequency class weights."
        ),
        "best_epoch": int(checkpoint["epoch"]),
        "best_dev_macro_f1": float(checkpoint["dev_macro_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(
            y_test, y_pred, average="macro",
            labels=list(range(NUM_CLASSES)), zero_division=0
        )),
        "test_weighted_f1": float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0
        )),
        "labels": LABELS,
        "train_samples": int(len(y_train)),
        "dev_samples": int(len(y_dev)),
        "test_samples": int(len(y_test)),
        "classification_report": classification_report(
            y_test, y_pred, labels=list(range(NUM_CLASSES)),
            target_names=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": matrix.tolist(),
    }

    (RESULT_DIR / "test_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        RESULT_DIR / "confusion_matrix.csv"
    )
    pd.DataFrame({
        "clip_id": test_ids,
        "true_label": [LABELS[i] for i in y_test],
        "predicted_label": [LABELS[i] for i in y_pred],
    }).to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    print("\n" + "=" * 65)
    print("V3 FINAL TEST RESULTS")
    print("=" * 65)
    print("Best epoch:", checkpoint["epoch"])
    print(f"Test accuracy:    {results['test_accuracy']:.4f}")
    print(f"Test macro-F1:    {results['test_macro_f1']:.4f}")
    print(f"Test weighted-F1: {results['test_weighted_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=LABELS, zero_division=0
    ))
    print("\nV3 results saved to:", RESULT_DIR)


if __name__ == "__main__":
    main()
