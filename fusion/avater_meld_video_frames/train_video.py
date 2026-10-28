
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "mobilenetv2_features"
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "results" / "avater_meld_mobilenetv2"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "dev", "test"]

CLASS_NAMES = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]
LABEL_TO_ID = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
INPUT_DIM = 1280
NUM_CLASSES = 7

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD AND ALIGN FEATURES WITH MANIFESTS
# ============================================================

def load_split(split):
    feature_path = (
        FEATURE_DIR / f"mobilenetv2_{split}_features.npz"
    )
    manifest_path = MANIFEST_DIR / f"{split}.csv"

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_path}"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    with np.load(feature_path, allow_pickle=False) as data:
        features = np.asarray(
            data["features"], dtype=np.float32
        )
        clip_ids = [
            str(value) for value in data["clip_ids"].tolist()
        ]

    manifest = pd.read_csv(manifest_path)

    required_columns = {"clip_id", "label"}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"{manifest_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest["clip_id"] = manifest["clip_id"].astype(str)
    manifest["label"] = (
        manifest["label"].astype(str).str.strip().str.lower()
    )

    if len(features) != len(clip_ids):
        raise ValueError(
            f"{split}: features and clip IDs have different lengths"
        )

    if features.ndim != 2 or features.shape[1] != INPUT_DIM:
        raise ValueError(
            f"{split}: expected feature shape (N, {INPUT_DIM}), "
            f"got {features.shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{split}: features contain NaN or infinite values"
        )

    if len(set(clip_ids)) != len(clip_ids):
        raise ValueError(f"{split}: duplicate feature clip IDs")

    if manifest["clip_id"].duplicated().any():
        raise ValueError(f"{split}: duplicate manifest clip IDs")

    feature_ids = set(clip_ids)
    manifest_ids = set(manifest["clip_id"])

    if feature_ids != manifest_ids:
        missing_features = sorted(manifest_ids - feature_ids)[:10]
        missing_manifest = sorted(feature_ids - manifest_ids)[:10]

        raise ValueError(
            f"{split}: feature/manifest clip IDs do not match.\n"
            f"Missing features (first 10): {missing_features}\n"
            f"Missing manifest IDs (first 10): {missing_manifest}"
        )

    unknown_labels = sorted(
        set(manifest["label"]) - set(LABEL_TO_ID)
    )

    if unknown_labels:
        raise ValueError(
            f"{split}: unexpected emotion labels: {unknown_labels}"
        )

    # Build the label lookup by clip ID. Never assume the rows
    # of the feature file and CSV are in the same order.
    label_lookup = dict(
        zip(manifest["clip_id"], manifest["label"])
    )

    labels = np.array(
        [
            LABEL_TO_ID[label_lookup[clip_id]]
            for clip_id in clip_ids
        ],
        dtype=np.int64,
    )

    print(
        f"{split.upper()}: aligned {len(clip_ids)} utterances | "
        f"features={features.shape}"
    )

    return features, labels, clip_ids


# ============================================================
# STANDARDIZE USING TRAINING DATA ONLY
# ============================================================

def load_all_splits():
    raw = {
        split: load_split(split)
        for split in SPLITS
    }

    train_features, _, _ = raw["train"]

    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    processed = {}

    for split in SPLITS:
        features, labels, clip_ids = raw[split]

        features = np.clip(
            (features - mean) / std,
            -10.0,
            10.0,
        ).astype(np.float32)

        processed[split] = (
            features,
            labels,
            clip_ids,
        )

    return processed, mean, std


# ============================================================
# CLASSIFIER
# ============================================================

class VideoEmotionClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def make_loader(features, labels, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).long(),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)
            loss = criterion(logits, labels)

            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, all_labels, all_predictions


# ============================================================
# TRAINING
# ============================================================

def main():
    set_seed(SEED)

    print("=" * 68)
    print("MELD MOBILENETV2 VIDEO-ONLY CLASSIFIER")
    print("=" * 68)
    print(f"Project: {ROOT}")
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Feature directory: {FEATURE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    splits, mean, std = load_all_splits()

    X_train, y_train, _ = splits["train"]
    X_dev, y_dev, _ = splits["dev"]
    X_test, y_test, test_clip_ids = splits["test"]

    train_loader = make_loader(X_train, y_train, True)
    dev_loader = make_loader(X_dev, y_dev, False)
    test_loader = make_loader(X_test, y_test, False)

    # Moderate inverse-frequency weights for class imbalance.
    counts = np.bincount(
        y_train, minlength=NUM_CLASSES
    )
    class_weights = (
        len(y_train) /
        (NUM_CLASSES * np.maximum(counts, 1))
    ) ** 0.5

    class_weights /= class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=DEVICE,
        ),
        label_smoothing=0.05,
    )

    model = VideoEmotionClassifier().to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )

    checkpoint_path = OUTPUT_DIR / "best_model.pt"
    history_path = OUTPUT_DIR / "training_history.json"

    history = []
    best_dev_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for features, labels in train_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(features)
            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            train_loss_sum += loss.item() * len(labels)
            train_count += len(labels)

        train_loss = train_loss_sum / max(train_count, 1)

        dev_metrics, _, _ = evaluate(
            model, dev_loader, criterion
        )

        scheduler.step(dev_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "dev_loss": float(dev_metrics["loss"]),
            "dev_accuracy": float(dev_metrics["accuracy"]),
            "dev_macro_f1": float(dev_metrics["macro_f1"]),
            "dev_weighted_f1": float(dev_metrics["weighted_f1"]),
            "learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_acc={dev_metrics['accuracy']:.4f} | "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        # Select checkpoints exclusively using development macro-F1.
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": INPUT_DIM,
                    "num_classes": NUM_CLASSES,
                    "class_names": CLASS_NAMES,
                    "best_epoch": best_epoch,
                    "best_dev_macro_f1": best_dev_macro_f1,
                    "normalization_mean": mean,
                    "normalization_std": std,
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    # Reload the checkpoint selected on development data.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final test evaluation after model selection.
    test_metrics, y_true, y_pred = evaluate(
        model, test_loader, criterion
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "experiment": "MobileNetV2 frame features + MLP",
        "feature_source": "four frame embeddings averaged per utterance",
        "feature_dimension": INPUT_DIM,
        "frames_per_utterance": 4,
        "best_epoch": int(best_epoch),
        "best_dev_macro_f1": float(best_dev_macro_f1),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_samples": len(y_true),
        "classification_report": report,
    }

    with open(
        OUTPUT_DIR / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    pd.DataFrame({
        "clip_id": test_clip_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_emotion": [CLASS_NAMES[i] for i in y_true],
        "predicted_emotion": [CLASS_NAMES[i] for i in y_pred],
    }).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    print("\n" + "=" * 68)
    print("FINAL VIDEO-ONLY TEST RESULTS")
    print("=" * 68)
    print(f"Best epoch:        {best_epoch}")
    print(f"Best dev macro-F1: {best_dev_macro_f1:.4%}")
    print(f"Test accuracy:     {test_metrics['accuracy']:.4%}")
    print(f"Test macro-F1:     {test_metrics['macro_f1']:.4%}")
    print(f"Test weighted-F1:  {test_metrics['weighted_f1']:.4%}")

    print("\nPer-class test report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Results: {OUTPUT_DIR}")
    print("Existing V2/V3 experiments remain unchanged.")


if __name__ == "__main__":
    main()
