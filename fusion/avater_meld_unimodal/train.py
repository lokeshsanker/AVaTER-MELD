
import json
import random
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
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIRS = {
    "text": ROOT / "data" / "bert_features",
    "audio": ROOT / "data" / "yamnet_features",
    "video": ROOT / "data" / "deepface_features",
}

OUTPUT_DIR = ROOT / "results" / "avater_meld_unimodal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 8
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
NUM_CLASSES = 7

CLASS_NAMES = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# FEATURE LOADING
# ============================================================

def load_split(modality, split):
    feature_dir = FEATURE_DIRS[modality]

    candidates = [
        feature_dir / f"{modality}_{split}_features.npz",
    ]

    # Support alternative filenames used by earlier pipelines.
    alternatives = {
        "text": [
            f"bert_{split}_features.npz",
            f"text_{split}_features.npz",
        ],
        "audio": [
            f"yamnet_{split}_features.npz",
            f"audio_{split}_features.npz",
        ],
        "video": [
            f"deepface_{split}_features.npz",
            f"video_{split}_features.npz",
        ],
    }

    candidates.extend(
        feature_dir / filename
        for filename in alternatives[modality]
    )

    feature_path = next(
        (path for path in candidates if path.exists()),
        None,
    )

    if feature_path is None:
        raise FileNotFoundError(
            f"Cannot find {modality} {split} feature cache.\n"
            f"Checked: {[str(p) for p in candidates]}"
        )

    with np.load(feature_path, allow_pickle=True) as data:
        if "features" not in data or "labels" not in data:
            raise KeyError(
                f"{feature_path} must contain 'features' and 'labels'. "
                f"Available keys: {list(data.keys())}"
            )

        features = np.asarray(data["features"], dtype=np.float32)
        labels_raw = np.asarray(data["labels"]).reshape(-1)

    if len(features) != len(labels_raw):
        raise ValueError(
            f"Feature/label count mismatch in {feature_path}: "
            f"{len(features)} vs {len(labels_raw)}"
        )

    # Convert either integer labels or emotion-name labels to IDs.
    label_to_id = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }

    if np.issubdtype(labels_raw.dtype, np.number):
        labels = labels_raw.astype(np.int64)
    else:
        normalized = [
            str(value).strip().lower()
            for value in labels_raw
        ]

        if all(value in label_to_id for value in normalized):
            labels = np.array(
                [label_to_id[value] for value in normalized],
                dtype=np.int64,
            )
        else:
            raise ValueError(
                f"Unrecognized labels in {feature_path}: "
                f"{sorted(set(normalized))}"
            )

    if not np.isin(labels, np.arange(NUM_CLASSES)).all():
        raise ValueError(
            f"Labels must be IDs 0-6 in {feature_path}. "
            f"Found: {np.unique(labels)}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"Non-finite feature values found in {feature_path}"
        )

    return features, labels


def load_modality(modality):
    splits = {}

    for split in ["train", "dev", "test"]:
        X, y = load_split(modality, split)
        splits[split] = (X, y)

        print(
            f"{modality.upper()} {split}: "
            f"features={X.shape}, labels={y.shape}"
        )

    # Standardize using training data only.
    X_train, y_train = splits["train"]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    for split in splits:
        X, y = splits[split]
        X = np.clip((X - mean) / std, -10.0, 10.0)
        splits[split] = (
            X.astype(np.float32),
            y.astype(np.int64),
        )

    # All modalities must use the same MELD label order.
    for split, (_, labels) in splits.items():
        if len(labels) == 0:
            raise ValueError(f"{modality} {split} split is empty")

    return splits, mean, std


# ============================================================
# UNIMODAL CLASSIFIER
# ============================================================

class UnimodalClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


def make_loader(X, y, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def evaluate(model, loader, criterion=None):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(features)

            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * len(labels)

            predictions = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            total_samples += len(labels)

    metrics = {
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

    if criterion is not None:
        metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics, all_labels, all_predictions


def train_modality(modality):
    print("\n" + "=" * 65)
    print(f"TRAINING {modality.upper()}-ONLY BASELINE")
    print("=" * 65)

    set_seed(SEED)

    splits, mean, std = load_modality(modality)

    X_train, y_train = splits["train"]
    X_dev, y_dev = splits["dev"]
    X_test, y_test = splits["test"]

    train_loader = make_loader(X_train, y_train, True)
    dev_loader = make_loader(X_dev, y_dev, False)
    test_loader = make_loader(X_test, y_test, False)

    # Moderate class weighting helps address MELD imbalance.
    counts = np.bincount(y_train, minlength=NUM_CLASSES)
    class_weights = (
        len(y_train) / (NUM_CLASSES * np.maximum(counts, 1))
    ) ** 0.5
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights, dtype=torch.float32, device=DEVICE
        ),
        label_smoothing=0.05,
    )

    model = UnimodalClassifier(
        input_dim=X_train.shape[1]
    ).to(DEVICE)

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

    modality_dir = OUTPUT_DIR / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = modality_dir / "best_model.pt"
    history = []
    best_dev_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_samples = 0

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

            total_train_loss += loss.item() * len(labels)
            train_samples += len(labels)

        train_loss = total_train_loss / max(train_samples, 1)

        dev_metrics, _, _ = evaluate(
            model, dev_loader, criterion
        )

        scheduler.step(dev_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_acc={dev_metrics['accuracy']:.4f} | "
            f"dev_macro_f1={dev_metrics['macro_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "modality": modality,
                    "model_state_dict": model.state_dict(),
                    "input_dim": X_train.shape[1],
                    "mean": mean,
                    "std": std,
                    "best_epoch": best_epoch,
                    "best_dev_macro_f1": best_dev_f1,
                    "class_names": CLASS_NAMES,
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    with open(
        modality_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)

    # Reload the checkpoint selected using development macro-F1.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

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

    result = {
        "modality": modality,
        "feature_dimension": int(X_train.shape[1]),
        "best_epoch": int(best_epoch),
        "best_dev_macro_f1": float(best_dev_f1),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "test_samples": len(y_true),
        "classification_report": report,
    }

    with open(
        modality_dir / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2)

    pd.DataFrame(
        confusion_matrix(
            y_true, y_pred, labels=list(range(NUM_CLASSES))
        ),
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(modality_dir / "confusion_matrix.csv")

    pd.DataFrame({
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_emotion": [CLASS_NAMES[i] for i in y_true],
        "predicted_emotion": [CLASS_NAMES[i] for i in y_pred],
    }).to_csv(
        modality_dir / "test_predictions.csv",
        index=False,
    )

    print(f"\n{modality.upper()} FINAL TEST RESULTS")
    print(f"Best epoch:       {best_epoch}")
    print(f"Best dev macro-F1:{best_dev_f1:.4f}")
    print(f"Test accuracy:    {test_metrics['accuracy']:.4%}")
    print(f"Test macro-F1:    {test_metrics['macro_f1']:.4%}")
    print(f"Test weighted-F1: {test_metrics['weighted_f1']:.4%}")
    print("\nPer-class test results:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return result


# ============================================================
# RUN ALL THREE BASELINES
# ============================================================

def main():
    print("AVaTER-INSPIRED MELD UNIMODAL BASELINES")
    print(f"Project root: {ROOT}")
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Existing V2/V3 experiments will not be modified.")

    results = []

    for modality in ["text", "audio", "video"]:
        results.append(train_modality(modality))

    comparison = pd.DataFrame([
        {
            "modality": result["modality"],
            "feature_dimension": result["feature_dimension"],
            "best_epoch": result["best_epoch"],
            "best_dev_macro_f1": result["best_dev_macro_f1"],
            "test_accuracy": result["test_accuracy"],
            "test_macro_f1": result["test_macro_f1"],
            "test_weighted_f1": result["test_weighted_f1"],
        }
        for result in results
    ])

    comparison.to_csv(
        OUTPUT_DIR / "unimodal_comparison.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "unimodal_comparison.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\n" + "=" * 65)
    print("UNIMODAL BASELINE COMPARISON")
    print("=" * 65)
    print(comparison.to_string(index=False))
    print(f"\nSaved comparison: {OUTPUT_DIR / 'unimodal_comparison.csv'}")


if __name__ == "__main__":
    main()
