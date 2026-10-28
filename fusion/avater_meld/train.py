import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from fusion.avater_meld.model import AVaTERMELD

LABELS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]

CACHE_PATHS = {
    "text": "data/text_v2_feature_cache_{split}.pt",
    "audio": "data/audio_v4_feature_cache_{split}.pt",
    "video": "data/video_v3_feature_cache_{split}.pt",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(split):
    manifest_path = f"data/manifests/{split}.csv"
    manifest = pd.read_csv(manifest_path, dtype={"clip_id": str})

    expected_ids = manifest["clip_id"].tolist()
    expected_labels = manifest["label"].tolist()
    label_to_id = {name: i for i, name in enumerate(LABELS)}
    y = torch.tensor([label_to_id[x] for x in expected_labels], dtype=torch.long)

    features = {}
    for modality, template in CACHE_PATHS.items():
        path = template.format(split=split)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing feature cache: {path}")

        cache = torch.load(path, map_location="cpu", weights_only=True)
        cache_ids = list(cache["clip_ids"])
        cache_features = cache["features"].float()
        cache_labels = cache["labels"]

        if cache_ids != expected_ids:
            raise ValueError(f"{split}: {modality} clip IDs do not match manifest order")

        if cache_features.shape[0] != len(expected_ids):
            raise ValueError(f"{split}: {modality} feature count mismatch")

        if torch.is_tensor(cache_labels):
            cache_labels = cache_labels.tolist()

        if len(cache_labels) != len(expected_labels):
            raise ValueError(f"{split}: {modality} label count mismatch")

        for i, (actual, expected) in enumerate(zip(cache_labels, expected_labels)):
            if isinstance(actual, str):
                if actual != expected:
                    raise ValueError(
                        f"{split}: {modality} label mismatch at row {i}: "
                        f"{actual!r} != {expected!r}"
                    )
            elif int(actual) != label_to_id[expected]:
                raise ValueError(
                    f"{split}: {modality} numeric label mismatch at row {i}"
                )

        features[modality] = cache_features
        print(f"{split.upper()} {modality}: {tuple(cache_features.shape)} verified")

    return features, y, expected_ids, expected_labels


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.text = features["text"]
        self.audio = features["audio"]
        self.video = features["video"]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (
            self.text[index],
            self.audio[index],
            self.video[index],
            self.labels[index],
        )


def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_y, all_pred, all_probs, all_loss = [], [], [], []

    with torch.no_grad():
        for text, audio, video, labels in loader:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            logits = model(text, audio, video)
            if criterion is not None:
                all_loss.append(criterion(logits, labels).item())

            probs = torch.softmax(logits, dim=1)
            predictions = probs.argmax(dim=1)

            all_y.extend(labels.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(all_y, all_pred),
        "macro_f1": f1_score(all_y, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_y, all_pred, average="weighted", zero_division=0),
    }
    if all_loss:
        metrics["loss"] = float(np.mean(all_loss))

    return metrics, np.array(all_y), np.array(all_pred), np.array(all_probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = "results/avater_meld"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "best_model.pt")

    print("\n========== AVaTER-MELD TRAINING ==========")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")
    print("Emotion classes:", LABELS)
    print("Configuration:", vars(args))

    train_x, train_y, _, _ = load_split("train")
    dev_x, dev_y, _, _ = load_split("dev")
    test_x, test_y, test_ids, test_labels = load_split("test")

    train_loader = DataLoader(
        FeatureDataset(train_x, train_y),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    dev_loader = DataLoader(
        FeatureDataset(dev_x, dev_y),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        FeatureDataset(test_x, test_y),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    # Inverse-frequency weights, normalized to mean 1.
    counts = torch.bincount(train_y, minlength=len(LABELS)).float()
    class_weights = counts.sum() / (len(LABELS) * counts)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model = AVaTERMELD().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_y_epoch, train_pred_epoch, losses = [], [], []

        for text, audio, video, labels in train_loader:
            text = text.to(device)
            audio = audio.to(device)
            video = video.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(text, audio, video)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            train_y_epoch.extend(labels.detach().cpu().tolist())
            train_pred_epoch.extend(logits.argmax(dim=1).detach().cpu().tolist())

        train_metrics = {
            "loss": float(np.mean(losses)),
            "accuracy": accuracy_score(train_y_epoch, train_pred_epoch),
            "macro_f1": f1_score(
                train_y_epoch, train_pred_epoch,
                average="macro", zero_division=0
            ),
        }

        dev_metrics, _, _, _ = evaluate(
            model, dev_loader, device, criterion
        )
        scheduler.step(dev_metrics["macro_f1"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_weighted_f1": dev_metrics["weighted_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train loss {row['train_loss']:.4f} | "
            f"Train acc {row['train_accuracy']:.4f} | "
            f"Train macro-F1 {row['train_macro_f1']:.4f} | "
            f"Dev loss {row['dev_loss']:.4f} | "
            f"Dev acc {row['dev_accuracy']:.4f} | "
            f"Dev macro-F1 {row['dev_macro_f1']:.4f} | "
            f"Dev weighted-F1 {row['dev_weighted_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = dev_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_dev_macro_f1": best_macro_f1,
                "labels": LABELS,
                "args": vars(args),
            }, checkpoint_path)
            print("  Saved new best checkpoint:", checkpoint_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping: validation macro-F1 did not improve.")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, "training_history.csv"), index=False
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")

    # Reload the selected checkpoint before evaluating test data.
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, y_true, y_pred, probs = evaluate(model, test_loader, device)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Accuracy:   {test_metrics['accuracy']:.4f}")
    print(f"Macro-F1:   {test_metrics['macro_f1']:.4f}")
    print(f"Weighted-F1:{test_metrics['weighted_f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred, labels=list(range(7)),
        target_names=LABELS, digits=4, zero_division=0
    ))

    pd.DataFrame({
        "clip_id": test_ids,
        "actual": test_labels,
        "predicted": [LABELS[i] for i in y_pred],
        "confidence": probs.max(axis=1),
        **{f"prob_{name}": probs[:, i] for i, name in enumerate(LABELS)},
    }).to_csv(
        os.path.join(output_dir, "test_predictions.csv"), index=False
    )

    pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(range(7))),
        index=LABELS, columns=LABELS
    ).to_csv(os.path.join(output_dir, "confusion_matrix.csv"))

    summary = {
        "architecture": "AVaTER-inspired pooled-feature cross-modal attention",
        "best_epoch": best_epoch,
        "best_dev_macro_f1": best_macro_f1,
        "test_samples": len(y_true),
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "labels": LABELS,
        "checkpoint": checkpoint_path,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved results to:", os.path.abspath(output_dir))
    print("Predictions: test_predictions.csv")
    print("History: training_history.csv")
    print("Confusion matrix: confusion_matrix.csv")
    print("Summary: summary.json")


if __name__ == "__main__":
    main()
