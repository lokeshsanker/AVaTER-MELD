import os
import sys

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# PROJECT PATH
# =========================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


# =========================================================
# CONFIG
# =========================================================

from config import (
    LABEL2IDX,
    DEVICE
)


# =========================================================
# DATASET
# =========================================================

from video.video_dataset import VideoEmotionDataset


# =========================================================
# EXCEL LOGGER
# =========================================================

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

SAMPLES_PER_CLASS = 100

BATCH_SIZE = 16
EPOCHS = 30

LR = 5e-4
WEIGHT_DECAY = 1e-4

PATIENCE = 6

FEATURE_CACHE = (
    "data/fer_vit_feature_cache.pt"
)

BEST_MODEL = (
    "checkpoints/fer_vit_gru_improved_best.pt"
)

MODEL_NAME = (
    "mo-thecreator/"
    "vit-Facial-Expression-Recognition"
)

VERSION = "V3"

print("Device:", DEVICE)


# =========================================================
# LOAD FER-ViT
# =========================================================

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification
)

from PIL import Image


processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

vit = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
).to(DEVICE)

vit.eval()


for p in vit.parameters():
    p.requires_grad = False


HIDDEN_SIZE = vit.config.hidden_size

print(
    "FER-ViT hidden size:",
    HIDDEN_SIZE
)


# =========================================================
# BUILD BALANCED DATASET
# =========================================================

df = pd.read_csv(
    "data/manifests/train.csv"
)

parts = []

for label in LABEL2IDX.keys():

    class_df = df[
        df["label"] == label
    ].sample(
        n=SAMPLES_PER_CLASS,
        random_state=42
    )

    parts.append(class_df)


balanced_df = pd.concat(
    parts,
    ignore_index=True
)


train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.20,
    random_state=42,
    stratify=balanced_df["label"]
)

train_df = train_df.reset_index(
    drop=True
)

val_df = val_df.reset_index(
    drop=True
)


print("\nTrain distribution:")

print(
    train_df["label"].value_counts()
)


print("\nValidation distribution:")

print(
    val_df["label"].value_counts()
)


# =========================================================
# DATASET
# =========================================================

train_dataset_frames = VideoEmotionDataset(
    train_df,
    augment=False
)

val_dataset_frames = VideoEmotionDataset(
    val_df,
    augment=False
)


train_loader_frames = DataLoader(
    train_dataset_frames,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

val_loader_frames = DataLoader(
    val_dataset_frames,
    batch_size=4,
    shuffle=False,
    num_workers=0
)


# =========================================================
# IMAGE NORMALIZATION REVERSAL
# =========================================================

mean = torch.tensor(
    [0.485, 0.456, 0.406],
    device=DEVICE
).view(
    3, 1, 1
)

std = torch.tensor(
    [0.229, 0.224, 0.225],
    device=DEVICE
).view(
    3, 1, 1
)


# =========================================================
# FER-ViT FEATURE EXTRACTION
# =========================================================

@torch.no_grad()
def extract_all_features(loader):

    feature_list = []
    label_list = []

    total = len(
        loader.dataset
    )

    for batch_idx, (
        frames,
        labels,
        clip_ids
    ) in enumerate(loader):

        frames = frames.to(
            DEVICE
        )

        B, T, C, H, W = frames.shape

        flat_frames = frames.reshape(
            B * T,
            C,
            H,
            W
        )

        # -------------------------------------------------
        # Undo ImageNet normalization
        # -------------------------------------------------

        flat_frames = (
            flat_frames * std
            + mean
        )

        flat_frames = (
            flat_frames
            .clamp(0, 1)
            .mul(255)
            .byte()
        )

        images = []

        for img in flat_frames:

            img = img.permute(
                1,
                2,
                0
            ).cpu().numpy()

            images.append(
                Image.fromarray(img)
            )


        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(DEVICE)
            for k, v in inputs.items()
        }


        outputs = vit.vit(
            **inputs
        )


        cls_features = (
            outputs
            .last_hidden_state[:, 0]
        )


        features = cls_features.reshape(
            B,
            T,
            HIDDEN_SIZE
        )


        feature_list.append(
            features.cpu()
        )

        label_list.append(
            labels
        )


        if (
            batch_idx == 0
            or (batch_idx + 1) % 20 == 0
        ):

            print(
                f"Extracting "
                f"{batch_idx + 1}/"
                f"{len(loader)} "
                f"("
                f"{min((batch_idx + 1) * 4, total)}/"
                f"{total}"
                f")"
            )


    features = torch.cat(
        feature_list,
        dim=0
    )

    labels = torch.cat(
        label_list,
        dim=0
    )

    return features, labels


# =========================================================
# EXTRACT FEATURES
# =========================================================

print(
    "\nExtracting FER-ViT features..."
)


train_features, train_labels = (
    extract_all_features(
        train_loader_frames
    )
)


val_features, val_labels = (
    extract_all_features(
        val_loader_frames
    )
)


print(
    "\nTrain feature shape:",
    train_features.shape
)

print(
    "Validation feature shape:",
    val_features.shape
)


# =========================================================
# SAVE FEATURE CACHE
# =========================================================

os.makedirs(
    os.path.dirname(FEATURE_CACHE),
    exist_ok=True
)

torch.save(
    {
        "train_features":
            train_features,

        "train_labels":
            train_labels,

        "val_features":
            val_features,

        "val_labels":
            val_labels,
    },
    FEATURE_CACHE
)

print(
    "\nFeature cache saved:",
    FEATURE_CACHE
)


# =========================================================
# FEATURE DATASETS
# =========================================================

train_dataset = TensorDataset(
    train_features,
    train_labels
)

val_dataset = TensorDataset(
    val_features,
    val_labels
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# IMPROVED FER-ViT + BiGRU MODEL
# =========================================================

class ImprovedFERVideoModel(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.norm = nn.LayerNorm(
            HIDDEN_SIZE
        )

        self.gru = nn.GRU(
            input_size=HIDDEN_SIZE,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.30
        )

        self.attention = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.Tanh(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                1
            )
        )

        self.classifier = nn.Sequential(

            nn.LayerNorm(
                256
            ),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.50
            ),

            nn.Linear(
                128,
                64
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                64,
                7
            )
        )


    def forward(self, x):

        x = self.norm(x)

        seq, _ = self.gru(x)

        scores = self.attention(
            seq
        ).squeeze(-1)

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.bmm(
            weights.unsqueeze(1),
            seq
        ).squeeze(1)

        logits = self.classifier(
            pooled
        )

        return logits


model = ImprovedFERVideoModel().to(
    DEVICE
)


print(
    "\nTrainable parameters:",
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=0.10
)


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )
)


# =========================================================
# TRAINING
# =========================================================

best_val_acc = 0.0
best_val_f1 = 0.0
best_epoch = 0

patience_counter = 0

epoch_history = []


for epoch in range(
    1,
    EPOCHS + 1
):

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    train_losses = []
    train_preds = []
    train_targets = []


    for features, labels in train_loader:

        features = features.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()


        train_losses.append(
            loss.item()
        )

        train_preds.extend(
            logits.argmax(
                dim=1
            ).detach().cpu().numpy()
        )

        train_targets.extend(
            labels.cpu().numpy()
        )


    train_loss = np.mean(
        train_losses
    )

    train_acc = accuracy_score(
        train_targets,
        train_preds
    )

    train_f1 = f1_score(
        train_targets,
        train_preds,
        average="macro",
        zero_division=0
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    val_losses = []
    val_preds = []
    val_targets = []


    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            val_losses.append(
                loss.item()
            )

            val_preds.extend(
                logits.argmax(
                    dim=1
                ).cpu().numpy()
            )

            val_targets.extend(
                labels.cpu().numpy()
            )


    val_loss = np.mean(
        val_losses
    )

    val_acc = accuracy_score(
        val_targets,
        val_preds
    )

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro",
        zero_division=0
    )


    scheduler.step(
        val_acc
    )


    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # -----------------------------------------------------
    # STORE EPOCH HISTORY
    # -----------------------------------------------------

    epoch_history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "macro_f1": val_f1
        }
    )


    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    print(
        f"\nEpoch {epoch:02d} | "
        f"LR={current_lr:.6f}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Acc: {train_acc:.4f} | "
        f"Train F1: {train_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )


    # -----------------------------------------------------
    # SAVE BEST
    # -----------------------------------------------------

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_epoch = epoch

        patience_counter = 0


        os.makedirs(
            os.path.dirname(BEST_MODEL),
            exist_ok=True
        )


        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "val_accuracy":
                    best_val_acc,

                "val_f1":
                    best_val_f1,

                "epoch":
                    best_epoch
            },
            BEST_MODEL
        )


        print(
            ">>> NEW BEST MODEL SAVED"
        )


    else:

        patience_counter += 1

        print(
            f"No improvement "
            f"({patience_counter}/"
            f"{PATIENCE})"
        )


    if patience_counter >= PATIENCE:

        print(
            "\nEarly stopping."
        )

        break


# =========================================================
# LOAD BEST MODEL
# =========================================================

print(
    "\nLoading best model for final evaluation..."
)


checkpoint = torch.load(
    BEST_MODEL,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# =========================================================
# FINAL BEST-MODEL EVALUATION
# =========================================================

final_preds = []
final_targets = []


with torch.no_grad():

    for features, labels in val_loader:

        features = features.to(
            DEVICE
        )

        logits = model(
            features
        )

        preds = logits.argmax(
            dim=1
        ).cpu().numpy()


        final_preds.extend(
            preds
        )

        final_targets.extend(
            labels.numpy()
        )


final_preds = np.array(
    final_preds
)

final_targets = np.array(
    final_targets
)


# =========================================================
# FINAL METRICS
# =========================================================

final_accuracy = accuracy_score(
    final_targets,
    final_preds
)

final_macro_f1 = f1_score(
    final_targets,
    final_preds,
    average="macro",
    zero_division=0
)

final_weighted_f1 = f1_score(
    final_targets,
    final_preds,
    average="weighted",
    zero_division=0
)


# =========================================================
# EMOTION LABELS
# =========================================================

emotion_labels = [
    label
    for label, idx
    in sorted(
        LABEL2IDX.items(),
        key=lambda x: x[1]
    )
]


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

report = classification_report(
    final_targets,
    final_preds,
    labels=list(range(7)),
    target_names=emotion_labels,
    output_dict=True,
    zero_division=0
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    final_targets,
    final_preds,
    labels=list(range(7))
)


# =========================================================
# CLASS DETECTION
# =========================================================

predicted_counts = {}


for class_index, emotion in enumerate(
    emotion_labels
):

    predicted_counts[emotion] = int(
        np.sum(
            final_preds == class_index
        )
    )


print(
    "\n" + "=" * 60
)

print(
    "CLASS DETECTION"
)

print(
    "=" * 60
)


for emotion in emotion_labels:

    count = predicted_counts[
        emotion
    ]

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


# =========================================================
# DISPLAY FINAL RESULTS
# =========================================================

print(
    "\n==================================="
)

print(
    "BEST FER-ViT + BiGRU RESULT"
)

print(
    "==================================="
)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Accuracy: "
    f"{final_accuracy:.4f} "
    f"({final_accuracy * 100:.2f}%)"
)

print(
    f"Macro-F1: "
    f"{final_macro_f1:.4f} "
    f"({final_macro_f1 * 100:.2f}%)"
)

print(
    f"Weighted-F1: "
    f"{final_weighted_f1:.4f} "
    f"({final_weighted_f1 * 100:.2f}%)"
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        final_targets,
        final_preds,
        labels=list(range(7)),
        target_names=emotion_labels,
        digits=4,
        zero_division=0
    )
)


print(
    "\nConfusion Matrix:"
)

print(cm)


print(
    "==================================="
)


# =========================================================
# EXCEL REPORTING
# =========================================================

print(
    "\nSaving video experiment to Excel..."
)


# ---------------------------------------------------------
# RUN SUMMARY
# ---------------------------------------------------------

run_id = log_run(

    modality="Video",

    model="FER-ViT + BiGRU",

    dataset="MELD Balanced 700",

    train_samples=len(train_df),

    validation_samples=len(val_df),

    test_samples=0,

    best_epoch=best_epoch,

    accuracy=final_accuracy,

    macro_f1=final_macro_f1,

    weighted_f1=final_weighted_f1,

    checkpoint=BEST_MODEL,

    status="Completed",

    notes=(
        "Balanced training subset with "
        f"{SAMPLES_PER_CLASS} samples per emotion. "
        "80/20 stratified train-validation split. "
        "Four frames per video. "
        "FER-ViT frozen; BiGRU trainable."
    )
)


# =========================================================
# EPOCH HISTORY → EXCEL
# =========================================================

for record in epoch_history:

    log_epoch(

        run_id,

        modality="Video",

        model="FER-ViT + BiGRU",

        epoch=record["epoch"],

        train_loss=record["train_loss"],

        train_accuracy=record[
            "train_accuracy"
        ],

        val_loss=record[
            "val_loss"
        ],

        val_accuracy=record[
            "val_accuracy"
        ],

        macro_f1=record[
            "macro_f1"
        ]

    )


# =========================================================
# CLASS RESULTS → EXCEL
# =========================================================

class_results = []


for emotion in emotion_labels:

    class_results.append(
        {
            "emotion":
                emotion,

            "precision":
                report[emotion][
                    "precision"
                ],

            "recall":
                report[emotion][
                    "recall"
                ],

            "f1":
                report[emotion][
                    "f1-score"
                ],

            "support":
                report[emotion][
                    "support"
                ]
        }
    )


log_class_results(

    run_id,

    modality="Video",

    model="FER-ViT + BiGRU",

    class_results=class_results
)


# =========================================================
# CONFIGURATION → EXCEL
# =========================================================

configuration = {

    "version":
        VERSION,

    "feature_extractor":
        MODEL_NAME,

    "feature_extractor_frozen":
        True,

    "feature_hidden_size":
        HIDDEN_SIZE,

    "frames_per_clip":
        4,

    "samples_per_class":
        SAMPLES_PER_CLASS,

    "total_balanced_samples":
        len(balanced_df),

    "train_samples":
        len(train_df),

    "validation_samples":
        len(val_df),

    "train_validation_split":
        "80/20",

    "split_method":
        "Stratified",

    "random_state":
        42,

    "batch_size":
        BATCH_SIZE,

    "gru_hidden_size":
        128,

    "gru_layers":
        2,

    "gru_bidirectional":
        True,

    "gru_dropout":
        0.30,

    "attention":
        "MLP attention",

    "classifier":
        "256 -> 128 -> 64 -> 7",

    "label_smoothing":
        0.10,

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "maximum_epochs":
        EPOCHS,

    "early_stopping_patience":
        PATIENCE,

    "optimizer":
        "AdamW",

    "scheduler":
        "ReduceLROnPlateau",

    "device":
        str(DEVICE),

    "feature_cache":
        FEATURE_CACHE,

    "best_checkpoint":
        BEST_MODEL,

    "best_epoch":
        best_epoch,

    "emotion_classes":
        ", ".join(emotion_labels)

}


log_configuration(

    run_id,

    modality="Video",

    model="FER-ViT + BiGRU",

    configuration=configuration
)


# =========================================================
# CONFUSION MATRIX → EXCEL
# =========================================================

log_confusion_matrix(

    run_id,

    modality="Video",

    model="FER-ViT + BiGRU",

    matrix=cm,

    emotion_labels=emotion_labels
)


# =========================================================
# CLASS DETECTION → EXCEL
# =========================================================

log_class_detection(

    run_id=run_id,

    version=VERSION,

    modality="Video",

    model="FER-ViT + BiGRU",

    emotion_labels=emotion_labels,

    predicted_counts=predicted_counts
)


# =========================================================
# FINAL MESSAGE
# =========================================================

print(
    f"\n[Excel] Experiment logged: "
    f"{run_id}"
)

print(
    "[Excel] Video experiment successfully recorded."
)

print(
    "[Excel] File: "
    "results\\experiment_report.xlsx"
)


print(
    "\n==================================="
)

print(
    "VIDEO EXPERIMENT COMPLETE"
)

print(
    "==================================="
)