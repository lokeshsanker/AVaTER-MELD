"""
Audio V2-B
---------
Wav2Vec2 partial fine-tuning for MELD emotion recognition.

Strategy:
- Freeze Wav2Vec2 early transformer layers
- Fine-tune only the last N transformer layers
- Train attention pooling + classifier
- Normal shuffled batches
- AMP for GPU memory efficiency
- Select checkpoint using validation Macro-F1

RTX 2050 4GB configuration:
- Batch size: 4
- Last 2 transformer layers trainable
- Encoder LR: 1e-5
- Head LR: 1e-4

Usage:
    python audio/train_audio_v2b.py `
        --manifest data/manifests/train.csv `
        --val_manifest data/manifests/dev.csv
"""

import argparse
import os
import sys
import random

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from transformers import Wav2Vec2Model

from config import (
    DEVICE,
    AUDIO_SAMPLE_RATE,
    WAV2VEC2_MODEL_NAME,
    AUDIO_HIDDEN_DIM,
    NUM_CLASSES,
)

from data.dataset import load_manifest
from audio.audio_dataset import AudioEmotionDataset


# ============================================================
# CONFIGURATION
# ============================================================

EMOTIONS = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
]

NUM_CLASSES = len(EMOTIONS)

SEED = 42

BATCH_SIZE = 4

MAX_EPOCHS = 12

EARLY_STOP_PATIENCE = 3

ENCODER_LR = 1e-5

HEAD_LR = 1e-4

WEIGHT_DECAY = 1e-4

MAX_GRAD_NORM = 1.0

NUM_WORKERS = 0

LAST_N_LAYERS = 2

CHECKPOINT_DIR = "checkpoints/audio_v2"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "audio_v2b_partial_finetune.pt",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# ATTENTION POOLING
# ============================================================

class AttentionPool1D(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(
            hidden_dim,
            1
        )

    def forward(
        self,
        seq,
        attention_mask=None,
    ):

        # seq:
        # (B, T, H)

        scores = self.attn(
            seq
        ).squeeze(-1)

        if attention_mask is not None:

            scores = scores.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        weights = torch.softmax(
            scores,
            dim=1
        )

        pooled = torch.bmm(
            weights.unsqueeze(1),
            seq
        ).squeeze(1)

        return pooled


# ============================================================
# AUDIO V2-B MODEL
# ============================================================

class AudioEmotionModelV2B(nn.Module):

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        last_n_layers=LAST_N_LAYERS,
    ):

        super().__init__()

        print(
            "\nLoading Wav2Vec2:"
        )

        print(
            WAV2VEC2_MODEL_NAME
        )

        self.encoder = Wav2Vec2Model.from_pretrained(
            WAV2VEC2_MODEL_NAME
        )

        # ----------------------------------------------------
        # Freeze entire encoder first
        # ----------------------------------------------------

        for param in self.encoder.parameters():

            param.requires_grad = False

        # ----------------------------------------------------
        # Unfreeze last N transformer layers
        # ----------------------------------------------------

        encoder_layers = self.encoder.encoder.layers

        total_layers = len(
            encoder_layers
        )

        start_layer = max(
            0,
            total_layers - last_n_layers
        )

        print(
            f"\nWav2Vec2 transformer layers: "
            f"{total_layers}"
        )

        print(
            f"Unfreezing last "
            f"{last_n_layers} layers:"
        )

        for i in range(
            start_layer,
            total_layers
        ):

            print(
                f"  Layer {i}"
            )

            for param in encoder_layers[
                i
            ].parameters():

                param.requires_grad = True

        # ----------------------------------------------------
        # Also train final layer norm
        # ----------------------------------------------------

        if hasattr(
            self.encoder.encoder,
            "layer_norm"
        ):

            for param in (
                self.encoder.encoder.layer_norm.parameters()
            ):

                param.requires_grad = True

        # ----------------------------------------------------
        # Attention pooling
        # ----------------------------------------------------

        self.pool = AttentionPool1D(
            AUDIO_HIDDEN_DIM
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                AUDIO_HIDDEN_DIM,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                num_classes
            ),
        )

    def forward(
        self,
        input_values,
    ):

        outputs = self.encoder(
            input_values
        )

        hidden = outputs.last_hidden_state

        pooled = self.pool(
            hidden
        )

        logits = self.classifier(
            pooled
        )

        return logits


# ============================================================
# PARAMETER REPORT
# ============================================================

def print_parameter_report(model):

    total = 0

    trainable = 0

    encoder_trainable = 0

    head_trainable = 0

    for name, param in model.named_parameters():

        count = param.numel()

        total += count

        if param.requires_grad:

            trainable += count

            if name.startswith(
                "encoder."
            ):

                encoder_trainable += count

            else:

                head_trainable += count

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL PARAMETERS"
    )

    print(
        "=" * 70
    )

    print(
        f"Total parameters: "
        f"{total:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Trainable encoder parameters: "
        f"{encoder_trainable:,}"
    )

    print(
        f"Trainable head parameters: "
        f"{head_trainable:,}"
    )

    print(
        f"Frozen parameters: "
        f"{total - trainable:,}"
    )


# ============================================================
# EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None,
    scaler=None,
):

    is_train = optimizer is not None

    if is_train:

        model.train()

    else:

        model.eval()

    total_loss = 0.0

    all_labels = []

    all_preds = []

    for input_values, labels, _ in tqdm(
        loader,
        leave=False,
    ):

        input_values = input_values.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        if is_train:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            is_train
        ):

            # AMP
            with torch.autocast(
                device_type="cuda"
                if DEVICE.type == "cuda"
                else "cpu",
                dtype=torch.float16
                if DEVICE.type == "cuda"
                else torch.bfloat16,
                enabled=(
                    DEVICE.type == "cuda"
                ),
            ):

                logits = model(
                    input_values
                )

                loss = criterion(
                    logits,
                    labels
                )

            if is_train:

                scaler.scale(
                    loss
                ).backward()

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        preds = logits.argmax(
            dim=-1
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_preds.extend(
            preds.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    avg_loss = (
        total_loss
        / len(all_labels)
    )

    accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted",
        zero_division=0,
    )

    return (
        avg_loss,
        accuracy,
        macro_f1,
        weighted_f1,
        all_labels,
        all_preds,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True
    )

    parser.add_argument(
        "--val_manifest",
        required=True
    )

    args = parser.parse_args()

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print(
        "=" * 70
    )

    print(
        "AUDIO V2-B: PARTIAL WAV2VEC2 FINE-TUNING"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Encoder LR: {ENCODER_LR}"
    )

    print(
        f"Head LR: {HEAD_LR}"
    )

    print(
        f"Trainable Wav2Vec2 layers: "
        f"{LAST_N_LAYERS}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_df = load_manifest(
        args.manifest
    )

    val_df = load_manifest(
        args.val_manifest
    )

    print(
        f"\nTraining samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_df)}"
    )

    print(
        "\nTraining class distribution:"
    )

    for emotion in EMOTIONS:

        count = (
            train_df["label"]
            == emotion
        ).sum()

        print(
            f"  {emotion:10s}: "
            f"{count}"
        )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_dataset = AudioEmotionDataset(
        train_df
    )

    val_dataset = AudioEmotionDataset(
        val_df
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = AudioEmotionModelV2B().to(
        DEVICE
    )

    print_parameter_report(
        model
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    encoder_params = []

    head_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:

            continue

        if name.startswith(
            "encoder."
        ):

            encoder_params.append(
                param
            )

        else:

            head_params.append(
                param
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENCODER_LR,
            },
            {
                "params": head_params,
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_macro_f1 = -1.0

    patience = EARLY_STOP_PATIENCE

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            "\n" + "-" * 70
        )

        print(
            f"Epoch {epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        (
            train_loss,
            train_acc,
            train_macro,
            train_weighted,
            _,
            _,
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
        )

        (
            val_loss,
            val_acc,
            val_macro,
            val_weighted,
            val_labels,
            val_preds,
        ) = run_epoch(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_acc:.4f}"
        )

        print(
            f"Train Macro-F1: "
            f"{train_macro:.4f}"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_acc:.4f}"
        )

        print(
            f"Val Macro-F1: "
            f"{val_macro:.4f}"
        )

        print(
            f"Val Weighted-F1: "
            f"{val_weighted:.4f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_macro > best_macro_f1:

            best_macro_f1 = val_macro

            patience = (
                EARLY_STOP_PATIENCE
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_accuracy":
                        val_acc,

                    "val_macro_f1":
                        val_macro,

                    "val_weighted_f1":
                        val_weighted,

                    "emotions":
                        EMOTIONS,

                    "last_n_layers":
                        LAST_N_LAYERS,
                },
                CHECKPOINT_PATH,
            )

            print(
                "\n  >>> NEW BEST MODEL"
            )

            print(
                f"  Macro-F1: "
                f"{val_macro:.4f}"
            )

            print(
                f"  Saved: "
                f"{CHECKPOINT_PATH}"
            )

        else:

            patience -= 1

            print(
                f"\n  No improvement."
                f" Patience remaining: "
                f"{patience}"
            )

            if patience <= 0:

                print(
                    "\nEarly stopping."
                )

                break

    # --------------------------------------------------------
    # LOAD BEST
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST AUDIO V2-B MODEL"
    )

    print(
        "=" * 70
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        val_loss,
        val_acc,
        val_macro,
        val_weighted,
        val_labels,
        val_preds,
    ) = run_epoch(
        model,
        val_loader,
        criterion,
    )

    print(
        f"\nBest Epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_acc:.4f}"
    )

    print(
        f"Validation Macro-F1: "
        f"{val_macro:.4f}"
    )

    print(
        f"Validation Weighted-F1: "
        f"{val_weighted:.4f}"
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            val_labels,
            val_preds,
            labels=list(
                range(NUM_CLASSES)
            ),
            target_names=EMOTIONS,
            zero_division=0,
        )
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


if __name__ == "__main__":

    main()