import os
import sys
import random
import numpy as np

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from config import (
    DEVICE,
    NUM_CLASSES,
    IDX2LABEL
)

from utils.experiment_logger import (
    log_run,
    log_epoch,
    log_class_results,
    log_configuration,
    log_confusion_matrix,
    log_class_detection
)


# =========================================================
# CONFIG
# =========================================================

TRAIN_CACHE = (
    "data/audio_v4_feature_cache_train.pt"
)

DEV_CACHE = (
    "data/audio_v4_feature_cache_dev.pt"
)

CHECKPOINT_DIR = (
    "checkpoints/audio_v4"
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "audio_v4_best.pt"
)

SEED = 42

BATCH_SIZE = 64

MAX_EPOCHS = 20

PATIENCE = 4

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

LABEL_SMOOTHING = 0.05

GRAD_CLIP = 1.0

SAMPLING_POWER = 0.5


# =========================================================
# SEED
# =========================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =========================================================
# DATASET
# =========================================================

class FeatureDataset(Dataset):

    def __init__(
        self,
        cache_path
    ):

        cache = torch.load(
            cache_path,
            map_location="cpu"
        )

        self.features = cache[
            "features"
        ].float()

        self.labels = cache[
            "labels"
        ].long()

        self.clip_ids = cache[
            "clip_ids"
        ]

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        return (
            self.features[idx],
            self.labels[idx],
            self.clip_ids[idx]
        )


# =========================================================
# MODEL
# =========================================================

class AudioV4Classifier(
    nn.Module
):

    def __init__(
        self,
        feature_dim=768,
        num_classes=7
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.LayerNorm(
                feature_dim
            ),

            nn.Linear(
                feature_dim,
                256
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                256,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                num_classes
            )

        )

    def forward(self, x):

        return self.network(x)


# =========================================================
# CLASS DISTRIBUTION
# =========================================================

def print_distribution(
    labels
):

    counts = np.bincount(
        labels.numpy(),
        minlength=NUM_CLASSES
    )

    print(
        "\nTraining class distribution:"
    )

    print("=" * 60)

    for i in range(NUM_CLASSES):

        print(
            f"{IDX2LABEL[i]:<10}: "
            f"{counts[i]}"
        )

    print("=" * 60)


# =========================================================
# SAMPLER
# =========================================================

def create_sampler(
    labels
):

    labels_np = labels.numpy()

    counts = np.bincount(
        labels_np,
        minlength=NUM_CLASSES
    )

    # -----------------------------------------------------
    # Mild balancing.
    #
    # power=0.5 prevents the rare classes from completely
    # dominating the training batches.
    # -----------------------------------------------------

    class_weights = np.zeros(
        NUM_CLASSES,
        dtype=np.float64
    )

    for i in range(NUM_CLASSES):

        class_weights[i] = (
            1.0 /
            max(counts[i], 1)
        ) ** SAMPLING_POWER

    sample_weights = np.array([
        class_weights[label]
        for label in labels_np
    ])

    sample_weights = torch.tensor(
        sample_weights,
        dtype=torch.double
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_np),
        replacement=True
    )

    print(
        "\nWeightedRandomSampler enabled"
    )

    print(
        f"Sampling power: "
        f"{SAMPLING_POWER}"
    )

    return sampler


# =========================================================
# EVALUATION
# =========================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    total_samples = 0

    y_true = []

    y_pred = []

    with torch.no_grad():

        for (
            features,
            labels,
            ids
        ) in loader:

            features = features.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            batch_size = labels.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            predictions = logits.argmax(
                dim=1
            )

            y_true.extend(
                labels.cpu()
                .numpy()
                .tolist()
            )

            y_pred.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    loss = (
        total_loss /
        total_samples
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    return (
        loss,
        accuracy,
        macro_f1,
        weighted_f1,
        y_true,
        y_pred
    )


# =========================================================
# TRAIN ONE EPOCH
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

    for (
        features,
        labels,
        ids
    ) in loader:

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

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels
        )

        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss: "
                f"{loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += (
            batch_size
        )

        predictions = logits.argmax(
            dim=1
        )

        y_true.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    loss = (
        total_loss /
        total_samples
    )

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=list(
            range(NUM_CLASSES)
        ),
        zero_division=0
    )

    return (
        loss,
        accuracy,
        macro_f1,
        weighted_f1
    )


# =========================================================
# MAIN
# =========================================================

def main():

    set_seed(SEED)

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("\n" + "=" * 70)

    print(
        "AUDIO V4 - CACHED WAV2VEC2 FEATURES"
    )

    print("=" * 70)

    print(
        f"PyTorch       : "
        f"{torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Device        : "
        f"{DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)


    # =====================================================
    # LOAD CACHES
    # =====================================================

    print(
        "\nLoading cached features..."
    )

    train_dataset = FeatureDataset(
        TRAIN_CACHE
    )

    dev_dataset = FeatureDataset(
        DEV_CACHE
    )

    print(
        f"Train features: "
        f"{train_dataset.features.shape}"
    )

    print(
        f"Dev features  : "
        f"{dev_dataset.features.shape}"
    )


    # =====================================================
    # CLASS DISTRIBUTION
    # =====================================================

    print_distribution(
        train_dataset.labels
    )


    # =====================================================
    # SAMPLER
    # =====================================================

    sampler = create_sampler(
        train_dataset.labels
    )


    # =====================================================
    # DATALOADERS
    # =====================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
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


    # =====================================================
    # MODEL
    # =====================================================

    model = AudioV4Classifier(
        feature_dim=768,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"\nModel parameters: "
        f"{total_parameters:,}"
    )


    # =====================================================
    # LOSS
    # =====================================================

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )


    # =====================================================
    # OPTIMIZER
    # =====================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    # =====================================================
    # TRAINING
    # =====================================================

    print("\nTraining configuration:")
    print("=" * 70)

    print(
        f"Batch size       : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max epochs       : "
        f"{MAX_EPOCHS}"
    )

    print(
        f"Learning rate    : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay     : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Label smoothing  : "
        f"{LABEL_SMOOTHING}"
    )

    print(
        f"Sampling power   : "
        f"{SAMPLING_POWER}"
    )

    print(
        f"Checkpoint metric: "
        f"Dev Macro-F1"
    )

    print("=" * 70)


    best_macro_f1 = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []


    print("\n" + "=" * 70)

    print(
        "STARTING AUDIO V4 TRAINING"
    )

    print("=" * 70)


    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        print(
            f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        )

        (
            train_loss,
            train_acc,
            train_macro_f1,
            train_weighted_f1
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        (
            dev_loss,
            dev_acc,
            dev_macro_f1,
            dev_weighted_f1,
            _,
            _
        ) = evaluate(
            model,
            dev_loader,
            criterion
        )

        print(
            f"Train Loss       : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy   : "
            f"{train_acc:.4f}"
        )

        print(
            f"Train Macro-F1   : "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Train Weighted-F1: "
            f"{train_weighted_f1:.4f}"
        )

        print(
            f"Dev Loss         : "
            f"{dev_loss:.4f}"
        )

        print(
            f"Dev Accuracy     : "
            f"{dev_acc:.4f}"
        )

        print(
            f"Dev Macro-F1     : "
            f"{dev_macro_f1:.4f}"
        )

        print(
            f"Dev Weighted-F1  : "
            f"{dev_weighted_f1:.4f}"
        )


        history.append({

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy": train_acc,

            "train_macro_f1":
                train_macro_f1,

            "train_weighted_f1":
                train_weighted_f1,

            "dev_loss": dev_loss,

            "dev_accuracy": dev_acc,

            "dev_macro_f1":
                dev_macro_f1,

            "dev_weighted_f1":
                dev_weighted_f1

        })


        # -------------------------------------------------
        # BEST MODEL
        # -------------------------------------------------

        if dev_macro_f1 > best_macro_f1:

            best_macro_f1 = (
                dev_macro_f1
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                model.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"✓ Best model saved "
                f"(Macro-F1 = "
                f"{dev_macro_f1:.4f})"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement. "
                f"Patience: "
                f"{patience_counter}/"
                f"{PATIENCE}"
            )

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


    # =====================================================
    # LOAD BEST
    # =====================================================

    print(
        "\nLoading best Audio V4 model..."
    )

    state = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        state
    )


    # =====================================================
    # FINAL DEV
    # =====================================================

    (
        final_loss,
        final_acc,
        final_macro_f1,
        final_weighted_f1,
        y_true,
        y_pred
    ) = evaluate(
        model,
        dev_loader,
        criterion
    )


    labels = list(
        range(NUM_CLASSES)
    )

    names = [
        IDX2LABEL[i]
        for i in labels
    ]


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    predicted_counts = {}

    detected_classes = 0

    for i in labels:

        emotion = IDX2LABEL[i]

        count = int(
            np.sum(
                y_pred == i
            )
        )

        predicted_counts[
            emotion
        ] = count

        if count > 0:

            detected_classes += 1


    # =====================================================
    # FINAL RESULTS
    # =====================================================

    print("\n" + "=" * 70)

    print(
        "AUDIO V4 FINAL DEV RESULTS"
    )

    print("=" * 70)

    print(
        f"Best Epoch : "
        f"{best_epoch}"
    )

    print(
        f"Accuracy   : "
        f"{final_acc:.4f} "
        f"({final_acc * 100:.2f}%)"
    )

    print(
        f"Macro-F1   : "
        f"{final_macro_f1:.4f} "
        f"({final_macro_f1 * 100:.2f}%)"
    )

    print(
        f"Weighted-F1: "
        f"{final_weighted_f1:.4f} "
        f"({final_weighted_f1 * 100:.2f}%)"
    )

    print(
        f"Classes detected: "
        f"{detected_classes}/7"
    )


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    print(
        "\nClass Detection:"
    )

    for emotion in names:

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


    # =====================================================
    # CLASSIFICATION REPORT
    # =====================================================

    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        digits=4,
        zero_division=0
    )

    print(
        "\nClassification Report:"
    )

    print(
        report_text
    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print(
        "Confusion Matrix:"
    )

    print(
        "             "
        + " ".join(
            f"{n[:8]:>9}"
            for n in names
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{names[i]:>10} "
            + " ".join(
                f"{v:9d}"
                for v in row
            )
        )


    # =====================================================
    # PREDICTIONS
    # =====================================================

    print(
        "\nPredicted class distribution:"
    )

    for emotion in names:

        print(
            f"{emotion:>10}: "
            f"{predicted_counts[emotion]:4d}"
        )


    # =====================================================
    # EXCEL LOGGING
    # =====================================================

    run_id = log_run(

        modality="Audio",

        model="Audio V4 - "
              "Cached Wav2Vec2 Features",

        dataset="MELD",

        train_samples=len(
            train_dataset
        ),

        validation_samples=len(
            dev_dataset
        ),

        test_samples=2610,

        best_epoch=best_epoch,

        accuracy=final_acc,

        macro_f1=final_macro_f1,

        weighted_f1=final_weighted_f1,

        checkpoint=CHECKPOINT_PATH,

        status="Complete",

        notes=(
            "Audio V4 uses frozen "
            "Wav2Vec2 mean-pooled "
            "768-D cached features "
            "with class-aware "
            "sampling and a "
            "two-layer MLP classifier. "
            "Checkpoint selected "
            "using validation Macro-F1."
        )

    )


    # =====================================================
    # EPOCH LOG
    # =====================================================

    for item in history:

    log_epoch(
        run_id=run_id,
        modality="Audio",
        model="Audio V4 - Cached Wav2Vec2 Features",
        epoch=item["epoch"],
        train_loss=item["train_loss"],
        train_accuracy=item["train_accuracy"],
        val_loss=item["dev_loss"],
        val_accuracy=item["dev_accuracy"],
        macro_f1=item["dev_macro_f1"]
    )

    # =====================================================
    # CLASS RESULTS
    # =====================================================

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0
    )

    class_results = []

    for emotion in names:

        class_results.append({

            "emotion": emotion,

            "precision":
                report_dict[
                    emotion
                ]["precision"],

            "recall":
                report_dict[
                    emotion
                ]["recall"],

            "f1":
                report_dict[
                    emotion
                ]["f1-score"],

            "support":
                report_dict[
                    emotion
                ]["support"]

        })


    log_class_results(

        run_id=run_id,

        modality="Audio",

        model="Audio V4 - "
              "Cached Wav2Vec2 Features",

        class_results=class_results

    )


    # =====================================================
    # CONFIGURATION
    # =====================================================

    configuration = {

        "version": "V4",

        "feature_extraction":
            "Frozen Wav2Vec2",

        "feature_dimension": 768,

        "pooling":
            "Mean pooling",

        "batch_size":
            BATCH_SIZE,

        "sampling":
            "WeightedRandomSampler",

        "sampling_power":
            SAMPLING_POWER,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "label_smoothing":
            LABEL_SMOOTHING,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "checkpoint_metric":
            "Dev Macro-F1",

        "classes":
            ", ".join(names),

        "device":
            str(DEVICE),

        "checkpoint":
            CHECKPOINT_PATH

    }


    log_configuration(

        run_id=run_id,

        modality="Audio",

        model="Audio V4 - "
              "Cached Wav2Vec2 Features",

        configuration=configuration

    )


    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    log_confusion_matrix(

        run_id=run_id,

        modality="Audio",

        model="Audio V4 - "
              "Cached Wav2Vec2 Features",

        matrix=cm,

        emotion_labels=names

    )


    # =====================================================
    # CLASS DETECTION
    # =====================================================

    log_class_detection(

        run_id=run_id,

        version="V4",

        modality="Audio",

        model="Audio V4 - "
              "Cached Wav2Vec2 Features",

        emotion_labels=names,

        predicted_counts=predicted_counts

    )


    print("\n" + "=" * 70)

    print(
        "AUDIO V4 EXPERIMENT COMPLETE"
    )

    print("=" * 70)

    print(
        f"Run ID       : {run_id}"
    )

    print(
        f"Best Epoch   : {best_epoch}"
    )

    print(
        f"Accuracy     : "
        f"{final_acc * 100:.2f}%"
    )

    print(
        f"Macro-F1     : "
        f"{final_macro_f1 * 100:.2f}%"
    )

    print(
        f"Weighted-F1  : "
        f"{final_weighted_f1 * 100:.2f}%"
    )

    print(
        f"Classes      : "
        f"{detected_classes}/7"
    )

    print(
        f"Checkpoint   : "
        f"{CHECKPOINT_PATH}"
    )

    print(
        "Excel Report : "
        "results\\experiment_report.xlsx"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()