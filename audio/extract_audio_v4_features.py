import os
import sys
import numpy as np
import pandas as pd
import torch
import librosa
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from config import (
    AUDIO_SAMPLE_RATE,
    WAV2VEC2_MODEL_NAME,
    LABEL2IDX,
)

# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_MANIFEST = "data/manifests/train.csv"
DEV_MANIFEST = "data/manifests/dev.csv"
TEST_MANIFEST = "data/manifests/test.csv"

TRAIN_CACHE = "data/audio_v4_feature_cache_train.pt"
DEV_CACHE = "data/audio_v4_feature_cache_dev.pt"
TEST_CACHE = "data/audio_v4_feature_cache_test.pt"

MAX_SECONDS = 3.5
BATCH_SIZE = 16
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================

class AudioFeatureDataset(Dataset):

    def __init__(self, manifest_path):

        self.df = pd.read_csv(manifest_path).reset_index(drop=True)

        self.max_samples = int(
            MAX_SECONDS * AUDIO_SAMPLE_RATE
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        wav_path = row["wav_path"]

        # Load audio
        wav, _ = librosa.load(
            wav_path,
            sr=AUDIO_SAMPLE_RATE,
            mono=True
        )

        # Limit duration
        if len(wav) > self.max_samples:

            wav = wav[:self.max_samples]

        # Pad shorter audio
        elif len(wav) < self.max_samples:

            wav = np.pad(
                wav,
                (0, self.max_samples - len(wav))
            )

        label = LABEL2IDX[row["label"]]

        return {
            "audio": torch.tensor(
                wav,
                dtype=torch.float32
            ),
            "label": label,
            "clip_id": row["clip_id"]
        }


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    audio = torch.stack(
        [item["audio"] for item in batch]
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long
    )

    clip_ids = [
        item["clip_id"]
        for item in batch
    ]

    return audio, labels, clip_ids


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(
    manifest_path,
    output_path,
    model,
    feature_extractor
):

    print()
    print("=" * 70)
    print(f"Processing: {manifest_path}")
    print("=" * 70)

    dataset = AudioFeatureDataset(
        manifest_path
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )

    all_features = []
    all_labels = []
    all_clip_ids = []

    for audio, labels, clip_ids in tqdm(
        loader,
        desc=os.path.basename(manifest_path)
    ):

        audio = audio.to(
            DEVICE,
            non_blocking=True
        )

        # Wav2Vec2 feature extractor
        inputs = feature_extractor(
            audio.cpu().numpy(),
            sampling_rate=AUDIO_SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs[
            "input_values"
        ].to(DEVICE)

        attention_mask = None

        if "attention_mask" in inputs:

            attention_mask = inputs[
                "attention_mask"
            ].to(DEVICE)

        # Wav2Vec2
        outputs = model(
            input_values=input_values,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        # ====================================================
        # MEAN POOLING
        # ====================================================

        features = hidden_states.mean(
            dim=1
        )

        # Expected:
        # [batch_size, 768]

        all_features.append(
            features.cpu()
        )

        all_labels.append(
            labels
        )

        all_clip_ids.extend(
            clip_ids
        )

    # Combine batches

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    # Save cache

    cache = {
        "features": features,
        "labels": labels,
        "clip_ids": all_clip_ids,
        "feature_dim": features.shape[1],
        "num_samples": features.shape[0],
        "model": "Wav2Vec2-base-960h",
        "pooling": "mean",
        "sample_rate": AUDIO_SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "manifest": manifest_path
    }

    torch.save(
        cache,
        output_path
    )

    print()
    print("Saved:")
    print(output_path)
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    print("Feature dimension:", features.shape[1])
    print("Samples:", features.shape[0])

    return features, labels


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AUDIO V4 - WAV2VEC2 FEATURE EXTRACTION")
    print("=" * 70)

    print(
        f"PyTorch       : {torch.__version__}"
    )

    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    print(
        f"Device        : {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print("=" * 70)

    # --------------------------------------------------------
    # Load Wav2Vec2
    # --------------------------------------------------------

    print()
    print("Loading Wav2Vec2 model...")

    feature_extractor = (
        Wav2Vec2FeatureExtractor
        .from_pretrained(
            WAV2VEC2_MODEL_NAME
        )
    )

    model = (
        Wav2Vec2Model
        .from_pretrained(
            WAV2VEC2_MODEL_NAME
        )
    )

    model = model.to(DEVICE)

    model.eval()

    print("Wav2Vec2 loaded successfully.")

    # --------------------------------------------------------
    # Extract TRAIN
    # --------------------------------------------------------

    train_features, train_labels = extract_features(
        TRAIN_MANIFEST,
        TRAIN_CACHE,
        model,
        feature_extractor
    )

    # --------------------------------------------------------
    # Extract DEV
    # --------------------------------------------------------

    dev_features, dev_labels = extract_features(
        DEV_MANIFEST,
        DEV_CACHE,
        model,
        feature_extractor
    )

    # --------------------------------------------------------
    # Extract TEST
    # --------------------------------------------------------

    test_features, test_labels = extract_features(
        TEST_MANIFEST,
        TEST_CACHE,
        model,
        feature_extractor
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("AUDIO V4 FEATURE EXTRACTION COMPLETE")
    print("=" * 70)

    print(
        "Train:",
        train_features.shape
    )

    print(
        "Dev  :",
        dev_features.shape
    )

    print(
        "Test :",
        test_features.shape
    )

    print()

    print(
        "Expected:"
    )

    print(
        "Train: torch.Size([9988, 768])"
    )

    print(
        "Dev  : torch.Size([1108, 768])"
    )

    print(
        "Test : torch.Size([2610, 768])"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()