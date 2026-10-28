import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE, TEXT_BATCH_SIZE
from data.dataset import load_manifest
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "text_v2"
    / "pytorch_model.bin"
)

SPLITS = {
    "train": PROJECT_ROOT / "data" / "manifests" / "train.csv",
    "dev": PROJECT_ROOT / "data" / "manifests" / "dev.csv",
    "test": PROJECT_ROOT / "data" / "manifests" / "test.csv",
}


def extract_features(model, manifest_path, output_path):

    print("\n" + "=" * 65)
    print(f"Processing: {manifest_path.name}")
    print("=" * 65)

    df = load_manifest(str(manifest_path))

    print("Samples:", len(df))

    dataset = TextEmotionDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=TEXT_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    features = []
    labels = []
    clip_ids = []

    with torch.no_grad():

        for input_ids, attention_mask, batch_labels, batch_clip_ids in tqdm(
            loader
        ):

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)

            outputs = model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            # DistilBERT representation
            cls_features = outputs.last_hidden_state[:, 0]

            features.append(
                cls_features.cpu()
            )

            labels.extend(
                batch_labels.tolist()
            )

            clip_ids.extend(
                list(batch_clip_ids)
            )

    features = torch.cat(
        features,
        dim=0
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    cache = {
        "features": features,
        "labels": labels,
        "clip_ids": clip_ids,
        "feature_dim": features.shape[1],
        "num_samples": features.shape[0],
        "model": "DistilBERT-Text-V2",
        "manifest": str(manifest_path),
    }

    torch.save(
        cache,
        output_path
    )

    print("\nSaved:")
    print(output_path)

    print("Feature shape:", tuple(features.shape))
    print("Labels shape :", tuple(labels.shape))
    print("Clip IDs     :", len(clip_ids))


def main():

    print("=" * 65)
    print("TEXT V2 FEATURE EXTRACTION")
    print("=" * 65)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    model = TextEmotionModel().to(DEVICE)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    print("Text V2 checkpoint loaded.")

    for split, manifest_path in SPLITS.items():

        output_path = (
            PROJECT_ROOT
            / "data"
            / f"text_v2_feature_cache_{split}.pt"
        )

        extract_features(
            model,
            manifest_path,
            output_path
        )

    print("\n" + "=" * 65)
    print("TEXT V2 FEATURE EXTRACTION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()