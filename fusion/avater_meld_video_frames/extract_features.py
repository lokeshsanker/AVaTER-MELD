
from pathlib import Path
import json
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FRAME_DIR = ROOT / "video_frames_4"
OUTPUT_DIR = ROOT / "data" / "mobilenetv2_features"

SPLITS = ["train", "dev", "test"]

BATCH_SIZE = 32
NUM_WORKERS = 0
EXPECTED_FRAMES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET
# ============================================================

class FrameDataset(Dataset):
    def __init__(self, split):
        self.split_dir = FRAME_DIR / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Frame directory not found: {self.split_dir}"
            )

        self.clip_dirs = sorted(
            path for path in self.split_dir.rglob("*")
            if path.is_dir()
            and len(list(path.glob("frame_*.png"))) > 0
        )

        if not self.clip_dirs:
            raise RuntimeError(
                f"No clip frame folders found in {self.split_dir}"
            )

        self.transform = MobileNet_V2_Weights.DEFAULT.transforms()

        print(
            f"{split}: found {len(self.clip_dirs)} utterance folders"
        )

    def __len__(self):
        return len(self.clip_dirs)

    def __getitem__(self, index):
        clip_dir = self.clip_dirs[index]

        frame_paths = sorted(clip_dir.glob("frame_*.png"))

        if len(frame_paths) != EXPECTED_FRAMES:
            raise ValueError(
                f"Expected {EXPECTED_FRAMES} frames in {clip_dir}; "
                f"found {len(frame_paths)}"
            )

        frames = []

        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                image = image.convert("RGB")
                frames.append(self.transform(image))

        # Shape: [4, 3, 224, 224]
        frames = torch.stack(frames)

        return frames, str(clip_dir.relative_to(self.split_dir))


# ============================================================
# MODEL
# ============================================================

def build_model():
    print("Loading pretrained MobileNetV2 weights...")

    model = mobilenet_v2(
        weights=MobileNet_V2_Weights.DEFAULT
    )

    # Retain the convolutional feature extractor and global pooling.
    model.classifier = torch.nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    print("MobileNetV2 output dimension: 1280")
    print(f"Device: {DEVICE}")

    return model


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_split(model, split):
    dataset = FrameDataset(split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_features = []
    all_clip_ids = []

    start_time = time.time()
    total = len(dataset)

    with torch.inference_mode():
        for batch_index, (frames, clip_ids) in enumerate(loader):
            # [batch, 4, 3, 224, 224]
            batch_size = frames.shape[0]

            frames = frames.reshape(
                -1, 3, 224, 224
            ).to(DEVICE)

            # [batch * 4, 1280]
            frame_features = model(frames)

            # [batch, 4, 1280]
            frame_features = frame_features.reshape(
                batch_size, EXPECTED_FRAMES, 1280
            )

            # Mean pooling over the four frames:
            # [batch, 1280]
            clip_features = frame_features.mean(dim=1)

            all_features.append(
                clip_features.cpu().numpy().astype(np.float32)
            )
            all_clip_ids.extend(list(clip_ids))

            processed = min(
                (batch_index + 1) * BATCH_SIZE,
                total,
            )

            if (
                batch_index == 0
                or (batch_index + 1) % 50 == 0
                or processed == total
            ):
                elapsed = time.time() - start_time

                print(
                    f"{split}: {processed}/{total} utterances "
                    f"({100 * processed / total:.1f}%) | "
                    f"{elapsed:.1f}s elapsed"
                )

    features = np.concatenate(all_features, axis=0)

    output_path = OUTPUT_DIR / f"mobilenetv2_{split}_features.npz"

    np.savez_compressed(
        output_path,
        features=features,
        clip_ids=np.asarray(all_clip_ids, dtype=str),
    )

    metadata = {
        "split": split,
        "utterances": int(len(features)),
        "frames_per_utterance": EXPECTED_FRAMES,
        "feature_dimension": 1280,
        "pooling": "mean across four frame embeddings",
        "backbone": "MobileNetV2 pretrained on ImageNet",
        "feature_shape": list(features.shape),
        "device": str(DEVICE),
        "feature_file": output_path.name,
    }

    with open(
        OUTPUT_DIR / f"mobilenetv2_{split}_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print(f"\nSaved: {output_path}")
    print(f"Feature shape: {features.shape}")

    del features, all_features

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("MELD MOBILENETV2 FRAME FEATURE EXTRACTION")
    print("=" * 65)

    print(f"Project: {ROOT}")
    print(f"Input frames: {FRAME_DIR}")
    print(f"Output features: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(
        "Expected utterances: train=9988, dev=1108, test=2610"
    )

    if not FRAME_DIR.exists():
        raise FileNotFoundError(
            f"Exported frames not found: {FRAME_DIR}"
        )

    model = build_model()

    for split in SPLITS:
        extract_split(model, split)

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Expected feature shapes:")
    print("  Train: (9988, 1280)")
    print("  Dev:   (1108, 1280)")
    print("  Test:  (2610, 1280)")
    print("Existing V2/V3 checkpoints were not modified.")


if __name__ == "__main__":
    main()
