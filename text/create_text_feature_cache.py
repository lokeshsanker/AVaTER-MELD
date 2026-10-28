import os
import sys
import torch
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

# Add project root to Python path
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from text.model import TextEmotionModel

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "checkpoints",
    "text_model"
)

CHECKPOINT_PATH = os.path.join(
    MODEL_DIR,
    "pytorch_model.bin"
)

MANIFEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "manifests"
)

CACHE_DIR = os.path.join(
    BASE_DIR,
    "data"
)

MAX_LENGTH = 64
BATCH_SIZE = 16

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 60)
print("TRAINED TEXT FEATURE CACHE CREATION")
print("=" * 60)

print(f"Device     : {DEVICE}")
print(f"Checkpoint : {CHECKPOINT_PATH}")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT_PATH}"
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_DIR
)

print("Tokenizer loaded.")


# ============================================================
# CREATE EXACT TRAINED MODEL
# ============================================================

print("\nCreating TextEmotionModel...")

model = TextEmotionModel(
    freeze_encoder=True
)


# ============================================================
# LOAD TRAINED CHECKPOINT
# ============================================================

print("\nLoading trained checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu"
)

model.load_state_dict(
    checkpoint,
    strict=True
)

model.to(DEVICE)
model.eval()

print("Trained TextEmotionModel loaded successfully.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(texts):

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = encoded[
        "input_ids"
    ].to(DEVICE)

    attention_mask = encoded[
        "attention_mask"
    ].to(DEVICE)

    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # 768-dimensional trained DistilBERT representation
    cls_features = outputs.last_hidden_state[:, 0, :]

    return cls_features.cpu()


# ============================================================
# PROCESS MANIFEST
# ============================================================

def process_manifest(manifest_name):

    manifest_path = os.path.join(
        MANIFEST_DIR,
        manifest_name
    )

    print("\n" + "=" * 60)
    print(f"Processing {manifest_name}")
    print("=" * 60)

    df = pd.read_csv(manifest_path)

    print(f"Samples: {len(df)}")

    if "transcript" not in df.columns:
        raise ValueError(
            "'transcript' column missing"
        )

    if "clip_id" not in df.columns:
        raise ValueError(
            "'clip_id' column missing"
        )

    texts = (
        df["transcript"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = (
        df["label"]
        .astype(str)
        .tolist()
    )

    clip_ids = (
        df["clip_id"]
        .astype(str)
        .tolist()
    )

    all_features = []

    # --------------------------------------------------------
    # Batch extraction
    # --------------------------------------------------------

    for start in tqdm(
        range(
            0,
            len(texts),
            BATCH_SIZE
        ),
        desc=f"Extracting {manifest_name}"
    ):

        end = min(
            start + BATCH_SIZE,
            len(texts)
        )

        batch_texts = texts[start:end]

        features = extract_features(
            batch_texts
        )

        all_features.append(
            features
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    features_tensor = torch.cat(
        all_features,
        dim=0
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    expected_samples = len(df)

    assert features_tensor.shape[0] == expected_samples

    assert features_tensor.shape[1] == 768

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    split_name = os.path.splitext(
        manifest_name
    )[0]

    output_path = os.path.join(
        CACHE_DIR,
        f"text_feature_cache_{split_name}.pt"
    )

    cache = {
        "features": features_tensor,
        "labels": labels,
        "clip_ids": clip_ids,
        "feature_dim": 768,
        "num_samples": expected_samples,
        "model": "Trained TextEmotionModel / DistilBERT",
        "max_length": MAX_LENGTH,
        "manifest": manifest_name
    }

    torch.save(
        cache,
        output_path
    )

    print("\nCache saved:")
    print(output_path)

    print(
        f"Feature shape: {features_tensor.shape}"
    )

    print(
        f"Feature dimension: {features_tensor.shape[1]}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    process_manifest("train.csv")

    process_manifest("dev.csv")

    process_manifest("test.csv")

    print("\n" + "=" * 60)
    print("TRAINED TEXT FEATURE CACHE COMPLETED")
    print("=" * 60)

    print("\nFiles created:")

    print(
        "data\\text_feature_cache_train.pt"
    )

    print(
        "data\\text_feature_cache_dev.pt"
    )

    print(
        "data\\text_feature_cache_test.pt"
    )


if __name__ == "__main__":
    main()