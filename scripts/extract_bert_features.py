import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "data" / "bert_features"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
MAX_LENGTH = 128

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading tokenizer and BERT-base:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    print("BERT-base loaded.")

    for split in ("train", "dev", "test"):
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            print(f"SKIP {split}: manifest not found: {manifest_path}")
            continue

        manifest = pd.read_csv(manifest_path)
        required = {"clip_id", "transcript", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

        all_features = []
        texts = manifest["transcript"].fillna("").astype(str).tolist()

        print(f"\nProcessing {split}: {len(manifest)} utterances")
        with torch.inference_mode():
            for start in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[start:start + BATCH_SIZE]
                tokens = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                )
                tokens = {key: value.to(device) for key, value in tokens.items()}
                outputs = model(**tokens)
                hidden = outputs.last_hidden_state
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                all_features.append(pooled.cpu().numpy().astype(np.float32))

                done = min(start + len(batch_texts), len(texts))
                if done % 500 < BATCH_SIZE or done == len(texts):
                    print(f"  Processed {done}/{len(texts)}")

        features = np.concatenate(all_features, axis=0)
        if features.shape != (len(manifest), 768):
            raise ValueError(f"Unexpected feature shape: {features.shape}")

        output_path = OUTPUT_DIR / f"bert_{split}_features.npz"
        np.savez_compressed(
            output_path,
            features=features,
            clip_ids=manifest["clip_id"].astype(str).to_numpy(),
            labels=manifest["label"].astype(str).to_numpy()
        )
        print("Saved:", output_path)
        print("Feature matrix:", features.shape)

    print("\nBERT-base feature extraction finished.")

if __name__ == "__main__":
    main()
