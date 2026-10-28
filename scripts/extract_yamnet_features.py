import os
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow_hub as hub
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "data" / "yamnet_features"
MODEL_URL = "https://tfhub.dev/google/yamnet/1"
SAMPLE_RATE = 16000

def main():
    os.environ.setdefault("TFHUB_CACHE_DIR", str(ROOT / "cache" / "yamnet_retry"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YAMNet...")
    model = hub.load(MODEL_URL)
    print("YAMNet loaded.")

    for split in ("train", "dev", "test"):
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            print(f"SKIP {split}: manifest not found: {manifest_path}")
            continue

        manifest = pd.read_csv(manifest_path)
        required = {"clip_id", "wav_path", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

        features = []
        clip_ids = []
        labels = []
        failures = []

        print(f"\nProcessing {split}: {len(manifest)} clips")
        for i, row in manifest.iterrows():
            path = Path(str(row["wav_path"]))
            if not path.is_absolute():
                path = ROOT / path

            try:
                audio, sr = sf.read(str(path), dtype="float32")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                if audio.size == 0:
                    raise ValueError("empty audio")
                if sr != SAMPLE_RATE:
                    raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")

                _, frame_embeddings, _ = model(audio)
                feature = frame_embeddings.numpy().mean(axis=0).astype(np.float32)

                if feature.shape != (1024,) or not np.isfinite(feature).all():
                    raise ValueError(f"invalid embedding: {feature.shape}")

                features.append(feature)
                clip_ids.append(str(row["clip_id"]))
                labels.append(str(row["label"]))

            except Exception as exc:
                failures.append((str(row["clip_id"]), str(exc)))

            if (i + 1) % 500 == 0 or i + 1 == len(manifest):
                print(f"  Checked {i + 1}/{len(manifest)}; successful={len(features)}; failed={len(failures)}")

        output_path = OUTPUT_DIR / f"yamnet_{split}_features.npz"
        np.savez_compressed(
            output_path,
            features=np.asarray(features, dtype=np.float32).reshape(-1, 1024),
            clip_ids=np.asarray(clip_ids, dtype=str),
            labels=np.asarray(labels, dtype=str),
        )

        print(f"Saved: {output_path}")
        print(f"Feature matrix: ({len(features)}, 1024)")
        print(f"Failures: {len(failures)}")

        if failures:
            failure_path = OUTPUT_DIR / f"yamnet_{split}_failures.csv"
            pd.DataFrame(failures, columns=["clip_id", "error"]).to_csv(failure_path, index=False)
            print(f"Failure report: {failure_path}")

    print("\nYAMNet extraction finished.")

if __name__ == "__main__":
    main()
