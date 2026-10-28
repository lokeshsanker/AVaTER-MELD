
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "text": ROOT / "data" / "bert_features",
    "audio": ROOT / "data" / "yamnet_features",
    "video": ROOT / "data" / "deepface_features",
}

for split in ("train", "dev", "test"):
    print(f"\n========== {split.upper()} ==========")
    manifest = pd.read_csv(ROOT / "data" / "manifests" / f"{split}.csv")
    expected_ids = manifest["clip_id"].astype(str).to_numpy()
    expected_labels = manifest["label"].astype(str).to_numpy()

    for modality, folder in FILES.items():
        prefix = {
            "text": "bert",
            "audio": "yamnet",
            "video": "deepface",
        }[modality]

        path = folder / f"{prefix}_{split}_features.npz"
        with np.load(path, allow_pickle=True) as data:
            print(f"{modality}: keys={data.files}")
            features = data["features"]
            ids = data["clip_ids"].astype(str)
            labels = data["labels"].astype(str)

            ids_match = np.array_equal(ids, expected_ids)
            labels_match = np.array_equal(labels, expected_labels)
            finite = np.isfinite(features).all()

            print(
                f"  shape={features.shape}, "
                f"IDs_match={ids_match}, "
                f"labels_match={labels_match}, "
                f"finite={finite}"
            )
