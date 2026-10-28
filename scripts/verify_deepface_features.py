
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
all_pass = True

for split in ("train", "dev", "test"):
    manifest = pd.read_csv(ROOT / "data" / "manifests" / f"{split}.csv")
    path = ROOT / "data" / "deepface_features" / f"deepface_{split}_features.npz"

    with np.load(path, allow_pickle=False) as data:
        features = data["features"]
        clip_ids = data["clip_ids"].astype(str)
        labels = data["labels"].astype(str)

    ids_match = np.array_equal(clip_ids, manifest["clip_id"].astype(str).to_numpy())
    labels_match = np.array_equal(labels, manifest["label"].astype(str).to_numpy())
    finite = features.shape == (len(manifest), 7) and np.isfinite(features).all()
    passed = ids_match and labels_match and finite
    all_pass = all_pass and passed

    print(
        f"{split}: shape={features.shape}, IDs match={ids_match}, "
        f"labels match={labels_match}, finite={finite}, PASS={passed}"
    )

print("OVERALL:", "PASS" if all_pass else "CHECK FAILED")
