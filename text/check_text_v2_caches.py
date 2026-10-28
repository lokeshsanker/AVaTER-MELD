import torch
from pathlib import Path

DATA_DIR = Path("data")

FILES = [
    "text_v2_feature_cache_train.pt",
    "text_v2_feature_cache_dev.pt",
    "text_v2_feature_cache_test.pt",
]

print("=" * 65)
print("TEXT V2 CACHE CHECK")
print("=" * 65)

for filename in FILES:

    path = DATA_DIR / filename

    print(f"\nChecking: {filename}")

    if not path.exists():
        print("ERROR: File not found")
        continue

    cache = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    features = cache["features"]
    labels = cache["labels"]
    num_samples = cache["num_samples"]

    print("Features shape :", tuple(features.shape))
    print("Labels shape   :", tuple(labels.shape))
    print("Num samples    :", num_samples)
    print("Feature dim    :", cache["feature_dim"])
    print("Model          :", cache["model"])

    if features.shape[0] != labels.shape[0]:
        print("STATUS: ERROR - feature/label count mismatch")
        continue

    if features.shape[0] != num_samples:
        print("STATUS: ERROR - sample count mismatch")
        continue

    if features.shape[1] != 768:
        print("STATUS: ERROR - feature dimension is not 768")
        continue

    print("STATUS: PASS")

print("\n" + "=" * 65)
print("CACHE CHECK COMPLETE")
print("=" * 65)