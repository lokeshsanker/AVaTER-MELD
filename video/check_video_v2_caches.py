import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


CACHES = {
    "train": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_train.pt"
    ),
    "dev": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_dev.pt"
    ),
    "test": os.path.join(
        PROJECT_ROOT,
        "data",
        "video_v2_feature_cache_test.pt"
    ),
}

EXPECTED = {
    "train": (9988, 4, 1280),
    "dev": (1108, 4, 1280),
    "test": (2610, 4, 1280),
}


def check_cache(split, path):

    print("\n" + "=" * 70)
    print(f"CHECKING {split.upper()}")
    print("=" * 70)

    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return False

    data = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    required_keys = [
        "features",
        "labels",
        "clip_ids",
        "feature_dim",
        "num_frames",
        "num_samples",
    ]

    for key in required_keys:

        if key not in data:
            print(f"❌ Missing key: {key}")
            return False

    features = data["features"]
    labels = data["labels"]
    clip_ids = data["clip_ids"]

    print(f"Features: {features.shape}")
    print(f"Labels:   {labels.shape}")
    print(f"Clips:    {len(clip_ids)}")

    expected_shape = EXPECTED[split]

    if tuple(features.shape) != expected_shape:
        print(
            f"❌ Wrong feature shape. "
            f"Expected {expected_shape}"
        )
        return False

    if labels.shape[0] != expected_shape[0]:
        print("❌ Label count mismatch")
        return False

    if len(clip_ids) != expected_shape[0]:
        print("❌ Clip ID count mismatch")
        return False

    if data["feature_dim"] != 1280:
        print("❌ Wrong feature dimension")
        return False

    if data["num_frames"] != 4:
        print("❌ Wrong number of frames")
        return False

    if data["num_samples"] != expected_shape[0]:
        print("❌ Wrong sample count")
        return False

    print("✅ PASS")

    return True


def main():

    print("=" * 70)
    print("VIDEO V2 CACHE VERIFICATION")
    print("=" * 70)

    results = []

    for split, path in CACHES.items():

        result = check_cache(
            split,
            path
        )

        results.append(result)

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    if all(results):

        print("✅ ALL VIDEO V2 CACHES PASSED")

        print("\nVerified:")
        print("  Train: 9988 × 4 × 1280")
        print("  Dev:   1108 × 4 × 1280")
        print("  Test:  2610 × 4 × 1280")

        print("\nNext step:")
        print("  Train Video V2")

    else:

        print("❌ CACHE VERIFICATION FAILED")
        print("Do not start training.")


if __name__ == "__main__":
    main()