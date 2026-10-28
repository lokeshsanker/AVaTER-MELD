
import os
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from deepface import DeepFace

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "data" / "deepface_features"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FRAMES_PER_VIDEO = 5
CHECKPOINT_EVERY = 100


def extract_video_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video has no readable frames")

    positions = np.linspace(
        0, frame_count - 1, min(FRAMES_PER_VIDEO, frame_count), dtype=int
    )
    frame_features = []

    try:
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if not results:
                continue

            scores = results[0].get("emotion", {})
            vector = np.array(
                [float(scores.get(name, 0.0)) for name in EMOTIONS],
                dtype=np.float32
            )

            if np.isfinite(vector).all():
                frame_features.append(vector)
    finally:
        cap.release()

    if not frame_features:
        raise ValueError("No valid emotion scores extracted")

    return np.mean(frame_features, axis=0).astype(np.float32)


def save_checkpoint(path, features, clip_ids, labels):
    temp_path = path.with_name(path.stem + "_temp.npz")
    np.savez_compressed(
        temp_path,
        features=np.asarray(features, dtype=np.float32).reshape(-1, len(EMOTIONS)),
        clip_ids=np.asarray(clip_ids, dtype=str),
        labels=np.asarray(labels, dtype=str),
        emotion_names=np.asarray(EMOTIONS, dtype=str)
    )
    os.replace(temp_path, path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("TensorFlow:", tf.__version__)
    print("Available GPUs:", tf.config.list_physical_devices("GPU"))

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print("GPU memory configuration:", exc)

    for split in ("train", "dev", "test"):
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            print(f"SKIP {split}: manifest not found")
            continue

        manifest = pd.read_csv(manifest_path)
        required = {"clip_id", "video_path", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")

        checkpoint = OUTPUT_DIR / f"deepface_{split}_checkpoint.npz"
        final_path = OUTPUT_DIR / f"deepface_{split}_features.npz"
        failure_path = OUTPUT_DIR / f"deepface_{split}_failures.csv"

        if final_path.exists():
            print(f"\nSKIP {split}: final file already exists: {final_path}")
            continue

        features, clip_ids, labels = [], [], []
        failures = []
        completed = set()

        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                features = saved["features"].tolist()
                clip_ids = saved["clip_ids"].tolist()
                labels = saved["labels"].tolist()
            completed = set(clip_ids)
            print(f"\nResuming {split}: {len(completed)} saved clips")
        else:
            print(f"\nStarting {split}: {len(manifest)} videos")

        if failure_path.exists():
            old_failures = pd.read_csv(failure_path, dtype=str)
            failures = list(old_failures[["clip_id", "error"]].itertuples(
                index=False, name=None
            ))

        processed_since_save = 0

        for _, row in manifest.iterrows():
            clip_id = str(row["clip_id"])
            if clip_id in completed:
                continue

            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = ROOT / video_path

            try:
                vector = extract_video_features(video_path)
                features.append(vector.tolist())
                clip_ids.append(clip_id)
                labels.append(str(row["label"]))
                completed.add(clip_id)
            except Exception as exc:
                failures.append((clip_id, str(exc)))

            processed_since_save += 1

            if processed_since_save >= CHECKPOINT_EVERY:
                save_checkpoint(checkpoint, features, clip_ids, labels)
                pd.DataFrame(
                    failures, columns=["clip_id", "error"]
                ).to_csv(failure_path, index=False)
                print(
                    f"  {split}: processed {len(completed)}/{len(manifest)} "
                    f"| failures={len(failures)} | checkpoint saved"
                )
                processed_since_save = 0

        save_checkpoint(checkpoint, features, clip_ids, labels)
        pd.DataFrame(
            failures, columns=["clip_id", "error"]
        ).to_csv(failure_path, index=False)

        if len(completed) != len(manifest):
            print(
                f"WARNING: {split} incomplete: {len(completed)}/"
                f"{len(manifest)} clips succeeded. Check the failure report."
            )
            continue

        os.replace(checkpoint, final_path)
        print("Saved:", final_path)
        print("Feature shape:", (len(features), len(EMOTIONS)))
        print("Failures:", len(failures))

    print("\nDeepFace extraction finished.")


if __name__ == "__main__":
    main()
