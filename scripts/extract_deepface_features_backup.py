import os
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from deepface import DeepFace

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUTPUT_DIR = ROOT / "data" / "deepface_features"
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FRAMES_PER_VIDEO = 5

def extract_video_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError("Could not open video")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video has no readable frames")

    positions = np.linspace(0, frame_count - 1, min(FRAMES_PER_VIDEO, frame_count), dtype=int)
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
            vector = np.array([float(scores.get(name, 0.0)) for name in EMOTIONS], dtype=np.float32)
            if np.isfinite(vector).all():
                frame_features.append(vector)
    finally:
        cap.release()

    if not frame_features:
        raise ValueError("No valid emotion scores extracted")

    return np.mean(frame_features, axis=0).astype(np.float32)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        manifest_path = MANIFEST_DIR / f"{split}.csv"
        if not manifest_path.exists():
            print(f"SKIP {split}: manifest not found: {manifest_path}")
            continue

        manifest = pd.read_csv(manifest_path)
        required = {"clip_id", "video_path", "label"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")

        features, clip_ids, labels, failures = [], [], [], []
        print(f"\nProcessing {split}: {len(manifest)} videos")

        for i, row in manifest.iterrows():
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = ROOT / video_path

            try:
                vector = extract_video_features(video_path)
                features.append(vector)
                clip_ids.append(str(row["clip_id"]))
                labels.append(str(row["label"]))
            except Exception as exc:
                failures.append((str(row["clip_id"]), str(exc)))

            if (i + 1) % 100 == 0 or i + 1 == len(manifest):
                print(f"  Checked {i + 1}/{len(manifest)}; successful={len(features)}; failed={len(failures)}")

        output_path = OUTPUT_DIR / f"deepface_{split}_features.npz"
        np.savez_compressed(
            output_path,
            features=np.asarray(features, dtype=np.float32).reshape(-1, len(EMOTIONS)),
            clip_ids=np.asarray(clip_ids, dtype=str),
            labels=np.asarray(labels, dtype=str),
            emotion_names=np.asarray(EMOTIONS, dtype=str)
        )

        print("Saved:", output_path)
        print("Feature matrix:", (len(features), len(EMOTIONS)))
        print("Failures:", len(failures))

        if failures:
            failure_path = OUTPUT_DIR / f"deepface_{split}_failures.csv"
            pd.DataFrame(failures, columns=["clip_id", "error"]).to_csv(failure_path, index=False)
            print("Failure report:", failure_path)

    print("\nDeepFace feature extraction finished.")

if __name__ == "__main__":
    main()
