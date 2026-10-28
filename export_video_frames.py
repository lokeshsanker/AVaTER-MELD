
from pathlib import Path

import numpy as np
from PIL import Image


# Project directory: folder containing this script
ROOT = Path(__file__).resolve().parent

SOURCE_DIR = ROOT / "data" / "face_cache"
OUTPUT_DIR = ROOT / "video_frames_4"

SPLITS = ["train", "dev", "test"]


def export_split(split):
    source_split = SOURCE_DIR / split
    output_split = OUTPUT_DIR / split

    if not source_split.exists():
        print(f"WARNING: Source folder not found: {source_split}")
        return 0

    files = sorted(source_split.rglob("*.npy"))
    exported = 0

    print(f"\nProcessing {split}: {len(files)} arrays")

    for index, npy_path in enumerate(files, start=1):
        try:
            frames = np.load(npy_path, allow_pickle=False)

            if (
                frames.ndim != 4
                or frames.shape[-1] != 3
                or frames.dtype != np.uint8
            ):
                print(
                    f"SKIPPED: {npy_path.name}, "
                    f"unexpected shape/dtype: "
                    f"{frames.shape}, {frames.dtype}"
                )
                continue

            # Preserve any subfolder structure relative to the split.
            relative_parent = npy_path.parent.relative_to(source_split)
            clip_name = npy_path.stem

            clip_output_dir = (
                output_split / relative_parent / clip_name
            )
            clip_output_dir.mkdir(parents=True, exist_ok=True)

            for frame_index, frame in enumerate(frames, start=1):
                output_path = (
                    clip_output_dir
                    / f"frame_{frame_index:02d}.png"
                )

                Image.fromarray(frame, mode="RGB").save(output_path)
                exported += 1

            if index % 500 == 0 or index == len(files):
                print(f"  Processed {index}/{len(files)} arrays")

        except Exception as exc:
            print(f"ERROR: {npy_path}: {exc}")

    print(f"Exported {exported} frames for {split}")
    return exported


def main():
    print("MELD CACHED FRAME EXPORT")
    print(f"Source: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Frame cache directory not found: {SOURCE_DIR}"
        )

    total_frames = 0

    for split in SPLITS:
        total_frames += export_split(split)

    print("\n" + "=" * 50)
    print(f"TOTAL PNG FRAMES EXPORTED: {total_frames}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Original .npy cache files were not modified.")
    print("=" * 50)


if __name__ == "__main__":
    main()
