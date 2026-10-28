import os
import subprocess
import pandas as pd

# ============================================================
# MELD MANIFEST BUILDER
# For ma'am's multimodal emotion pipeline
# ============================================================

MELD_ROOT = r"C:\MELD\MELD.Raw"
OUTPUT_ROOT = r"C:\MELD\project\emotion_pipeline\emotion_pipeline\data"

WAV_ROOT = os.path.join(OUTPUT_ROOT, "wavs")
MANIFEST_ROOT = os.path.join(OUTPUT_ROOT, "manifests")

os.makedirs(WAV_ROOT, exist_ok=True)
os.makedirs(MANIFEST_ROOT, exist_ok=True)

# MELD labels used by config.py
VALID_LABELS = {
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
}

SPLITS = {
    "train": {
        "csv": os.path.join(MELD_ROOT, "train_sent_emo.csv"),
        "video_dir": os.path.join(MELD_ROOT, "train_splits"),
    },
    "dev": {
        "csv": os.path.join(MELD_ROOT, "dev_sent_emo.csv"),
        "video_dir": os.path.join(MELD_ROOT, "dev_splits_complete"),
    },
    "test": {
        "csv": os.path.join(MELD_ROOT, "test_sent_emo.csv"),
        "video_dir": os.path.join(MELD_ROOT, "output_repeated_splits_test"),
    },
}


def make_clip_id(row):
    return f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}"


def extract_wav(video_path, wav_path):
    """
    Convert MP4 audio to:
        16 kHz
        mono
        PCM WAV
    """

    if os.path.exists(wav_path):
        return True

    os.makedirs(os.path.dirname(wav_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        wav_path,
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("\n[ERROR] FFmpeg failed:")
        print(video_path)
        print(result.stderr[-1000:])
        return False

    return os.path.exists(wav_path)


def process_split(split_name, split_info):

    print("\n" + "=" * 70)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 70)

    csv_path = split_info["csv"]
    video_dir = split_info["video_dir"]

    df = pd.read_csv(csv_path)

    print(f"CSV rows: {len(df)}")
    print(f"Video directory: {video_dir}")

    records = []

    missing_video = 0
    failed_audio = 0
    invalid_label = 0
    successful = 0

    for i, row in df.iterrows():

        clip_id = make_clip_id(row)

        video_path = os.path.join(
            video_dir,
            clip_id + ".mp4"
        )

        # Ignore macOS metadata files such as ._dia...
        if clip_id.startswith("._"):
            continue

        # ----------------------------------------------------
        # Check video
        # ----------------------------------------------------

        if not os.path.isfile(video_path):
            missing_video += 1

            print(
                f"[MISSING VIDEO] {clip_id}"
            )

            continue

        # ----------------------------------------------------
        # Check label
        # ----------------------------------------------------

        label = str(row["Emotion"]).strip().lower()

        if label not in VALID_LABELS:
            invalid_label += 1

            print(
                f"[INVALID LABEL] {clip_id}: {label}"
            )

            continue

        # ----------------------------------------------------
        # WAV path
        # ----------------------------------------------------

        wav_path = os.path.join(
            WAV_ROOT,
            split_name,
            clip_id + ".wav"
        )

        # ----------------------------------------------------
        # Extract audio
        # ----------------------------------------------------

        if not extract_wav(video_path, wav_path):
            failed_audio += 1
            continue

        # ----------------------------------------------------
        # Transcript
        # ----------------------------------------------------

        transcript = str(row["Utterance"])

        # ----------------------------------------------------
        # Manifest record
        # ----------------------------------------------------

        records.append({
            "clip_id": clip_id,
            "video_path": os.path.abspath(video_path),
            "wav_path": os.path.abspath(wav_path),
            "transcript": transcript,
            "label": label,
        })

        successful += 1

        # Progress
        if successful % 100 == 0:
            print(
                f"Processed: {successful} | "
                f"CSV row: {i + 1}/{len(df)}"
            )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest = pd.DataFrame(
        records,
        columns=[
            "clip_id",
            "video_path",
            "wav_path",
            "transcript",
            "label",
        ],
    )

    manifest_path = os.path.join(
        MANIFEST_ROOT,
        f"{split_name}.csv"
    )

    manifest.to_csv(
        manifest_path,
        index=False,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print(f"{split_name.upper()} SUMMARY")
    print("-" * 70)

    print(f"Original CSV rows : {len(df)}")
    print(f"Valid samples     : {len(manifest)}")
    print(f"Missing videos    : {missing_video}")
    print(f"Audio failures    : {failed_audio}")
    print(f"Invalid labels    : {invalid_label}")

    print(f"\nManifest:")
    print(manifest_path)

    if len(manifest) > 0:
        print("\nLabel distribution:")
        print(manifest["label"].value_counts())

    return manifest


def main():

    print("=" * 70)
    print("MELD FULL MANIFEST BUILDER")
    print("=" * 70)

    print(f"MELD root   : {MELD_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"WAV root    : {WAV_ROOT}")
    print(f"Manifest    : {MANIFEST_ROOT}")

    all_manifests = {}

    for split_name, split_info in SPLITS.items():

        manifest = process_split(
            split_name,
            split_info
        )

        all_manifests[split_name] = manifest

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MELD SUMMARY")
    print("=" * 70)

    total = 0

    for split_name, manifest in all_manifests.items():

        count = len(manifest)
        total += count

        print(
            f"{split_name:5s}: {count} samples"
        )

    print("-" * 70)
    print(f"TOTAL : {total} samples")

    print("\nManifests created:")
    for split_name in SPLITS:
        print(
            f"  data\\manifests\\{split_name}.csv"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()