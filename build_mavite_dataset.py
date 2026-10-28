
import csv
import json
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "mavite_dataset"
MANIFESTS = DATA / "manifests"

CANDIDATES = MANIFESTS / "youtube_candidates.csv"
REVIEWS = MANIFESTS / "candidate_reviews.csv"
SAMPLES = MANIFESTS / "samples.csv"
METADATA = DATA / "dataset_metadata.json"

TARGET = 1002
EMOTIONS = [
    "neutral", "joy", "sadness", "anger",
    "fear", "disgust", "surprise"
]
INTENSITIES = ["low", "medium", "high", "unknown"]

CANDIDATE_FIELDS = [
    "video_id", "title", "channel", "published_at",
    "description", "url", "search_query"
]
REVIEW_FIELDS = [
    "video_id", "title", "url", "decision",
    "reason", "reviewed_at"
]
SAMPLE_FIELDS = [
    "sample_id", "source_video_id", "source_title",
    "source_url", "clip_path", "transcript", "emotion",
    "intensity", "annotator", "permission_notes",
    "split", "review_status"
]


def ensure_files():
    DATA.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)

    for path, fields in [
        (CANDIDATES, CANDIDATE_FIELDS),
        (REVIEWS, REVIEW_FIELDS),
        (SAMPLES, SAMPLE_FIELDS)
    ]:
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reviews_by_id():
    return {r["video_id"]: r for r in read_csv(REVIEWS)}


def review_candidates():
    candidates = read_csv(CANDIDATES)
    reviews = reviews_by_id()

    if not candidates:
        print("No candidates found. Run YouTube search first.")
        return

    pending = [
        c for c in candidates
        if c.get("video_id") not in reviews
    ]

    if not pending:
        print("All current candidates have been reviewed.")
        return

    print(f"\nUnreviewed candidates: {len(pending)}")
    print("Open each video in your browser and inspect its content.")
    print("A title alone is not sufficient to assign an emotion.")

    for candidate in pending:
        print("\n" + "-" * 65)
        print("Title:", candidate.get("title", ""))
        print("Channel:", candidate.get("channel", ""))
        print("URL:", candidate.get("url", ""))

        print("\n1. Approve as a candidate for clip review")
        print("2. Reject")
        print("3. Skip and continue")
        print("4. Stop reviewing")

        choice = input("Choice: ").strip()

        if choice == "4":
            break
        if choice == "3":
            continue
        if choice not in {"1", "2"}:
            print("Invalid choice; candidate left pending.")
            continue

        decision = "approved" if choice == "1" else "rejected"
        reason = input("Reason/notes (optional): ").strip()

        reviews = read_csv(REVIEWS)
        reviews.append({
            "video_id": candidate.get("video_id", ""),
            "title": candidate.get("title", ""),
            "url": candidate.get("url", ""),
            "decision": decision,
            "reason": reason,
            "reviewed_at": datetime.now().isoformat(timespec="seconds")
        })
        write_csv(REVIEWS, REVIEW_FIELDS, reviews)
        print(f"Saved: {decision}")


def register_clip():
    reviews = reviews_by_id()
    candidates = read_csv(CANDIDATES)

    approved = [
        c for c in candidates
        if reviews.get(c.get("video_id", ""), {}).get("decision")
        == "approved"
    ]

    if not approved:
        print("No approved candidates. Review candidates first.")
        return

    samples = read_csv(SAMPLES)
    used_ids = {s["source_video_id"] for s in samples}

    available = [c for c in approved if c["video_id"] not in used_ids]

    if not available:
        print("All approved candidates have already been registered.")
        return

    for index, candidate in enumerate(available, 1):
        print(f"{index}. {candidate['title']} [{candidate['video_id']}]")

    raw = input("Choose candidate number (or Enter to cancel): ").strip()
    if not raw:
        return
    if not raw.isdigit() or not 1 <= int(raw) <= len(available):
        print("Invalid selection.")
        return

    candidate = available[int(raw) - 1]
    path = Path(input("Local permitted clip file path: ").strip().strip('"'))

    if not path.is_file():
        print("Clip file does not exist.")
        return

    permission = input(
        "Permission/licence/consent notes for using this clip: "
    ).strip()

    if not permission:
        print("Please record the applicable permission or licence basis.")
        return

    sample_id = f"SAMPLE_{len(samples) + 1:04d}"
    samples.append({
        "sample_id": sample_id,
        "source_video_id": candidate["video_id"],
        "source_title": candidate["title"],
        "source_url": candidate["url"],
        "clip_path": str(path.resolve()),
        "transcript": "",
        "emotion": "",
        "intensity": "unknown",
        "annotator": "",
        "permission_notes": permission,
        "split": "",
        "review_status": "pending_annotation"
    })

    write_csv(SAMPLES, SAMPLE_FIELDS, samples)
    print(f"Registered {sample_id}.")


def annotate_sample():
    samples = read_csv(SAMPLES)
    pending = [s for s in samples if not s["emotion"]]

    if not pending:
        print("No samples awaiting annotation.")
        return

    sample = pending[0]
    clip = Path(sample["clip_path"])

    print("\nSample:", sample["sample_id"])
    print("Title:", sample["source_title"])
    print("Clip:", clip)
    print("Source:", sample["source_url"])

    if not clip.is_file():
        print("Warning: clip file is currently missing.")

    for i, emotion in enumerate(EMOTIONS, 1):
        print(f"{i}. {emotion}")

    choice = input("Observed dominant emotion (1-7, Enter to skip): ").strip()
    if not choice:
        return
    if not choice.isdigit() or not 1 <= int(choice) <= len(EMOTIONS):
        print("Invalid choice.")
        return

    print("Intensity:", ", ".join(INTENSITIES))
    intensity = input("Intensity: ").strip().lower()
    if intensity not in INTENSITIES:
        print("Invalid intensity.")
        return

    sample["emotion"] = EMOTIONS[int(choice) - 1]
    sample["intensity"] = intensity
    sample["transcript"] = input("Transcript (optional): ").strip()
    sample["annotator"] = input("Annotator ID: ").strip()
    sample["review_status"] = "annotated"

    write_csv(SAMPLES, SAMPLE_FIELDS, samples)
    print("Annotation saved.")


def assign_splits():
    samples = read_csv(SAMPLES)
    print("Assign splits by source video/speaker to reduce data leakage.")

    for sample in samples:
        if sample["split"]:
            continue
        if not sample["emotion"]:
            print(f"Skipping unannotated {sample['sample_id']}.")
            continue

        value = input(
            f"{sample['sample_id']} train/dev/test (Enter to skip): "
        ).strip().lower()

        if value in {"train", "dev", "test"}:
            sample["split"] = value

    write_csv(SAMPLES, SAMPLE_FIELDS, samples)
    print("Split assignments saved.")


def export_annotated():
    samples = read_csv(SAMPLES)
    output = MANIFESTS / "annotated_samples.csv"

    rows = [
        s for s in samples
        if s["emotion"] in EMOTIONS
        and Path(s["clip_path"]).is_file()
    ]

    write_csv(output, SAMPLE_FIELDS, rows)
    print(f"Exported {len(rows)} annotated samples to {output}")


def status():
    candidates = read_csv(CANDIDATES)
    reviews = read_csv(REVIEWS)
    samples = read_csv(SAMPLES)

    annotated = [s for s in samples if s["emotion"] in EMOTIONS]
    approved = sum(r["decision"] == "approved" for r in reviews)
    rejected = sum(r["decision"] == "rejected" for r in reviews)

    print("\nDATASET STATUS")
    print(f"Target samples: {TARGET}")
    print(f"YouTube candidates: {len(candidates)}")
    print(f"Approved candidates: {approved}")
    print(f"Rejected candidates: {rejected}")
    print(f"Registered clips: {len(samples)}")
    print(f"Annotated clips: {len(annotated)}")
    print(f"Remaining to target: {max(0, TARGET - len(annotated))}")

    print("\nEmotion counts:")
    for emotion in EMOTIONS:
        print(f"{emotion}: {sum(s['emotion'] == emotion for s in annotated)}")

    print("\nSplit counts:")
    for split in ("train", "dev", "test"):
        print(f"{split}: {sum(s['split'] == split for s in annotated)}")

    print("Candidates:", CANDIDATES)
    print("Reviews:", REVIEWS)
    print("Samples:", SAMPLES)


def save_metadata():
    metadata = {
        "dataset_name": "Independent AVaTER-inspired dataset",
        "target_samples": TARGET,
        "emotion_classes": EMOTIONS,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_file": str(CANDIDATES),
        "review_file": str(REVIEWS),
        "sample_manifest": str(SAMPLES),
        "note": (
            "Independent research dataset, not the original AVaTER dataset. "
            "Confirm rights and permissions. Emotion labels require human review."
        )
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Metadata saved:", METADATA)


def main():
    ensure_files()

    while True:
        print("\n" + "=" * 45)
        print("AVaTER-INSPIRED DATASET BUILDER")
        print("=" * 45)
        print("1. Review YouTube candidates")
        print("2. Register approved local clip")
        print("3. Annotate next clip")
        print("4. Assign train/dev/test splits")
        print("5. Show dataset status")
        print("6. Export annotated samples")
        print("7. Save dataset metadata")
        print("8. Exit")

        choice = input("Select option: ").strip()

        if choice == "1":
            review_candidates()
        elif choice == "2":
            register_clip()
        elif choice == "3":
            annotate_sample()
        elif choice == "4":
            assign_splits()
        elif choice == "5":
            status()
        elif choice == "6":
            export_annotated()
        elif choice == "7":
            save_metadata()
        elif choice == "8":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
