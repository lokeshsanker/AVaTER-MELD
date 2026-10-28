
"""
AVaTER-MELD V3: Single-video emotion inference + Gemini report generation.

Text  : BERT-base-uncased, masked mean pooling (768-D)
Audio : YAMNet frame-embedding mean pooling (1024-D)
Video : DeepFace emotion scores (7-D)
Fusion: Existing AVaTER-MELD V3 checkpoint

The existing checkpoint, training caches and evaluation results are
read-only. After successful inference, a Gemini research PDF is generated.
"""

import argparse
import subprocess
import tempfile
import uuid
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
import tensorflow_hub as hub
import whisper

from deepface import DeepFace
from transformers import AutoTokenizer, AutoModel


# =========================================================
# PROJECT PATHS AND IMPORTS
# =========================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "fusion" / "avater_meld_v2"))

from model import AVaTERMELDV2
from config import WHISPER_MODEL_SIZE


# =========================================================
# MODEL CONFIGURATION
# =========================================================

RESULT_DIR = ROOT / "results" / "avater_meld_v3"
CHECKPOINT_PATH = RESULT_DIR / "best_model.pt"

TEXT_TRAIN = ROOT / "data" / "bert_features" / "bert_train_features.npz"
AUDIO_TRAIN = ROOT / "data" / "yamnet_features" / "yamnet_train_features.npz"
VIDEO_TRAIN = ROOT / "data" / "deepface_features" / "deepface_train_features.npz"

LABELS = [
    "neutral", "joy", "sadness", "anger",
    "fear", "disgust", "surprise"
]

DEEPFACE_EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]

BERT_MODEL_NAME = "bert-base-uncased"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1"
AUDIO_SAMPLE_RATE = 16000
BERT_MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# AUDIO EXTRACTION
# =========================================================

def extract_audio(video_path):
    temp = tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    )
    wav_path = temp.name
    temp.close()

    command = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        "-c:a", "pcm_s16le", wav_path
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
    except Exception as exc:
        Path(wav_path).unlink(missing_ok=True)
        raise RuntimeError(
            "Audio extraction failed. Check FFmpeg installation "
            "and confirm that the video contains an audio stream."
        ) from exc

    return wav_path


# =========================================================
# TEXT FEATURE: BERT
# =========================================================

def extract_text_feature(transcript, tokenizer, bert_model):
    tokens = tokenizer(
        transcript,
        truncation=True,
        padding=True,
        max_length=BERT_MAX_LENGTH,
        return_tensors="pt"
    )

    tokens = {key: value.to(DEVICE) for key, value in tokens.items()}

    with torch.inference_mode():
        hidden = bert_model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)

        pooled = (
            (hidden * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

    return pooled.squeeze(0).cpu().numpy().astype(np.float32)


# =========================================================
# AUDIO FEATURE: YAMNET
# =========================================================

def extract_audio_feature(wav_path, yamnet_model):
    audio, sample_rate = sf.read(wav_path, dtype="float32")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(
            f"Expected {AUDIO_SAMPLE_RATE} Hz audio, got {sample_rate} Hz."
        )

    if audio.size == 0:
        raise ValueError("Extracted audio is empty.")

    _, frame_embeddings, _ = yamnet_model(audio)
    feature = tf_mean(frame_embeddings)

    if feature.shape != (1024,):
        raise ValueError(f"Unexpected YAMNet feature shape: {feature.shape}")

    return feature.astype(np.float32)


def tf_mean(embeddings):
    return np.asarray(
        embeddings.numpy().mean(axis=0),
        dtype=np.float32
    )


# =========================================================
# VIDEO FEATURE: DEEPFACE
# =========================================================

def extract_video_feature(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            raise ValueError("Video contains no readable frames.")

        indices = np.linspace(
            0,
            frame_count - 1,
            min(5, frame_count),
            dtype=int
        )

        vectors = []

        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()

            if not ok:
                continue

            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=False,
                silent=True
            )

            if isinstance(result, list):
                if not result:
                    continue
                result = result[0]

            scores = result.get("emotion", {})

            vector = np.asarray(
                [
                    float(scores.get(name, 0.0))
                    for name in DEEPFACE_EMOTIONS
                ],
                dtype=np.float32
            )

            vectors.append(vector)

        if not vectors:
            raise ValueError("Could not extract video-frame features.")

        feature = np.mean(vectors, axis=0).astype(np.float32)

        if feature.shape != (7,):
            raise ValueError(
                f"Unexpected DeepFace feature shape: {feature.shape}"
            )

        return feature

    finally:
        cap.release()


# =========================================================
# TRAINING-CACHE NORMALIZATION STATISTICS
# =========================================================

def load_training_statistics():
    paths = {
        "text": TEXT_TRAIN,
        "audio": AUDIO_TRAIN,
        "video": VIDEO_TRAIN,
    }

    stats = {}

    for modality, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required training cache is missing: {path}"
            )

        with np.load(path, allow_pickle=True) as data:
            features = np.asarray(data["features"], dtype=np.float32)

        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite values found in {path}")

        stats[modality] = (
            features.mean(axis=0, keepdims=True),
            np.maximum(
                features.std(axis=0, keepdims=True), 1e-6
            )
        )

    return stats


def normalize_feature(feature, stats, modality):
    mean, std = stats[modality]

    normalized = np.clip(
        (feature.reshape(1, -1) - mean) / std,
        -10,
        10
    )

    return torch.from_numpy(
        normalized.astype(np.float32)
    ).to(DEVICE)


# =========================================================
# GEMINI REPORT INTEGRATION
# =========================================================

def generate_research_report_for_video(video_path, inference_result):
    """
    Generate a PDF using the actual current V3 inference result and
    the saved V3 experiment evaluation evidence.

    This function does not run inference again.
    """
    report_app_dir = ROOT / "report_app"

    if not report_app_dir.is_dir():
        print(
            "Research PDF skipped: report_app directory was not found."
        )
        return None

    if str(report_app_dir) not in sys.path:
        sys.path.insert(0, str(report_app_dir))

    try:
        import app as report_app

        print("\nLoading saved V3 evaluation evidence...")
        evaluation = report_app.load_evaluation_evidence()

        print("Sending inference and evaluation evidence to Gemini...")
        report_text = report_app.generate_llm_report(
            inference_result,
            evaluation
        )

        report_path = (
            report_app.REPORT_DIR
            / f"research_report_{uuid.uuid4().hex}.pdf"
        )

        report_app.create_pdf(
            inference_result,
            evaluation,
            report_text,
            report_path
        )

        print("\n" + "=" * 60)
        print("GEMINI RESEARCH REPORT GENERATED")
        print("=" * 60)
        print("Video:", Path(video_path).name)
        print("Prediction:", inference_result["predicted_emotion"])
        print("PDF:", report_path)

        return report_path

    except Exception as exc:
        # A report-service error should not erase a successful prediction.
        print(
            "Research PDF generation failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None


# =========================================================
# V3 INFERENCE
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run AVaTER-MELD V3 inference and generate a Gemini research PDF."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video file."
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Run V3 inference without generating a Gemini PDF."
    )

    args = parser.parse_args()
    video_path = Path(args.video).expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"V3 checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("=" * 60)
    print("AVaTER-MELD V3 INFERENCE")
    print("=" * 60)
    print("Device:", DEVICE)
    print("Video:", video_path)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint.get("labels") != LABELS:
        raise ValueError("Checkpoint label order does not match V3.")

    model = AVaTERMELDV2(
        text_dim=768,
        audio_dim=1024,
        video_dim=7,
        hidden_dim=checkpoint["hidden_dim"],
        num_heads=checkpoint["num_heads"],
        num_classes=len(LABELS)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded V3 checkpoint from epoch:", checkpoint["epoch"])

    print("\nLoading BERT, YAMNet and Whisper...")
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    bert_model.eval()

    yamnet_model = hub.load(YAMNET_URL)
    whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)

    print("Loading training-set normalization statistics...")
    stats = load_training_statistics()

    wav_path = None
    inference_result = None

    try:
        print("\nExtracting audio...")
        wav_path = extract_audio(video_path)

        print("Transcribing speech with Whisper...")
        transcript = whisper_model.transcribe(
            wav_path,
            fp16=False
        )["text"].strip()

        if not transcript:
            transcript = ""

        print(
            "Transcript:",
            transcript if transcript else "(no speech recognized)"
        )

        print("\nExtracting modality features...")

        text_feature = extract_text_feature(
            transcript, tokenizer, bert_model
        )

        audio_feature = extract_audio_feature(
            wav_path, yamnet_model
        )

        video_feature = extract_video_feature(video_path)

        text_tensor = normalize_feature(text_feature, stats, "text")
        audio_tensor = normalize_feature(audio_feature, stats, "audio")
        video_tensor = normalize_feature(video_feature, stats, "video")

        print("Text feature shape:", tuple(text_tensor.shape))
        print("Audio feature shape:", tuple(audio_tensor.shape))
        print("Video feature shape:", tuple(video_tensor.shape))

        with torch.inference_mode():
            logits = model(text_tensor, audio_tensor, video_tensor)
            probabilities = torch.softmax(
                logits, dim=1
            )[0].cpu().numpy()

        predicted_index = int(np.argmax(probabilities))

        # Preserve the actual prediction and scores for Gemini.
        inference_result = {
            "predicted_emotion": LABELS[predicted_index],
            "model_softmax_score": float(
                probabilities[predicted_index]
            ),
            "emotion_scores": {
                LABELS[i]: float(probabilities[i])
                for i in range(len(LABELS))
            },
            "transcript": transcript,
            "filename": video_path.name
        }

        print("\n" + "=" * 60)
        print("PREDICTION")
        print("=" * 60)
        print("Predicted emotion:", LABELS[predicted_index])
        print(
            "Model softmax score: "
            f"{probabilities[predicted_index]:.4f}"
        )

        print("\nEmotion scores:")
        for index in np.argsort(probabilities)[::-1]:
            print(
                f"{LABELS[index]:10s}: {probabilities[index]:.4f}"
            )

        print(
            "\nNote: these are model softmax scores, not calibrated "
            "probabilities of real-world correctness."
        )

    finally:
        if wav_path and Path(wav_path).exists():
            Path(wav_path).unlink()

        del bert_model
        del yamnet_model
        del whisper_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Run report generation only after successful inference and cleanup.
    if inference_result is not None and not args.no_report:
        generate_research_report_for_video(
            video_path,
            inference_result
        )

    return inference_result


if __name__ == "__main__":
    main()
