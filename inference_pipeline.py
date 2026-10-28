"""
End-to-end inference on a single new 3-second video clip: extracts video
frames, audio, and a transcript; runs all three branches; fuses their
predictions; prints the final emotion.

Usage:
    python inference_pipeline.py --video path/to/clip.mp4 [--use_quantized]
"""

import argparse
import json
import os
import subprocess
import tempfile

import numpy as np
import torch

from config import (
    DEVICE, NUM_FRAMES, FACE_SIZE, IDX2LABEL,
    VIDEO_CKPT, AUDIO_CKPT, TEXT_CKPT,
    VIDEO_CKPT_QUANT, AUDIO_CKPT_QUANT, TEXT_CKPT_QUANT,
    FUSION_WEIGHTS_PATH,
)
from video.face_utils import extract_aligned_faces
from video.model import VideoEmotionModel
from audio.model import AudioEmotionModel
from text.model import TextEmotionModel
from text.asr import transcribe

import torchvision.transforms as T
import librosa
from transformers import Wav2Vec2FeatureExtractor, DistilBertTokenizerFast
from config import AUDIO_SAMPLE_RATE, WAV2VEC2_MODEL_NAME, TEXT_MODEL_NAME, MAX_TEXT_LEN

_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_MODEL_NAME)
_bert_tokenizer = DistilBertTokenizerFast.from_pretrained(TEXT_MODEL_NAME)


def extract_wav_from_video(video_path: str) -> str:
    """Extracts a 16kHz mono wav file from the video using ffmpeg."""
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE),
        tmp_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav


def prepare_video_input(video_path: str) -> torch.Tensor:
    faces = extract_aligned_faces(video_path, NUM_FRAMES, FACE_SIZE)  # (T,H,W,3) uint8
    frames = torch.from_numpy(faces).float().permute(0, 3, 1, 2) / 255.0
    frames = torch.stack([_normalize(f) for f in frames])
    return frames.unsqueeze(0)  # (1, T, 3, H, W)


def prepare_audio_input(wav_path: str, max_seconds: float = 3.5) -> torch.Tensor:
    wav, _ = librosa.load(wav_path, sr=AUDIO_SAMPLE_RATE, mono=True)
    max_samples = int(max_seconds * AUDIO_SAMPLE_RATE)
    if len(wav) > max_samples:
        wav = wav[:max_samples]
    else:
        wav = np.pad(wav, (0, max_samples - len(wav)))
    inputs = _wav2vec_extractor(wav, sampling_rate=AUDIO_SAMPLE_RATE, return_tensors="pt")
    return inputs["input_values"]  # (1, num_samples)


def prepare_text_input(text: str):
    enc = _bert_tokenizer(
        text, truncation=True, padding="max_length", max_length=MAX_TEXT_LEN, return_tensors="pt"
    )
    return enc["input_ids"], enc["attention_mask"]


def load_models(use_quantized: bool):
    if use_quantized:
        import torch.nn as nn
        video_model = VideoEmotionModel()
        video_model.load_state_dict(torch.load(VIDEO_CKPT_QUANT, map_location="cpu"))

        audio_model = AudioEmotionModel()
        audio_model = torch.quantization.quantize_dynamic(audio_model, {nn.Linear}, dtype=torch.qint8)
        audio_model.load_state_dict(torch.load(AUDIO_CKPT_QUANT, map_location="cpu"))

        text_model = TextEmotionModel()
        text_model = torch.quantization.quantize_dynamic(text_model, {nn.Linear}, dtype=torch.qint8)
        text_model.load_state_dict(torch.load(TEXT_CKPT_QUANT, map_location="cpu"))

        device = "cpu"
    else:
        video_model = VideoEmotionModel().to(DEVICE)
        video_model.load_state_dict(torch.load(VIDEO_CKPT, map_location=DEVICE))

        audio_model = AudioEmotionModel().to(DEVICE)
        audio_model.load_state_dict(torch.load(AUDIO_CKPT, map_location=DEVICE))

        text_model = TextEmotionModel().to(DEVICE)
        text_model.load_state_dict(torch.load(os.path.join(TEXT_CKPT, "pytorch_model.bin"), map_location=DEVICE))

        device = DEVICE

    video_model.eval()
    audio_model.eval()
    text_model.eval()
    return video_model, audio_model, text_model, device


def run_pipeline(video_path: str, use_quantized: bool = False):
    video_model, audio_model, text_model, device = load_models(use_quantized)

    wav_path = extract_wav_from_video(video_path)
    try:
        transcript = transcribe(wav_path)

        frames = prepare_video_input(video_path).to(device)
        input_values = prepare_audio_input(wav_path).to(device)
        input_ids, attn_mask = prepare_text_input(transcript)
        input_ids, attn_mask = input_ids.to(device), attn_mask.to(device)

        with torch.no_grad():
            P_img = video_model.predict_proba(frames).cpu().numpy()[0]
            P_audio = audio_model.predict_proba(input_values).cpu().numpy()[0]
            P_text = text_model.predict_proba(input_ids, attn_mask).cpu().numpy()[0]

        with open(FUSION_WEIGHTS_PATH) as f:
            weights = json.load(f)

        P_final = weights["image"] * P_img + weights["audio"] * P_audio + weights["text"] * P_text
        pred_idx = int(np.argmax(P_final))

        return {
            "transcript": transcript,
            "predicted_emotion": IDX2LABEL[pred_idx],
            "confidence": float(P_final[pred_idx]),
            "per_branch": {
                "image": {IDX2LABEL[i]: float(p) for i, p in enumerate(P_img)},
                "audio": {IDX2LABEL[i]: float(p) for i, p in enumerate(P_audio)},
                "text": {IDX2LABEL[i]: float(p) for i, p in enumerate(P_text)},
            },
            "fused": {IDX2LABEL[i]: float(p) for i, p in enumerate(P_final)},
        }
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--use_quantized", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(args.video, use_quantized=args.use_quantized)

    print(f"\nTranscript: {result['transcript']!r}")
    print(f"\nPredicted emotion: {result['predicted_emotion']}  (confidence={result['confidence']:.3f})")
    print("\nPer-branch probabilities:")
    for branch, probs in result["per_branch"].items():
        print(f"  {branch}: " + ", ".join(f"{k}={v:.3f}" for k, v in probs.items()))


if __name__ == "__main__":
    main()
