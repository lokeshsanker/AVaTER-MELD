"""
Quantize the audio branch (Wav2Vec2 encoder + attention pool + MLP head).

Wav2Vec2's cost is dominated by its transformer Linear layers, so dynamic INT8
quantization (weights quantized ahead of time, activations quantized on the
fly at inference) gives most of the available size/latency win with no
calibration data required and typically minimal accuracy loss.

If you need to go smaller still, consider first distilling to a smaller speech
encoder (e.g. DistilHuBERT) before quantizing, rather than pushing PTQ/QAT
further on the full wav2vec2-base.

Usage:
    python quantization/quantize_audio.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config import AUDIO_CKPT, AUDIO_CKPT_QUANT
from audio.model import AudioEmotionModel


def main():
    model = AudioEmotionModel()
    model.load_state_dict(torch.load(AUDIO_CKPT, map_location="cpu"))
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), AUDIO_CKPT_QUANT)

    fp32_size = os.path.getsize(AUDIO_CKPT) / 1e6
    int8_size = os.path.getsize(AUDIO_CKPT_QUANT) / 1e6
    print(f"FP32 checkpoint: {fp32_size:.2f} MB")
    print(f"INT8 checkpoint: {int8_size:.2f} MB  ({fp32_size / int8_size:.2f}x smaller)")
    print(f"Saved quantized audio model to {AUDIO_CKPT_QUANT}")


if __name__ == "__main__":
    main()
