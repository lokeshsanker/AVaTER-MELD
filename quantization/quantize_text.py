"""
Quantize the text branch (DistilBERT encoder + MLP head).

Dynamic INT8 quantization is the standard, well-supported approach for
transformer Linear layers and is what we apply here. For an even lighter
model, consider starting from TinyBERT or MobileBERT instead of DistilBERT
and quantizing on top -- the two techniques compound.

Usage:
    python quantization/quantize_text.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config import TEXT_CKPT, TEXT_CKPT_QUANT
from text.model import TextEmotionModel


def main():
    model = TextEmotionModel()
    model.load_state_dict(torch.load(os.path.join(TEXT_CKPT, "pytorch_model.bin"), map_location="cpu"))
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    torch.save(quantized_model.state_dict(), TEXT_CKPT_QUANT)

    fp32_path = os.path.join(TEXT_CKPT, "pytorch_model.bin")
    fp32_size = os.path.getsize(fp32_path) / 1e6
    int8_size = os.path.getsize(TEXT_CKPT_QUANT) / 1e6
    print(f"FP32 checkpoint: {fp32_size:.2f} MB")
    print(f"INT8 checkpoint: {int8_size:.2f} MB  ({fp32_size / int8_size:.2f}x smaller)")
    print(f"Saved quantized text model to {TEXT_CKPT_QUANT}")


if __name__ == "__main__":
    main()
