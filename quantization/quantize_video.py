"""
Quantize the video branch (MobileNetV2 backbone + BiGRU + classifier).

CNNs benefit from *static* post-training quantization (PTQ) with a small
calibration set (activation ranges are calibrated, not just weights). If the
accuracy drop from PTQ is too large (check against your validation set),
switch to quantization-aware training (QAT) — a QAT scaffold is included
below, commented, since it requires re-running a few epochs of fine-tuning
with fake-quantization ops inserted.

Note: PyTorch's native quantization currently targets CPU inference (x86 or
ARM via qnnpack), which matches typical lightweight/edge deployment targets.
The recurrent (GRU) part of the model is quantized dynamically since static
quantization support for RNNs is limited; the CNN backbone is quantized
statically.

Usage:
    python quantization/quantize_video.py --calib_manifest data/val.csv
"""

import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE, VIDEO_CKPT, VIDEO_CKPT_QUANT
from data.dataset import load_manifest
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel, FrameEncoder


def quantize_backbone_static(model: VideoEmotionModel, calib_loader) -> VideoEmotionModel:
    """Static INT8 PTQ of the MobileNetV2 backbone only (the part that dominates
    parameter count and FLOPs). The GRU + classifier head are left in FP32 here
    and separately dynamically quantized below, since static quantization of
    recurrent layers is not well supported in stock PyTorch."""

    model = model.to("cpu")
    model.eval()

    backbone = model.frame_encoder
    backbone.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    torch.backends.quantized.engine = "qnnpack"

    torch.quantization.prepare(backbone, inplace=True)

    # Calibration pass: run a handful of real batches through the backbone so
    # activation ranges are estimated from real data, not just weights.
    with torch.no_grad():
        for frames, _, _ in calib_loader:
            B, T, C, H, W = frames.shape
            backbone(frames.view(B * T, C, H, W))

    torch.quantization.convert(backbone, inplace=True)
    model.frame_encoder = backbone
    return model


def quantize_gru_and_head_dynamic(model: VideoEmotionModel) -> VideoEmotionModel:
    """Dynamic INT8 quantization of the GRU + linear classifier layers."""
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    return quantized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib_manifest", required=True,
                         help="A small manifest (e.g. 100-200 clips from val) used to "
                              "calibrate activation ranges for static quantization.")
    args = parser.parse_args()

    calib_df = load_manifest(args.calib_manifest)
    calib_loader = DataLoader(VideoEmotionDataset(calib_df), batch_size=4, shuffle=False)

    model = VideoEmotionModel()
    model.load_state_dict(torch.load(VIDEO_CKPT, map_location="cpu"))

    print("Statically quantizing CNN backbone (calibrating on real data)...")
    model = quantize_backbone_static(model, calib_loader)

    print("Dynamically quantizing GRU + classifier head...")
    model = quantize_gru_and_head_dynamic(model)

    torch.save(model.state_dict(), VIDEO_CKPT_QUANT)

    fp32_size = os.path.getsize(VIDEO_CKPT) / 1e6
    int8_size = os.path.getsize(VIDEO_CKPT_QUANT) / 1e6
    print(f"FP32 checkpoint: {fp32_size:.2f} MB")
    print(f"INT8 checkpoint: {int8_size:.2f} MB  ({fp32_size / int8_size:.2f}x smaller)")
    print(f"Saved quantized video model to {VIDEO_CKPT_QUANT}")
    print("\nIf validation accuracy drops more than ~2-3 points versus the FP32 model, "
          "switch to quantization-aware training (QAT): insert "
          "torch.quantization.prepare_qat before a few epochs of fine-tuning, "
          "then torch.quantization.convert, instead of plain static PTQ.")


if __name__ == "__main__":
    main()
