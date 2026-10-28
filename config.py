"""
Shared configuration for all three branches (video / audio / text) and fusion.

IMPORTANT: EMOTIONS is the single source of truth for the label space. Every
branch's classifier head has len(EMOTIONS) output units, and every manifest CSV's
`label` column must use exactly these strings. This is what makes weighted /
stacked fusion across modalities meaningful (see the "matched label spaces"
requirement discussed in the pipeline design).
"""

import os
import torch

# ---------------------------------------------------------------------------
# Label space (edit here only)
# ---------------------------------------------------------------------------
EMOTIONS = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
LABEL2IDX = {label: i for i, label in enumerate(EMOTIONS)}
IDX2LABEL = {i: label for label, i in LABEL2IDX.items()}
NUM_CLASSES = len(EMOTIONS)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Video branch
# ---------------------------------------------------------------------------
NUM_FRAMES = 4              # frames sampled per 3-second clip
FACE_SIZE = 224              # input resolution to the CNN backbone
CNN_FEATURE_DIM = 1280       # MobileNetV2 penultimate feature dim
GRU_HIDDEN = 128
VIDEO_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Audio branch
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 16000
WAV2VEC2_MODEL_NAME = "facebook/wav2vec2-base-960h"
AUDIO_HIDDEN_DIM = 768        # wav2vec2-base hidden size
AUDIO_BATCH_SIZE = 8

# ---------------------------------------------------------------------------
# Text branch
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "base"          # tiny / base / small ...
TEXT_MODEL_NAME = "distilbert-base-uncased"
MAX_TEXT_LEN = 64
TEXT_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR = 2e-4
EPOCHS = 20
EARLY_STOP_PATIENCE = 4

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

VIDEO_CKPT = os.path.join(CHECKPOINT_DIR, "video_model.pt")
AUDIO_CKPT = os.path.join(CHECKPOINT_DIR, "audio_model.pt")
TEXT_CKPT = os.path.join(CHECKPOINT_DIR, "text_model")   # HF save_pretrained dir
FUSION_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "fusion_weights.json")
STACKER_CKPT = os.path.join(CHECKPOINT_DIR, "stacker.joblib")

VIDEO_CKPT_QUANT = os.path.join(CHECKPOINT_DIR, "video_model_int8.pt")
AUDIO_CKPT_QUANT = os.path.join(CHECKPOINT_DIR, "audio_model_int8.pt")
TEXT_CKPT_QUANT = os.path.join(CHECKPOINT_DIR, "text_model_int8.pt")
