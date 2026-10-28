"""Defaults for the non-destructive MELD research upgrade.
CLI flags in fusion/train_research_fusion.py override these documented defaults.
"""
LOSS_TYPE = 'weighted_ce'       # cross_entropy | weighted_ce | focal
FOCAL_GAMMA = 2.0
LABEL_SMOOTHING = 0.1
FUSION_TYPE = 'concat'          # concat | gated | cross_attention
AUDIO_BACKBONE = 'wav2vec2'     # wav2vec2 | hubert
VIDEO_BACKBONE = 'mobilenet'    # mobilenet | videomae | timesformer | swin
SEED = 42
BATCH_SIZE_FUSION = 64
BATCH_SIZE_AUDIO = 2            # RTX 2050 4GB recommendation
BATCH_SIZE_VIDEO = 1            # conservative for video transformers
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
USE_AMP = True
USE_TENSORBOARD = False
USE_WANDB = False
