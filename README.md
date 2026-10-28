# MELD Research Upgrade (non-destructive patch)

This patch adds a separate research pipeline. It does **not** overwrite V6/V7 checkpoints or existing training scripts.

## Included
- `fusion/train_research_fusion.py`: weighted CE / CE / Focal Loss; concat/gated/cross-attention fusion; Macro-F1 early stopping; AdamW, cosine schedule, clipping, AMP; validation-only temperature scaling; ECE, reliability plots, confidence histograms; classification report, confusion matrix, per-class metrics, ROC-AUC/PR curves, error analysis, predictions and optional modality gates.
- `fusion/run_research_ablation.py`: seven modality ablations.
- `audio/model_advanced.py`, `audio/train_audio_advanced.py`: Wav2Vec2 or HuBERT -> BiLSTM -> attention -> 768-D embedding, train/evaluate and export compatible feature caches.
- `video/model_advanced.py`, `video/train_video_advanced.py`: MobileNet, VideoMAE, TimeSformer, or torchvision Swin3D selector, train/evaluate and export 512-D caches.

## Critical data assumptions
Your ZIP did not include `data/`, MELD media, feature caches or checkpoints. Run these scripts from the original project root where those resources exist. The expected manifest paths are `data/manifests/train.csv`, `dev.csv`, `test.csv`, and face caches under `data/face_cache/{train,dev,test}`.

Default cached fusion uses Text V2 (768), Audio V4 (768), Video V3 (512). The cache loaders verify shapes, sample counts, labels and clip IDs where available. Audio V4's historical cache is a plain tensor, so its ordering cannot be independently checked from IDs; it assumes the established extraction order matches the manifest.

## Recommended first run (fast, cached features)
```powershell
python -m py_compile .\fusion\train_research_fusion.py .\fusion\run_research_ablation.py .\audio\model_advanced.py .\audio\train_audio_advanced.py .\video\model_advanced.py .\video\train_video_advanced.py
python .\fusion\train_research_fusion.py --fusion-type gated --loss-type weighted_ce --run-name research_gated_weighted
```

Other loss options: `cross_entropy`, `weighted_ce`, `focal`. Fusion options: `concat`, `gated`, `cross_attention`.

## Seven-way ablation
```powershell
python .\fusion\run_research_ablation.py --fusion-type concat --loss-type weighted_ce
```

## Advanced audio branch (uses raw WAVs; may take a long time)
```powershell
python .\audio\train_audio_advanced.py --backbone wav2vec2 --batch-size 2 --epochs 12
# Alternative backbone:
python .\audio\train_audio_advanced.py --backbone hubert --batch-size 2 --epochs 12
```

## Advanced video branch (GPU memory caution)
Start with MobileNet:
```powershell
python .\video\train_video_advanced.py --backbone mobilenet --batch-size 1 --epochs 12
```
Try transformer models separately only after MobileNet works. VideoMAE/TimeSformer/Swin3D may need dependencies/model downloads and can exceed 4GB VRAM; use batch size 1 and frozen backbones.
```powershell
python .\video\train_video_advanced.py --backbone videomae --batch-size 1 --epochs 6
python .\video\train_video_advanced.py --backbone timesformer --batch-size 1 --epochs 6
python .\video\train_video_advanced.py --backbone swin --batch-size 1 --epochs 6
```

## Fuse advanced features
After the audio and video scripts successfully export caches:
```powershell
python .\fusion\train_research_fusion.py --fusion-type cross_attention --loss-type focal --audio-cache-dir data --video-cache-dir data
```
Use the explicit cache prefixes so the original cache files remain untouched:
```powershell
python .\fusion\train_research_fusion.py --fusion-type cross_attention --loss-type focal --audio-cache-prefix audio_advanced_feature_cache --video-cache-prefix video_advanced_feature_cache
```

## Metrics and calibration
Temperature is fit using the **dev** split only; test labels are never used for calibration. Test reports and plots are saved to `results/research_runs/<run-name>/`. The scalar temperature changes confidence and ECE but normally does not change argmax-based Accuracy/F1.

## TensorBoard / W&B
CSV epoch history is always saved. This patch does not require W&B. To add TensorBoard later, install `tensorboard` and add a `SummaryWriter` to the epoch loop; W&B can similarly be optional behind an environment flag. No online tracking is started by default.

## Optional experiment tracking
```powershell
pip install tensorboard
python .\fusion\train_research_fusion.py --fusion-type gated --loss-type focal --tensorboard
# In another terminal, from project root:
tensorboard --logdir .\results\research_runs
```
For Weights & Biases, install `wandb`, log in explicitly, then add `--wandb`; no data is uploaded unless you enable it.
```powershell
pip install wandb
wandb login
python .\fusion\train_research_fusion.py --fusion-type gated --loss-type focal --wandb
```

## Installation into the existing Windows project
1. Keep your existing project open in VS Code.
2. Back up the project first (or at least `fusion`, `audio`, `video`, `config.py`, and existing checkpoints).
3. Extract this ZIP directly into `C:\MELD\project\emotion_pipeline\emotion_pipeline`, allowing the new files to merge into the existing `fusion`, `audio`, and `video` folders. No existing files in this patch have the same names as the baseline scripts.
4. `config_research.py` is an additional research settings file; it does not replace your existing `config.py`.

The included smoke test used synthetic in-memory feature caches to validate one-epoch runs of concat, gated and cross-attention fusion, loss functions, calibration, and report generation. The uploaded source ZIP did not include the real `data` caches, MELD media, or checkpoints, so no new real MELD result is claimed until you run it locally.
