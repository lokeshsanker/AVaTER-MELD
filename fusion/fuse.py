"""
Fusion of the three branches' softmax outputs.

Two modes are supported:

1. Weighted soft voting (default, no extra training needed):
       P_final = w_img * P_img + w_audio * P_audio + w_text * P_text
   Weights can be set manually or fit automatically from each branch's
   standalone validation accuracy (`fit_weights_from_accuracy`).

2. Stacking meta-learner (optional, usually a small accuracy gain if you have
   enough validation data): a logistic regression trained on the concatenation
   of the three probability vectors, letting the model learn per-class,
   per-modality trust rather than a single global weight per modality.

Usage:
    python fusion/fuse.py --mode fit --val_manifest data/val.csv
    python fusion/fuse.py --mode predict --manifest data/test.csv
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader

from config import (
    DEVICE, NUM_CLASSES, IDX2LABEL, VIDEO_CKPT, AUDIO_CKPT, TEXT_CKPT,
    FUSION_WEIGHTS_PATH, STACKER_CKPT,
)
from data.dataset import load_manifest, encode_labels
from video.video_dataset import VideoEmotionDataset
from video.model import VideoEmotionModel
from audio.audio_dataset import AudioEmotionDataset
from audio.model import AudioEmotionModel
from text.text_dataset import TextEmotionDataset
from text.model import TextEmotionModel


def _load_video_model():
    model = VideoEmotionModel().to(DEVICE)
    model.load_state_dict(torch.load(VIDEO_CKPT, map_location=DEVICE))
    model.eval()
    return model


def _load_audio_model():
    model = AudioEmotionModel().to(DEVICE)
    model.load_state_dict(torch.load(AUDIO_CKPT, map_location=DEVICE))
    model.eval()
    return model


def _load_text_model():
    model = TextEmotionModel().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(TEXT_CKPT, "pytorch_model.bin"), map_location=DEVICE))
    model.eval()
    return model


def get_branch_probs(df, batch_size: int = 8):
    """
    Runs all three branches over `df` (which must be ordered consistently,
    i.e. one clip per row shared across modalities) and returns
    (P_img, P_audio, P_text, y_true), each an (N, NUM_CLASSES) array except
    y_true which is (N,).
    """
    video_model = _load_video_model()
    audio_model = _load_audio_model()
    text_model = _load_text_model()

    video_loader = DataLoader(VideoEmotionDataset(df), batch_size=batch_size, shuffle=False)
    audio_loader = DataLoader(AudioEmotionDataset(df), batch_size=batch_size, shuffle=False)
    text_loader = DataLoader(TextEmotionDataset(df), batch_size=batch_size, shuffle=False)

    P_img, P_audio, P_text, y_true = [], [], [], []

    with torch.no_grad():
        for frames, labels, _ in video_loader:
            probs = video_model.predict_proba(frames.to(DEVICE))
            P_img.append(probs.cpu().numpy())
            y_true.append(labels.numpy())

        for input_values, _, _ in audio_loader:
            probs = audio_model.predict_proba(input_values.to(DEVICE))
            P_audio.append(probs.cpu().numpy())

        for input_ids, attn_mask, _, _ in text_loader:
            probs = text_model.predict_proba(input_ids.to(DEVICE), attn_mask.to(DEVICE))
            P_text.append(probs.cpu().numpy())

    return (np.concatenate(P_img), np.concatenate(P_audio),
            np.concatenate(P_text), np.concatenate(y_true))


def fit_weights_from_accuracy(P_img, P_audio, P_text, y_true):
    """Set fusion weights proportional to each branch's standalone accuracy."""
    accs = {}
    for name, P in [("image", P_img), ("audio", P_audio), ("text", P_text)]:
        preds = P.argmax(axis=1)
        accs[name] = accuracy_score(y_true, preds)

    total = sum(accs.values())
    weights = {k: v / total for k, v in accs.items()}
    print(f"Standalone branch accuracies: {accs}")
    print(f"Fitted fusion weights: {weights}")
    return weights


def weighted_fuse(P_img, P_audio, P_text, weights: dict):
    P_final = (weights["image"] * P_img + weights["audio"] * P_audio + weights["text"] * P_text)
    return P_final


def fit_stacker(P_img, P_audio, P_text, y_true):
    """Alternative to hand-set weights: learn a small logistic regression on
    the concatenated probability vectors."""
    X = np.concatenate([P_img, P_audio, P_text], axis=1)  # (N, 3*NUM_CLASSES)
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X, y_true)
    return clf


def stacker_predict(clf, P_img, P_audio, P_text):
    X = np.concatenate([P_img, P_audio, P_text], axis=1)
    return clf.predict_proba(X)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fit", "predict"], required=True)
    parser.add_argument("--val_manifest", help="Required for --mode fit")
    parser.add_argument("--manifest", help="Required for --mode predict")
    parser.add_argument("--method", choices=["weighted", "stacking"], default="weighted")
    args = parser.parse_args()

    if args.mode == "fit":
        df = load_manifest(args.val_manifest)
        P_img, P_audio, P_text, y_true = get_branch_probs(df)

        if args.method == "weighted":
            weights = fit_weights_from_accuracy(P_img, P_audio, P_text, y_true)
            with open(FUSION_WEIGHTS_PATH, "w") as f:
                json.dump(weights, f)
            P_final = weighted_fuse(P_img, P_audio, P_text, weights)
        else:
            clf = fit_stacker(P_img, P_audio, P_text, y_true)
            joblib.dump(clf, STACKER_CKPT)
            P_final = stacker_predict(clf, P_img, P_audio, P_text)

        preds = P_final.argmax(axis=1)
        print("\nFused validation performance:")
        print(classification_report(y_true, preds, target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)]))

    else:  # predict
        df = load_manifest(args.manifest)
        P_img, P_audio, P_text, y_true = get_branch_probs(df)

        if args.method == "weighted":
            with open(FUSION_WEIGHTS_PATH) as f:
                weights = json.load(f)
            P_final = weighted_fuse(P_img, P_audio, P_text, weights)
        else:
            clf = joblib.load(STACKER_CKPT)
            P_final = stacker_predict(clf, P_img, P_audio, P_text)

        preds = P_final.argmax(axis=1)
        for clip_id, pred, true in zip(df["clip_id"], preds, y_true):
            print(f"{clip_id}: predicted={IDX2LABEL[pred]}  true={IDX2LABEL[true]}")

        print("\nFused test performance:")
        print(classification_report(y_true, preds, target_names=[IDX2LABEL[i] for i in range(NUM_CLASSES)]))


if __name__ == "__main__":
    main()
