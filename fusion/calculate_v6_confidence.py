import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ============================================================
# V6 CONFIDENCE SCORE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

TEXT_CACHE = "data/text_v2_feature_cache_test.pt"
AUDIO_CACHE = "data/audio_v4_feature_cache_test.pt"
VIDEO_CACHE = "data/fer_vit_feature_cache_test.pt"

CHECKPOINT = "checkpoints/v6_textv2_fusion_best.pt"

OUTPUT_FILE = "results/v6_confidence_scores.csv"


# ============================================================
# MODEL
# ============================================================

class V6FusionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.LayerNorm(2304),

            nn.Linear(2304, 512),
            nn.GELU(),
            nn.Dropout(0.30),

            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.network(x)

# ============================================================
# LOAD FEATURES
# ============================================================

print()
print("=" * 80)
print("V6 MODEL CONFIDENCE SCORE")
print("=" * 80)

print()
print("Device:", DEVICE)

print()
print("Loading test features...")


text_data = torch.load(
    TEXT_CACHE,
    map_location="cpu",
    weights_only=False
)

audio_data = torch.load(
    AUDIO_CACHE,
    map_location="cpu",
    weights_only=False
)

video_data = torch.load(
    VIDEO_CACHE,
    map_location="cpu",
    weights_only=False
)


# ============================================================
# EXTRACT FEATURES
# ============================================================

text_features = text_data["features"].float()

audio_features = audio_data["features"].float()

video_features = video_data["features"].float()


labels = text_data["labels"]

clip_ids = text_data["clip_ids"]


print("Text features :", text_features.shape)
print("Audio features:", audio_features.shape)
print("Video features:", video_features.shape)


# ============================================================
# ALIGNMENT CHECK
# ============================================================

assert len(text_features) == len(audio_features)
assert len(text_features) == len(video_features)

assert len(labels) == len(text_features)
assert len(clip_ids) == len(text_features)

print()
print("Feature alignment: PASS")
print("Test samples:", len(text_features))


# ============================================================
# CONCATENATION
# ============================================================

features = torch.cat(
    [
        text_features,
        audio_features,
        video_features
    ],
    dim=1
)

print("Fused feature shape:", features.shape)


# ============================================================
# LOAD MODEL
# ============================================================

model = V6FusionModel().to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)


# Handle different checkpoint formats
if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

else:
    state_dict = checkpoint


# Remove possible "module." prefix
clean_state_dict = {}

for key, value in state_dict.items():

    new_key = key

    if new_key.startswith("module."):
        new_key = new_key[7:]

    clean_state_dict[new_key] = value


model.load_state_dict(
    clean_state_dict,
    strict=True
)

model.eval()


print()
print("V6 checkpoint loaded successfully.")


# ============================================================
# LABELS
# ============================================================

emotion_names = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise"
]


# ============================================================
# PREDICTION
# ============================================================

all_confidences = []
all_predictions = []
all_probabilities = []
all_correct = []

batch_size = 128


print()
print("Calculating prediction confidence...")


with torch.no_grad():

    for start in range(
        0,
        len(features),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(features)
        )

        batch = features[start:end].to(DEVICE)

        logits = model(batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

        all_confidences.extend(
            confidence.cpu().numpy()
        )

        all_predictions.extend(
            prediction.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_confidences = np.array(
    all_confidences
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)

labels_numeric = np.array(
    [
        int(x)
        for x in labels
    ]
)


# ============================================================
# CORRECT / INCORRECT
# ============================================================

all_correct = (
    all_predictions == labels_numeric
)


# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

mean_confidence = np.mean(
    all_confidences
)

median_confidence = np.median(
    all_confidences
)

min_confidence = np.min(
    all_confidences
)

max_confidence = np.max(
    all_confidences
)


correct_confidence = all_confidences[
    all_correct
]

incorrect_confidence = all_confidences[
    ~all_correct
]


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 80)
print("V6 CONFIDENCE RESULTS")
print("=" * 80)

print(
    "Average Confidence : {:.2f}%".format(
        mean_confidence * 100
    )
)

print(
    "Median Confidence  : {:.2f}%".format(
        median_confidence * 100
    )
)

print(
    "Minimum Confidence : {:.2f}%".format(
        min_confidence * 100
    )
)

print(
    "Maximum Confidence : {:.2f}%".format(
        max_confidence * 100
    )
)


# ============================================================
# CORRECT VS INCORRECT CONFIDENCE
# ============================================================

if len(correct_confidence) > 0:

    print(
        "Correct Prediction Confidence   : {:.2f}%".format(
            np.mean(correct_confidence) * 100
        )
    )


if len(incorrect_confidence) > 0:

    print(
        "Incorrect Prediction Confidence : {:.2f}%".format(
            np.mean(incorrect_confidence) * 100
        )
    )


# ============================================================
# CONFIDENCE RANGES
# ============================================================

high_confidence = np.mean(
    all_confidences >= 0.80
)

medium_confidence = np.mean(
    (all_confidences >= 0.50)
    & (all_confidences < 0.80)
)

low_confidence = np.mean(
    all_confidences < 0.50
)


print()
print("CONFIDENCE DISTRIBUTION")
print("-" * 50)

print(
    "High   (>=80%) : {:.2f}%".format(
        high_confidence * 100
    )
)

print(
    "Medium (50-80%): {:.2f}%".format(
        medium_confidence * 100
    )
)

print(
    "Low    (<50%)  : {:.2f}%".format(
        low_confidence * 100
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print()
print("=" * 80)
print("SAMPLE PREDICTIONS WITH CONFIDENCE")
print("=" * 80)

print(
    "{:<20}{:<15}{:<15}{:<15}".format(
        "Clip ID",
        "Actual",
        "Predicted",
        "Confidence"
    )
)

print("-" * 80)


for i in range(
    min(20, len(features))
):

    actual = emotion_names[
        labels_numeric[i]
    ]

    predicted = emotion_names[
        all_predictions[i]
    ]

    confidence = all_confidences[i] * 100

    print(
        "{:<20}{:<15}{:<15}{:>10.2f}%".format(
            str(clip_ids[i])[:20],
            actual,
            predicted,
            confidence
        )
    )


# ============================================================
# SAVE ALL SAMPLE CONFIDENCES
# ============================================================

rows = []

for i in range(
    len(features)
):

    row = {
        "clip_id": clip_ids[i],
        "actual": emotion_names[
            labels_numeric[i]
        ],
        "predicted": emotion_names[
            all_predictions[i]
        ],
        "confidence": all_confidences[i],
        "confidence_percent":
            all_confidences[i] * 100,
        "correct":
            bool(all_correct[i])
    }

    for j, emotion in enumerate(
        emotion_names
    ):

        row[
            "prob_" + emotion
        ] = all_probabilities[i, j]

    rows.append(row)


confidence_df = pd.DataFrame(
    rows
)


confidence_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("CONFIDENCE ANALYSIS COMPLETED")
print("=" * 80)

print()
print(
    "Average Model Confidence: {:.2f}%".format(
        mean_confidence * 100
    )
)

print(
    "Test Samples: {}".format(
        len(features)
    )
)

print()
print(
    "Saved detailed confidence file:"
)

print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print()