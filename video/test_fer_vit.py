import os
import sys
import numpy as np
import pandas as pd
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LABEL2IDX


MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"

MANIFEST = "data/manifests/dev.csv"
CACHE_DIR = "data/face_cache/dev"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)
print("Loading:", MODEL_NAME)

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("\nModel labels:")
print(model.config.id2label)


df = pd.read_csv(MANIFEST)

print("\nDev samples:", len(df))


# ---------------------------------------------------------
# MELD -> FER label mapping
# ---------------------------------------------------------

meld_to_fer = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
}


correct = 0
total = 0

class_correct = {x: 0 for x in LABEL2IDX.keys()}
class_total = {x: 0 for x in LABEL2IDX.keys()}


# ---------------------------------------------------------
# Test every dev clip
# ---------------------------------------------------------

with torch.no_grad():

    for idx, row in df.iterrows():

        clip_id = row["clip_id"]
        true_label = row["label"]

        path = os.path.join(
            CACHE_DIR,
            clip_id + ".npy"
        )

        faces = np.load(path)

        # Same 4-frame sampling as our dataset
        if faces.shape[0] > 4:

            indices = np.linspace(
                0,
                faces.shape[0] - 1,
                4
            ).astype(int)

            faces = faces[indices]


        # Predict each frame separately
        frame_predictions = []

        for face in faces:

            image = Image.fromarray(
                face.astype(np.uint8)
            )

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            inputs = {
                k: v.to(DEVICE)
                for k, v in inputs.items()
            }

            outputs = model(**inputs)

            probs = torch.softmax(
                outputs.logits,
                dim=-1
            )

            pred_idx = probs.argmax(
                dim=-1
            ).item()

            pred_label = model.config.id2label[
                pred_idx
            ].lower()

            frame_predictions.append(pred_label)


        # Majority vote over four frames
        from collections import Counter

        predicted_fer = Counter(
            frame_predictions
        ).most_common(1)[0][0]


        # Convert FER label to MELD label
        fer_to_meld = {
            "neutral": "neutral",
            "happy": "joy",
            "sad": "sadness",
            "angry": "anger",
            "fear": "fear",
            "disgust": "disgust",
            "surprise": "surprise",
        }

        predicted_meld = fer_to_meld.get(
            predicted_fer,
            predicted_fer
        )


        class_total[true_label] += 1

        if predicted_meld == true_label:
            correct += 1
            class_correct[true_label] += 1

        total += 1


        if idx < 20:

            print(
                f"{clip_id:15s} "
                f"true={true_label:9s} "
                f"frames={frame_predictions} "
                f"pred={predicted_meld}"
            )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n" + "=" * 60)

print(
    f"Overall accuracy: "
    f"{correct}/{total} = "
    f"{correct / total:.4f}"
)

print("\nPer-class accuracy:")

for label in LABEL2IDX.keys():

    acc = (
        class_correct[label] /
        class_total[label]
        if class_total[label] > 0
        else 0
    )

    print(
        f"{label:9s}: "
        f"{class_correct[label]:4d}/"
        f"{class_total[label]:4d} "
        f"= {acc:.4f}"
    )

print("=" * 60)
