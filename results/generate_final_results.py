import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# MELD MULTIMODAL EMOTION RECOGNITION
# FINAL RESULTS GENERATOR
# ============================================================

RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "final_visualizations"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. VERIFIED OVERALL RESULTS
# ============================================================

results = {
    "Video V3": {
        "Accuracy": 0.3153,
        "Weighted_F1": 0.3091,
        "Macro_F1": 0.1558,
    },

    "Audio V4": {
        "Accuracy": 0.4360,
        "Weighted_F1": 0.3788,
        "Macro_F1": 0.1831,
    },

    "Text V2": {
        "Accuracy": 0.5908,
        "Weighted_F1": 0.5933,
        "Macro_F1": 0.4169,
    },

    "V4 Fusion": {
        "Accuracy": 0.4782,
        "Weighted_F1": 0.4018,
        "Macro_F1": 0.1910,
    },

    "V5 Cross-Attention": {
        "Accuracy": 0.4176,
        "Weighted_F1": 0.3810,
        "Macro_F1": 0.1920,
    },

    "V6 Fusion": {
        "Accuracy": 0.6176,
        "Weighted_F1": 0.6031,
        "Macro_F1": 0.4208,
    },

    "V7 Fusion": {
        "Accuracy": 0.6153,
        "Weighted_F1": 0.5954,
        "Macro_F1": 0.4010,
    },
}


# ============================================================
# 2. DATAFRAME
# ============================================================

df = pd.DataFrame.from_dict(
    results,
    orient="index"
)


# ============================================================
# 3. TERMINAL FINAL SUMMARY
# ============================================================

print()
print("=" * 95)
print("FINAL SUMMARY - MELD MULTIMODAL EMOTION RECOGNITION")
print("=" * 95)

header = "{:<25}{:>15}{:>18}{:>15}".format(
    "Model",
    "Accuracy",
    "Weighted-F1",
    "Macro-F1"
)

print(header)
print("-" * 95)

for model, row in df.iterrows():

    line = "{:<25}{:>15.4f}{:>18.4f}{:>15.4f}".format(
        model,
        row["Accuracy"],
        row["Weighted_F1"],
        row["Macro_F1"]
    )

    print(line)

print("=" * 95)


# ============================================================
# 4. BEST MODEL
# ============================================================

best_model = df["Accuracy"].idxmax()

print()
print("BEST MODEL BY ACCURACY")
print("-" * 40)
print("Model       :", best_model)
print("Accuracy    : {:.2%}".format(
    df.loc[best_model, "Accuracy"]
))
print("Weighted-F1 : {:.2%}".format(
    df.loc[best_model, "Weighted_F1"]
))
print("Macro-F1    : {:.2%}".format(
    df.loc[best_model, "Macro_F1"]
))
print()


# ============================================================
# 5. SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "final_model_comparison.csv"
)

df.to_csv(csv_path)

print("Saved:", csv_path)


# ============================================================
# 6. ACCURACY GRAPH
# ============================================================

models = list(df.index)

plt.figure(figsize=(13, 7))

bars = plt.bar(
    models,
    df["Accuracy"] * 100
)

plt.title(
    "Model Accuracy Comparison on MELD Test Set",
    fontsize=16
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 70)

for bar, value in zip(
    bars,
    df["Accuracy"] * 100
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        "{:.2f}%".format(value),
        ha="center"
    )

plt.tight_layout()

accuracy_path = os.path.join(
    OUTPUT_DIR,
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", accuracy_path)


# ============================================================
# 7. WEIGHTED F1 GRAPH
# ============================================================

plt.figure(figsize=(13, 7))

bars = plt.bar(
    models,
    df["Weighted_F1"] * 100
)

plt.title(
    "Weighted-F1 Comparison on MELD Test Set",
    fontsize=16
)

plt.ylabel("Weighted-F1 (%)")
plt.xlabel("Model")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 70)

for bar, value in zip(
    bars,
    df["Weighted_F1"] * 100
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.8,
        "{:.2f}%".format(value),
        ha="center"
    )

plt.tight_layout()

weighted_path = os.path.join(
    OUTPUT_DIR,
    "weighted_f1_comparison.png"
)

plt.savefig(
    weighted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", weighted_path)


# ============================================================
# 8. MACRO F1 GRAPH
# ============================================================

plt.figure(figsize=(13, 7))

bars = plt.bar(
    models,
    df["Macro_F1"] * 100
)

plt.title(
    "Macro-F1 Comparison on MELD Test Set",
    fontsize=16
)

plt.ylabel("Macro-F1 (%)")
plt.xlabel("Model")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 50)

for bar, value in zip(
    bars,
    df["Macro_F1"] * 100
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.7,
        "{:.2f}%".format(value),
        ha="center"
    )

plt.tight_layout()

macro_path = os.path.join(
    OUTPUT_DIR,
    "macro_f1_comparison.png"
)

plt.savefig(
    macro_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", macro_path)


# ============================================================
# 9. COMPLETE MODEL COMPARISON
# ============================================================

x = np.arange(len(models))

width = 0.25

plt.figure(figsize=(15, 8))

plt.bar(
    x - width,
    df["Accuracy"] * 100,
    width,
    label="Accuracy"
)

plt.bar(
    x,
    df["Weighted_F1"] * 100,
    width,
    label="Weighted-F1"
)

plt.bar(
    x + width,
    df["Macro_F1"] * 100,
    width,
    label="Macro-F1"
)

plt.ylabel("Score (%)")

plt.title(
    "MELD Multimodal Emotion Recognition - Model Comparison",
    fontsize=16
)

plt.xticks(
    x,
    models,
    rotation=30,
    ha="right"
)

plt.ylim(0, 70)

plt.legend()

plt.tight_layout()

comparison_path = os.path.join(
    OUTPUT_DIR,
    "complete_model_comparison.png"
)

plt.savefig(
    comparison_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", comparison_path)


# ============================================================
# 10. PER-CLASS F1
# ============================================================

classes = [
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise"
]


per_class = {

    "Video V3": [
        0.4821,
        0.2430,
        0.0938,
        0.2021,
        0.0000,
        0.0247,
        0.0451
    ],

    "Audio V4": [
        0.61,
        0.27,
        0.17,
        0.36,
        0.14,
        0.00,
        0.04
    ],

    "Text V2": [
        0.75,
        0.57,
        0.27,
        0.43,
        0.13,
        0.22,
        0.54
    ],

    "V6 Fusion": [
        0.77,
        0.57,
        0.27,
        0.43,
        0.09,
        0.26,
        0.55
    ]
}


class_df = pd.DataFrame(
    per_class,
    index=classes
).T


# ============================================================
# 11. PER-CLASS F1 TERMINAL TABLE
# ============================================================

print()
print("=" * 105)
print("PER-CLASS F1 COMPARISON")
print("=" * 105)

class_header = "{:<20}".format("Model")

for emotion in classes:
    class_header += "{:>12}".format(emotion)

print(class_header)
print("-" * 105)

for model, row in class_df.iterrows():

    line = "{:<20}".format(model)

    for value in row:
        line += "{:>12.3f}".format(value)

    print(line)

print("=" * 105)


# ============================================================
# 12. PER-CLASS F1 HEATMAP
# ============================================================

plt.figure(figsize=(12, 6))

plt.imshow(
    class_df.values,
    aspect="auto"
)

plt.colorbar(
    label="F1 Score"
)

plt.xticks(
    range(len(classes)),
    classes,
    rotation=30
)

plt.yticks(
    range(len(class_df.index)),
    class_df.index
)

plt.title(
    "Per-Class F1: Modalities vs Best Fusion",
    fontsize=16
)

for i in range(len(class_df.index)):

    for j in range(len(classes)):

        value = class_df.iloc[i, j]

        plt.text(
            j,
            i,
            "{:.2f}".format(value),
            ha="center",
            va="center"
        )

plt.tight_layout()

heatmap_path = os.path.join(
    OUTPUT_DIR,
    "per_class_f1_heatmap.png"
)

plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", heatmap_path)


# ============================================================
# 13. RADAR CHART
# ============================================================

angles = np.linspace(
    0,
    2 * np.pi,
    len(classes),
    endpoint=False
).tolist()

angles += angles[:1]

plt.figure(figsize=(9, 9))

radar_models = [
    "Video V3",
    "Audio V4",
    "Text V2",
    "V6 Fusion"
]

for model in radar_models:

    values = class_df.loc[model].tolist()

    values += values[:1]

    plt.plot(
        angles,
        values,
        linewidth=2,
        label=model
    )

    plt.fill(
        angles,
        values,
        alpha=0.05
    )


plt.xticks(
    angles[:-1],
    classes
)

plt.ylim(
    0,
    1.0
)

plt.title(
    "Per-Class F1 Radar - Modalities vs Best Fusion",
    fontsize=16
)

plt.legend(
    loc="upper right",
    bbox_to_anchor=(1.30, 1.10)
)

plt.tight_layout()

radar_path = os.path.join(
    OUTPUT_DIR,
    "per_class_f1_radar.png"
)

plt.savefig(
    radar_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved:", radar_path)


# ============================================================
# 14. FINAL TEXT REPORT
# ============================================================

summary_path = os.path.join(
    OUTPUT_DIR,
    "FINAL_RESULTS_SUMMARY.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "=" * 95 + "\n"
    )

    f.write(
        "FINAL SUMMARY - MELD MULTIMODAL EMOTION RECOGNITION\n"
    )

    f.write(
        "=" * 95 + "\n"
    )

    f.write(
        "{:<25}{:>15}{:>18}{:>15}\n".format(
            "Model",
            "Accuracy",
            "Weighted-F1",
            "Macro-F1"
        )
    )

    f.write(
        "-" * 95 + "\n"
    )

    for model, row in df.iterrows():

        f.write(
            "{:<25}{:>15.4f}{:>18.4f}{:>15.4f}\n".format(
                model,
                row["Accuracy"],
                row["Weighted_F1"],
                row["Macro_F1"]
            )
        )

    f.write(
        "=" * 95 + "\n\n"
    )

    f.write(
        "BEST MODEL: {}\n".format(best_model)
    )

    f.write(
        "Accuracy: {:.2%}\n".format(
            df.loc[best_model, "Accuracy"]
        )
    )

    f.write(
        "Weighted-F1: {:.2%}\n".format(
            df.loc[best_model, "Weighted_F1"]
        )
    )

    f.write(
        "Macro-F1: {:.2%}\n".format(
            df.loc[best_model, "Macro_F1"]
        )
    )


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print()
print("=" * 95)
print("ALL FINAL RESULTS GENERATED SUCCESSFULLY")
print("=" * 95)
print()
print("Output folder:")
print(
    os.path.abspath(OUTPUT_DIR)
)
print()