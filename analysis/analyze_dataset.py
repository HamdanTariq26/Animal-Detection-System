# -*- coding: utf-8 -*-
"""
analyze_dataset.py

Run:
    python analysis/analyze_dataset.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WILDLIFE_DATASET_INPUT, RAW_CLASS_NAMES  # noqa: E402


def count_images_per_class(label_dir: Path, class_names):
    image_counts = {name: 0 for name in class_names}

    for label_file in os.listdir(label_dir):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, label_file), "r") as f:
            unique_classes_in_photo = set()
            for line in f:
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                unique_classes_in_photo.add(class_names[class_id])

            for species in unique_classes_in_photo:
                image_counts[species] += 1

    return image_counts


def build_dataset_dataframe(image_counts: dict) -> pd.DataFrame:
    df = pd.DataFrame(list(image_counts.items()), columns=["Class_Name", "Image_Count"])
    df = df.sort_values(by="Image_Count", ascending=False).reset_index(drop=True)
    return df


def plot_overview(df_dataset: pd.DataFrame):
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    sns.barplot(x="Image_Count", y="Class_Name", data=df_dataset.head(20), palette="viridis")
    plt.title("Top 20 Species by Image Count", fontsize=14)
    plt.xlabel("Number of Images")

    plt.subplot(2, 2, 2)
    sns.histplot(df_dataset["Image_Count"], bins=15, kde=True, color="teal")
    plt.title("Distribution of Training Samples", fontsize=14)
    plt.xlabel("Images per Class")

    plt.subplot(2, 2, 3)
    top_10 = df_dataset.head(10)
    plt.pie(
        top_10["Image_Count"],
        labels=top_10["Class_Name"],
        autopct="%1.1f%%",
        colors=sns.color_palette("pastel"),
        startangle=140,
    )
    plt.title("Market Share of Top 10 Species", fontsize=14)

    plt.subplot(2, 2, 4)
    sns.boxplot(x=df_dataset["Image_Count"], color="skyblue")
    plt.title("Dataset Variation & Outliers", fontsize=14)
    plt.xlabel("Sample Volume")

    plt.tight_layout()
    plt.show()


def plot_cleaned(df_dataset: pd.DataFrame):
    """Same overview plots but with the Butterfly class excluded, matching
    the original notebook's Cell 3 (post-filtering visualization)."""
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_cleaned = df_dataset[df_dataset["Class_Name"] != "Butterfly"].reset_index(drop=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    sns.barplot(x="Image_Count", y="Class_Name", data=df_cleaned.head(20), palette="magma")
    plt.title("Top 20 Species (Post-Filtering)", fontsize=14)
    plt.xlabel("Number of Images")

    plt.subplot(2, 2, 2)
    sns.histplot(df_cleaned["Image_Count"], bins=12, kde=True, color="indigo")
    plt.title("Balanced Training Sample Distribution", fontsize=14)
    plt.xlabel("Images per Class")

    plt.subplot(2, 2, 3)
    top_10_clean = df_cleaned.head(10)
    plt.pie(
        top_10_clean["Image_Count"],
        labels=top_10_clean["Class_Name"],
        autopct="%1.1f%%",
        colors=sns.color_palette("muted"),
        startangle=140,
    )
    plt.title("Relative Share of Top 10 Wildlife Categories", fontsize=14)

    plt.subplot(2, 2, 4)
    sns.boxplot(x=df_cleaned["Image_Count"], color="salmon")
    plt.title("Data Spread After Outlier Removal", fontsize=14)
    plt.xlabel("Sample Volume")

    plt.tight_layout()
    plt.show()


def summary_stats(df_dataset: pd.DataFrame) -> pd.DataFrame:
    stats_summary = {
        "Metric": [
            "Mean (Average Images per Class)",
            "Median (Middle Value)",
            "Variance",
            "Standard Deviation",
            "Total Images",
        ],
        "Value": [
            df_dataset["Image_Count"].mean(),
            df_dataset["Image_Count"].median(),
            df_dataset["Image_Count"].var(),
            df_dataset["Image_Count"].std(),
            df_dataset["Image_Count"].sum(),
        ],
    }
    return pd.DataFrame(stats_summary)


def main():
    label_dir = WILDLIFE_DATASET_INPUT / "train" / "labels"

    image_counts = count_images_per_class(label_dir, RAW_CLASS_NAMES)
    df_dataset = build_dataset_dataframe(image_counts)

    print(" --- STEP 1: DATASET CLASS OVERVIEW ---")
    print(df_dataset.head(43))
    print(f"\nTotal Dataset Rows: {len(df_dataset)}")
    df_dataset.info()

    df_stats = summary_stats(df_dataset)
    print("\n --- SUMMARY STATISTICS ---")
    print(df_stats.to_string(index=False))

    try:
        plot_overview(df_dataset)
        plot_cleaned(df_dataset)
    except Exception as exc:  # pragma: no cover - plotting is optional/interactive
        print(f"(Plotting skipped: {exc})")


if __name__ == "__main__":
    main()
