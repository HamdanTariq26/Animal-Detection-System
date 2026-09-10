# -*- coding: utf-8 -*-
"""
evaluate.py

Evaluates both trained experts against the clean test split and produces
the same accuracy comparison table and confusion-matrix visualization as
the original notebook.

Loads checkpoints from the stable project locations:
    models/precision_expert/best.pt
    models/kinetic_expert/best.pt


Run:
    python training/evaluate.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DATA_YAML, PRECISION_MODEL_PATH, KINETIC_MODEL_PATH  # noqa: E402


def get_f1(p, r):
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0


def main():
    from ultralytics import YOLO, RTDETR

    for path, label in [(PRECISION_MODEL_PATH, "Precision"), (KINETIC_MODEL_PATH, "Kinetic")]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} Expert checkpoint not found at {path}. "
                "Run training/train.py first."
            )

    model_precision = RTDETR(str(PRECISION_MODEL_PATH))
    model_kinetic = YOLO(str(KINETIC_MODEL_PATH))

    print(" Evaluating Both Experts...")

    res_rt = model_precision.val(data=str(CLEAN_DATA_YAML), split="test")
    res_yolo = model_kinetic.val(data=str(CLEAN_DATA_YAML), split="test")

    comparison_data = {
        "Metric": ["Accuracy (mAP50)", "Precision", "Recall", "F1-Score"],
        "RT-DETR (Precision Expert)": [
            res_rt.results_dict["metrics/mAP50(B)"],
            res_rt.results_dict["metrics/precision(B)"],
            res_rt.results_dict["metrics/recall(B)"],
            get_f1(
                res_rt.results_dict["metrics/precision(B)"],
                res_rt.results_dict["metrics/recall(B)"],
            ),
        ],
        "YOLO11m (Kinetic Expert)": [
            res_yolo.results_dict["metrics/mAP50(B)"],
            res_yolo.results_dict["metrics/precision(B)"],
            res_yolo.results_dict["metrics/recall(B)"],
            get_f1(
                res_yolo.results_dict["metrics/precision(B)"],
                res_yolo.results_dict["metrics/recall(B)"],
            ),
        ],
    }

    df_comp = pd.DataFrame(comparison_data)
    cols_to_format = ["RT-DETR (Precision Expert)", "YOLO11m (Kinetic Expert)"]
    for col in cols_to_format:
        df_comp[col] = df_comp[col].apply(lambda x: f"{x * 100:.2f}%")

    print("\n --- FINAL ACCURACY COMPARISON ---")
    print(df_comp.to_string(index=False))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        plt.imshow(plt.imread(os.path.join(res_rt.save_dir, "confusion_matrix.png")))
        plt.title("RT-DETR: Precision Expert (Accuracy Map)", fontsize=15)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(plt.imread(os.path.join(res_yolo.save_dir, "confusion_matrix.png")))
        plt.title("YOLO11m: Kinetic Expert (Accuracy Map)", fontsize=15)
        plt.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception as exc:  # pragma: no cover - plotting is optional/interactive
        print(f"(Confusion matrix plotting skipped: {exc})")

    return df_comp


if __name__ == "__main__":
    main()
