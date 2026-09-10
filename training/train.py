# -*- coding: utf-8 -*-
"""
train.py

Trains both experts and promotes their best checkpoints into stable,
project-relative locations:

    models/precision_expert/best.pt
    models/kinetic_expert/best.pt

Run:
    python training/train.py --precision --kinetic
    python training/train.py --precision          # RT-DETR-L only
    python training/train.py --kinetic             # YOLO11m only
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CLEAN_DATA_YAML,
    BLURRY_DATA_YAML,
    RUNS_PROJECT_DIR,
    PRECISION_RUN_NAME,
    KINETIC_RUN_NAME,
    PRECISION_MODEL_PATH,
    KINETIC_MODEL_PATH,
)


def promote_checkpoint(model, destination: Path):
    """Locate the Ultralytics run's best.pt from the trained model object
    itself (not a hard-coded path) and copy it into a stable model dir.
    """
    run_dir = Path(model.trainer.save_dir)
    best_checkpoint = run_dir / "weights" / "best.pt"

    if not best_checkpoint.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found at {best_checkpoint}. "
            "Training may not have completed successfully."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_checkpoint, destination)
    print(f" Promoted checkpoint: {best_checkpoint} -> {destination}")
    return destination


def train_precision_expert():
    """RT-DETR-L trained on the clean (Butterfly-removed) dataset."""
    from ultralytics import RTDETR

    model_clear = RTDETR("rtdetr-l.pt")

    model_clear.train(
        data=str(CLEAN_DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        device=[0, 1],
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        amp=True,
        workers=8,
        patience=15,
        project=str(RUNS_PROJECT_DIR),
        name=PRECISION_RUN_NAME,
        plots=False,
        save=True,
        verbose=True,
        deterministic=False,
    )

    promote_checkpoint(model_clear, PRECISION_MODEL_PATH)
    return model_clear


def train_kinetic_expert():
    """YOLO11m trained on the degraded/blurry dataset."""
    from ultralytics import YOLO

    model_blur = YOLO("yolo11m.pt")

    model_blur.train(
        data=str(BLURRY_DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=32,
        device=[0, 1],
        mosaic=0.5,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        amp=True,
        project=str(RUNS_PROJECT_DIR),
        name=KINETIC_RUN_NAME,
    )

    promote_checkpoint(model_blur, KINETIC_MODEL_PATH)
    return model_blur


def main():
    parser = argparse.ArgumentParser(description="Train Wildlife-MoE experts.")
    parser.add_argument("--precision", action="store_true", help="Train the RT-DETR-L Precision Expert.")
    parser.add_argument("--kinetic", action="store_true", help="Train the YOLO11m Kinetic Expert.")
    args = parser.parse_args()

    if not args.precision and not args.kinetic:
        args.precision = True
        args.kinetic = True

    if args.precision:
        train_precision_expert()
    if args.kinetic:
        train_kinetic_expert()


if __name__ == "__main__":
    main()
