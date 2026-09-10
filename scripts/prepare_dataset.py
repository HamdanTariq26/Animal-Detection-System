# -*- coding: utf-8 -*-
"""
prepare_dataset.py

Dataset preparation only -- no model training happens here.

Two functions :

1. build_clean_dataset()
   Removes the Butterfly class (original class id = BUTTERFLY_ID) from the
   raw wildlife dataset and re-indexes every class id greater than
   BUTTERFLY_ID down by one, producing a 53-class dataset + YOLO data.yaml.
   This is the dataset the Precision Expert (RT-DETR-L) trains on.

2. build_blurry_dataset()
   Applies the  Albumentations degradation pipeline (motion blur, gaussian noise, JPEG compression, resize) to the
   *clean* dataset's train/valid splits, producing the dataset the Kinetic
   Expert (YOLO11m) trains on.

Run:
    python scripts/prepare_dataset.py --clean --blurry
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    WILDLIFE_DATASET_INPUT,
    CLEAN_DATASET_DIR,
    CLEAN_DATA_YAML,
    BLURRY_DATASET_DIR,
    BLURRY_DATA_YAML,
    BUTTERFLY_ID,
    CLASS_NAMES,
)

SPLITS = ["train", "valid", "test"]


def build_clean_dataset():
    """Remove the Butterfly class and re-index remaining classes.
    Produces CLEAN_DATASET_DIR/{split}/{images,labels} and CLEAN_DATA_YAML.
    """
    print(" Starting Dataset Preparation (Removing Butterflies)...")

    for split in SPLITS:
        print(f"Processing {split}...")

        src_img_dir = WILDLIFE_DATASET_INPUT / split / "images"
        src_lbl_dir = WILDLIFE_DATASET_INPUT / split / "labels"
        dst_img_dir = CLEAN_DATASET_DIR / split / "images"
        dst_lbl_dir = CLEAN_DATASET_DIR / split / "labels"

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_lbl_dir.exists():
            print(f"  (skipping {split}: {src_lbl_dir} not found)")
            continue

        label_files = [f for f in os.listdir(src_lbl_dir) if f.endswith(".txt")]

        for lbl_file in tqdm(label_files):
            with open(os.path.join(src_lbl_dir, lbl_file), "r") as f:
                lines = f.readlines()

            cleaned_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue

                cid = int(parts[0])
                if cid == BUTTERFLY_ID:
                    continue

                if cid > BUTTERFLY_ID:
                    parts[0] = str(cid - 1)

                cleaned_lines.append(" ".join(parts) + "\n")

            if len(cleaned_lines) > 0:
                with open(os.path.join(dst_lbl_dir, lbl_file), "w") as f:
                    f.writelines(cleaned_lines)

                img_name_base = os.path.splitext(lbl_file)[0]
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_file = img_name_base + ext
                    src_img_path = os.path.join(src_img_dir, img_file)
                    if os.path.exists(src_img_path):
                        shutil.copy(src_img_path, os.path.join(dst_img_dir, img_file))
                        break

    data_yaml = {
        "path": str(CLEAN_DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    CLEAN_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_DATA_YAML, "w") as f:
        yaml.dump(data_yaml, f)

    print(" DONE! Butterfly-only images were removed.")
    print(f"Final Class Count: {len(CLASS_NAMES)}")
    print(f"Clean dataset YAML: {CLEAN_DATA_YAML}")


def build_blurry_dataset():
    """Apply the degradation pipeline to the clean dataset's train/valid
    splits, producing the dataset used to train the Kinetic Expert.
    """
    import albumentations as A
    import cv2
    from glob import glob

    for split in ["train", "valid"]:
        input_img_dir = CLEAN_DATASET_DIR / split / "images"
        input_lbl_dir = CLEAN_DATASET_DIR / split / "labels"
        output_img_dir = BLURRY_DATASET_DIR / split / "images"
        output_lbl_dir = BLURRY_DATASET_DIR / split / "labels"

        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_dir.mkdir(parents=True, exist_ok=True)

        aug = A.Compose(
            [
                A.MotionBlur(blur_limit=7, p=0.4),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
                A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
                A.Resize(640, 640, p=1.0),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        image_paths = glob(os.path.join(input_img_dir, "*.jpg"))
        print(f"🚀 Blurring {split} split: {len(image_paths)} images...")

        for img_path in tqdm(image_paths):
            basename = os.path.basename(img_path)
            label_path = os.path.join(input_lbl_dir, basename.replace(".jpg", ".txt"))

            if not os.path.exists(label_path):
                continue

            bboxes = []
            class_labels = []
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:]])

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            try:
                transformed = aug(image=image, bboxes=bboxes, class_labels=class_labels)

                cv2.imwrite(
                    os.path.join(output_img_dir, basename),
                    cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR),
                )

                with open(
                    os.path.join(output_lbl_dir, basename.replace(".jpg", ".txt")), "w"
                ) as f:
                    for i in range(len(transformed["bboxes"])):
                        b = transformed["bboxes"][i]
                        l = transformed["class_labels"][i]
                        f.write(f"{l} {' '.join([f'{x:.6f}' for x in b])}\n")
            except Exception:
                continue

    data_yaml = {
        "path": str(BLURRY_DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    BLURRY_DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(BLURRY_DATA_YAML, "w") as f:
        yaml.dump(data_yaml, f)

    print(f" DONE! Blurry Expert dataset ready at {BLURRY_DATASET_DIR}")
    print(f"Blurry dataset YAML: {BLURRY_DATA_YAML}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Wildlife-MoE datasets.")
    parser.add_argument("--clean", action="store_true", help="Build the clean (Butterfly-removed) dataset.")
    parser.add_argument("--blurry", action="store_true", help="Build the degraded/blurry dataset.")
    args = parser.parse_args()

    if not args.clean and not args.blurry:
        # Default: run both, matching the original notebook's full flow.
        args.clean = True
        args.blurry = True

    if args.clean:
        build_clean_dataset()
    if args.blurry:
        build_blurry_dataset()


if __name__ == "__main__":
    main()
