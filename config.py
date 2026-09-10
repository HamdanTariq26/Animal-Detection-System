# -*- coding: utf-8 -*-
"""
config.py

Central path configuration for the Wildlife-MoE project.

"""

import os
from pathlib import Path

# Root of the repository (this file lives at the repo root).
PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

# Original raw dataset (as provided on Kaggle).
WILDLIFE_DATASET_INPUT = Path(
    os.environ.get(
        "WILDLIFE_DATASET_INPUT",
        "/kaggle/input/datasets/banuprasadb/wildlife-dataset",
    )
)

# Working directory where generated/cleaned datasets are written.
WILDLIFE_DATASET_WORK = Path(
    os.environ.get("WILDLIFE_DATASET_WORK", "/kaggle/working")
)

# Cleaned ("clear") dataset used to train the Precision Expert (RT-DETR-L).
CLEAN_DATASET_DIR = WILDLIFE_DATASET_WORK / "wildlife-dataset"
CLEAN_DATA_YAML = WILDLIFE_DATASET_WORK / "data.yaml"

BLURRY_DATASET_DIR = Path(
    os.environ.get(
        "WILDLIFE_BLURRY_DATASET_DIR",
        str(WILDLIFE_DATASET_WORK / "wildlife-dataset-blurry"),
    )
)
BLURRY_DATA_YAML = Path(
    os.environ.get(
        "WILDLIFE_DATA_YAML",
        str(WILDLIFE_DATASET_WORK / "blurry_expert.yaml"),
    )
)

# ---------------------------------------------------------------------------
# Model checkpoint paths (stable, project-relative locations)
# ---------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"

PRECISION_MODEL_PATH = Path(
    os.environ.get(
        "PRECISION_MODEL_PATH",
        str(MODELS_DIR / "precision_expert" / "best.pt"),
    )
)

KINETIC_MODEL_PATH = Path(
    os.environ.get(
        "KINETIC_MODEL_PATH",
        str(MODELS_DIR / "kinetic_expert" / "best.pt"),
    )
)

# ---------------------------------------------------------------------------
# Training run output (Ultralytics `project`/`name` args)
# ---------------------------------------------------------------------------

RUNS_PROJECT_DIR = Path(os.environ.get("WILDLIFE_RUNS_DIR", str(PROJECT_ROOT / "runs" / "wildlife_moe")))

PRECISION_RUN_NAME = "Expert_Clear_RTDETR_v2"
KINETIC_RUN_NAME = "Expert_Blurry_YOLO11m_v3_Augmented"

# ---------------------------------------------------------------------------
# Fixed system constants (DO NOT change without changing the model design)
# ---------------------------------------------------------------------------

BUTTERFLY_ID = 6  # class index removed from the original 54-class dataset

MOE_THRESHOLD_DEFAULT = 120  # Laplacian-variance routing threshold
INFERENCE_CONFIDENCE = 0.25
TRACKER_CONFIG = "botsort.yaml"

CLASS_NAMES = [
    'Zebra', 'Lion', 'Leopard', 'Cheetah', 'Tiger', 'Bear',
    'Canary', 'Crocodile', 'Bull', 'Camel', 'Centipede', 'Caterpillar',
    'Duck', 'Squirrel', 'Spider', 'Ladybug', 'Elephant', 'Horse', 'Fox',
    'Tortoise', 'Frog', 'Kangaroo', 'Deer', 'Eagle', 'Monkey', 'Snake',
    'Owl', 'Swan', 'Goat', 'Rabbit', 'Giraffe', 'Goose', 'PolarBear',
    'Raven', 'Hippopotamus', 'BrownBear', 'Rhinoceros', 'Woodpecker',
    'Sheep', 'Magpie', 'Ostrich', 'Jaguar', 'Hedgehog', 'Turkey',
    'Raccoon', 'Worm', 'Harbor', 'Panda', 'RedPanda', 'Otter', 'Lynx',
    'Scorpion', 'Koala'
]  # 53 classes (Butterfly already removed / re-indexed)

# Original 54-class list, pre-Butterfly-removal, used only by the
# dataset-analysis script to read the *raw* label files.
RAW_CLASS_NAMES = [
    'Zebra', 'Lion', 'Leopard', 'Cheetah', 'Tiger', 'Bear',
    'Butterfly', 'Canary', 'Crocodile', 'Bull', 'Camel', 'Centipede',
    'Caterpillar', 'Duck', 'Squirrel', 'Spider', 'Ladybug', 'Elephant',
    'Horse', 'Fox', 'Tortoise', 'Frog', 'Kangaroo', 'Deer', 'Eagle',
    'Monkey', 'Snake', 'Owl', 'Swan', 'Goat', 'Rabbit', 'Giraffe',
    'Goose', 'PolarBear', 'Raven', 'Hippopotamus', 'BrownBear',
    'Rhinoceros', 'Woodpecker', 'Sheep', 'Magpie', 'Ostrich', 'Jaguar',
    'Hedgehog', 'Turkey', 'Raccoon', 'Worm', 'Harbor', 'Panda',
    'RedPanda', 'Otter', 'Lynx', 'Scorpion', 'Koala'
]
