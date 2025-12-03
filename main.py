#!/usr/bin/env python3
# main.py

import random
from pathlib import Path

import cv2
import numpy as np

from src.baseline_eval_on_images import eval_split, save_metrics_json
from src.preprocess_dataset import preprocess_dataset
from src.train import train_model

# ========== CONFIGURATION ==========

# Preprocessing config
DATASET_NAME = "innominate817/hagrid-sample-30k-384p"
TARGET_CLASSES = ["palm", "peace", "fist"]
DATA_DIR = "data"

# Training config
ANNOTATIONS_PATH = "./hagrid_annotations"  # or path to JSON file
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 4
OUTPUT_DIR = "checkpoints"
DEVICE = "cuda"

# Evaluation config
EVAL_SPLIT = "val"
METRICS_OUTPUT = "figs/baseline_metrics.json"

random.seed(42)


def main():

    preprocess_dataset(
        dataset_name=DATASET_NAME,
        target_classes=TARGET_CLASSES,
        data_dir=DATA_DIR,
        force_redownload=False,
        cleanup_raw=False,
    )

    train_model(
        data_root=DATA_DIR,
        annotations_path=ANNOTATIONS_PATH,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
    )

    split_root = Path(DATA_DIR) / EVAL_SPLIT
    if not split_root.exists():
        raise FileNotFoundError(
            f"Split path {split_root} does not exist. Run preprocess() first."
        )

    overall_acc, per_class = eval_split(split_root)
    save_metrics_json(overall_acc, per_class, Path(METRICS_OUTPUT))


# ========== RUN ==========

if __name__ == "__main__":
    main()
