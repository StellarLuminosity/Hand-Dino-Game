#!/usr/bin/env python3
# main.py

import random
from pathlib import Path

import cv2
import numpy as np

import config
from src.baseline_eval_on_images import eval_split, save_metrics_json
from src.preprocess_dataset import preprocess_dataset
from src.train import train_model


def main():

    random.seed(config.random_seed)
    preprocess_dataset()

    train_model()

    split_root = Path(config.data_dir) / "val"
    if not split_root.exists():
        raise FileNotFoundError(
            f"Split path {split_root} does not exist. Run preprocess() first."
        )

    overall_acc, per_class = eval_split(split_root)
    save_metrics_json(overall_acc, per_class, Path(config.metrics_output))


# ========== RUN ==========

if __name__ == "__main__":
    main()
