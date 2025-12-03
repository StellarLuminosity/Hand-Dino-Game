#!/usr/bin/env python3
# main.py

import json
import os
import torch
import random
from pathlib import Path

import numpy as np

import config
from src.baseline_eval_on_images import eval_split, save_metrics_json
from src.model import HandGestureCNN
from src.preprocess_dataset import preprocess_dataset
from src.train import evaluate, get_dataloaders, train


def main():
    random.seed(config.random_seed)

    if not (Path(config.data_dir) / "train").exists():
        preprocess_dataset()

    train_loader, val_loader, test_loader = get_dataloaders()

    device = config.device
    num_classes = len(train_loader.dataset.classes)
    model = HandGestureCNN(num_classes=num_classes).to(device)

    train(
        model,
        train_loader,
        val_loader,
        device,
        eval_every=1,
        max_val_batches=None,
    )

    print("Final evaluation...")
    model = HandGestureCNN(num_classes=num_classes).to(device) # re-initialize model so that we can evaluate the best one
    ckpt_path = os.path.join(config.output_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, per_class_acc, confusion = evaluate(
        model, test_loader, device, max_batches=None
    )

    cnn_metrics = {
        "split": "test",
        "loss": test_loss,
        "accuracy": test_acc,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": confusion.tolist(),
        "idx_to_class": {
            idx: cls for cls, idx in test_loader.dataset.class_to_idx.items()
        },
    }

    cnn_metrics_path = Path(config.cnn_metrics_output)
    cnn_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cnn_metrics_path.write_text(json.dumps(cnn_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved CNN metrics to {cnn_metrics_path}")

    # Baseline evaluation
    split_root = Path(config.data_dir) / "test"
    overall_acc, per_class = eval_split(split_root)
    save_metrics_json(overall_acc, per_class, Path(config.metrics_output))


if __name__ == "__main__":
    main()
