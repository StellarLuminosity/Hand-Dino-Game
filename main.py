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

    model = HandGestureCNN(num_classes=num_classes).to(device) # re-initialize model so that we evaluate the best saved one
    ckpt_path = os.path.join(config.output_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # -----------------------------
    # Final evaluation: CNN model
    # -----------------------------
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

    print("\n=== CNN evaluation on: data/test ===")
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.3f}")
    print("Per-class accuracy:")
    for cls_name, acc in per_class_acc.items():
        print(f"  {cls_name:>5}: {acc:.3f}")
    print(f"CNN metrics saved to: {cnn_metrics_path}")

    # -----------------------------------
    # Final evaluation: OpenCV baseline
    # -----------------------------------
    split_root = Path(config.data_dir) / "test"
    overall_acc, per_class = eval_split(split_root)

    baseline_metrics_path = Path(config.metrics_output)
    baseline_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    save_metrics_json(overall_acc, per_class, baseline_metrics_path)

    print("\n=== OpenCV baseline evaluation on: data/test ===")
    print(f"Overall accuracy: {overall_acc:.3f}")
    for cls_name, stats in per_class.items():
        correct = stats["correct"]
        total = stats["total"]
        acc = correct / total if total > 0 else 0.0
        print(f"  {cls_name:>5}: {acc:.3f} ({correct}/{total})")
    print(f"Baseline metrics saved to: {baseline_metrics_path}")


if __name__ == "__main__":
    main()
