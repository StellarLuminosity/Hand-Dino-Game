import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import config

CLASSES = config.target_classes


def predict_one(img_bgr):
    """
    Very simple OpenCV baseline:
    - convert to HSV
    - threshold for skin
    - largest contour -> convex hull -> convexity defects
    - use # of defects as proxy for finger count
    """

    if img_bgr is None:
        return "peace"  # safe fallback

    # 1) Preprocess
    img = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # crude skin color range
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 170, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # clean up mask
    mask = cv2.medianBlur(mask, 7)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 2) Largest contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # no hand found → call it neutral
        return "peace"

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 500:  # very small blob → probably noise
        return "peace"

    # 3) Convex hull + defects
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return "fist"  # degenerate hull --> treat as closed

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        num_defects = 0
    else:
        num_defects = defects.shape[0]

    # more defects --> more spread fingers (palm)
    if num_defects >= 3:
        return "palm"  # open hand
    elif num_defects <= 1:
        return "fist"  # closed hand
    else:
        return "peace"  # in-between → peace / neutral


def eval_split(split_root: Path):
    """
    Evaluate baseline on a folder like data/val with subfolders:
      data/val/palm, data/val/fist, data/val/peace
    """
    total = 0
    total_correct = 0

    per_class = {cname: {"correct": 0, "total": 0} for cname in CLASSES}

    for cname in CLASSES:
        class_dir = split_root / cname
        if not class_dir.exists():
            print(f"[WARN] Missing folder: {class_dir} (skipping)")
            continue

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in config.img_extensions:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] Cannot read {img_path}, skipping")
                continue

            pred = predict_one(img)
            total += 1
            per_class[cname]["total"] += 1

            if pred == cname:
                total_correct += 1
                per_class[cname]["correct"] += 1

    if total == 0:
        print("No images found – check your data path / split.")
        return 0.0, per_class

    overall_acc = total_correct / total

    print(f"\n=== Baseline evaluation on: {split_root} ===")
    print(f"Overall accuracy: {overall_acc:.3f} ({total_correct}/{total})")

    for cname, stats in per_class.items():
        if stats["total"] == 0:
            print(f"{cname:>6}: no samples")
        else:
            acc = stats["correct"] / stats["total"]
            print(f"{cname:>6}: {acc:.3f} " f"({stats['correct']}/{stats['total']})")

    return overall_acc, per_class


def save_metrics_json(overall_acc, per_class, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall_accuracy": overall_acc,
        "per_class": per_class,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Root folder with train/val/test subfolders",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="figs/baseline_metrics.json",
        help="Where to save metrics JSON",
    )
    args = parser.parse_args()

    split_root = Path(args.data_root) / args.split
    if not split_root.exists():
        raise FileNotFoundError(
            f"Split path {split_root} does not exist. "
            "Did you run preprocess_dataset.py?"
        )

    overall_acc, per_class = eval_split(split_root)
    save_metrics_json(overall_acc, per_class, Path(args.out_json))


if __name__ == "__main__":
    main()
