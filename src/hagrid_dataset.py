import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class HagridBBoxImageFolder(Dataset):
    """
    Dataset that crops images using HaGRID bounding boxes before applying transforms.
    """

    def __init__(self, root, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        self.annotations = self._load_annotations()

        # Discover classes from subfolders
        class_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subfolders found under '{self.root}'")

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Build samples: (image_path, label_idx, bbox_normalized)
        self.samples = []
        missing_count = 0

        for cls_name in self.classes:
            class_dir = self.root / cls_name
            label_idx = self.class_to_idx[cls_name]

            for img_path in sorted(class_dir.iterdir()):
                if (
                    not img_path.is_file()
                    or img_path.suffix.lower() not in config.img_extensions_torch
                ):
                    continue

                key = img_path.stem  # image ID without extension
                ann = self.annotations.get(key)

                if ann is None:
                    missing_count += 1
                    continue

                bbox = self._get_bbox(ann)
                self.samples.append((str(img_path), label_idx, bbox))

        if not self.samples:
            raise RuntimeError(
                f"No samples found. Check that images and annotations match. "
                f"Missing annotations for {missing_count} images."
            )

        if missing_count > 0:
            print(f"[WARN] {missing_count} images missing annotations (skipped)")

        print(f"[INFO] Loaded {len(self.samples)} samples from {self.root}")

    def _load_annotations(self):
        """Load all annotation JSON files from annotations_path directory."""
        annotations_path = Path(config.annotations_path)
        if not annotations_path.exists():
            raise FileNotFoundError(
                f"Annotations directory not found: {annotations_path}"
            )

        annotations = {}
        json_files = list(annotations_path.rglob("*.json"))

        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {annotations_path}")

        for json_file in json_files:
            with open(json_file, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Failed to parse {json_file}: {e}")
                    continue

                for key, value in data.items():
                    # Keep first occurrence if duplicate (shouldn't happen)
                    if key not in annotations:
                        annotations[key] = value

        print(
            f"[INFO] Loaded {len(annotations)} annotations from {len(json_files)} files"
        )
        return annotations

    @staticmethod
    def _get_bbox(ann):
        """
        Extract bounding box from annotation.
        """
        # Try united_bbox first (for multi-hand gestures)
        united = ann.get("united_bbox")
        if (
            united
            and isinstance(united, list)
            and len(united) > 0
            and united[0] is not None
        ):
            bbox = united[0]
        else:
            # Fall back to first bbox
            bboxes = ann.get("bboxes")
            if not bboxes or len(bboxes) == 0:
                raise ValueError(f"Annotation has no valid bbox: {ann.keys()}")
            bbox = bboxes[0]

        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}")

        return bbox

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, bbox_norm = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Convert normalized bbox [x, y, w, h] to pixel coordinates
        x_norm, y_norm, w_norm, h_norm = bbox_norm
        x1 = int(x_norm * w)
        y1 = int(y_norm * h)
        x2 = int((x_norm + w_norm) * w)
        y2 = int((y_norm + h_norm) * h)

        # Add 10% margin
        margin = 0.1
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        dx = int(box_w * margin)
        dy = int(box_h * margin)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        # Safety check
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, w, h

        img = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img = self.transform(img)

        return img, label_idx
