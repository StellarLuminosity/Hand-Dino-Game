# src/dataset.py

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class HagridBBoxImageFolder(Dataset):
    """
    Custom dataset that crops images using HaGRID bounding boxes before applying transforms.

    Expected directory structure:
      root/
        class_0/
          img_0.jpg
          img_1.jpg
          ...
        class_1/
          ...

    annotations_path can be:
      - A single JSON file with all annotations, or
      - A directory containing multiple JSON files (e.g. 'hagrid_annotations/').
        In that case, all '*.json' files are loaded recursively and merged.

    HaGRID annotations format (per GitHub):
      {
        "image_id_without_ext": {
          "bboxes": [[x, y, w, h], ...],          # normalized [0,1]
          "labels": [...],
          "united_bbox": [[x, y, w, h]] or null,
          "united_label": [...],
          ...
        },
        ...
      }
    """

    def __init__(self, root, annotations_path, transform=None):
        super().__init__()
        self.root = Path(root)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root '{self.root}' does not exist")

        self.annotations = self._load_annotations(annotations_path)

        # Discover classes (subfolders)
        class_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        if not class_dirs:
            raise RuntimeError(f"No class subfolders found under '{self.root}'")

        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Build list of samples: (path, label_idx, bbox_norm)
        self.samples = []
        missing_ann = []

        for cls_name in self.classes:
            class_dir = self.root / cls_name
            label_idx = self.class_to_idx[cls_name]

            for img_path in sorted(class_dir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in IMG_EXTENSIONS:
                    continue

                key = img_path.stem  # image name without extension
                ann = self.annotations.get(key)

                if ann is None:
                    # Keep track of missing ones (useful debug)
                    missing_ann.append(str(img_path))
                    continue

                bbox_norm = self._choose_bbox(ann)
                self.samples.append((str(img_path), label_idx, bbox_norm))

        if not self.samples:
            raise RuntimeError(
                f"No samples built. Either there are no images under '{self.root}' "
                f"or annotations did not match image filenames. "
                f"Check that you pointed --annotations_path to HaGRID annotations "
                f"and that filenames were preserved when preprocessing."
            )

        if missing_ann:
            # Only warn; if you want hard fail, raise instead.
            print(
                f"[WARN] {len(missing_ann)} images under '{self.root}' "
                f"do not have bounding-box annotations. They were skipped."
            )

        print(
            f"[INFO] HagridBBoxImageFolder at '{self.root}': "
            f"{len(self.samples)} samples across {len(self.classes)} classes."
        )

    @staticmethod
    def _load_annotations(annotations_path):
        annotations_path = Path(annotations_path)
        annotations = {}

        if annotations_path.is_file():
            # Single JSON file
            with open(annotations_path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                annotations[str(k)] = v
            print(
                f"[INFO] Loaded {len(annotations)} annotations from '{annotations_path}'"
            )
        elif annotations_path.is_dir():
            # Recursively load all JSON files under this directory
            json_files = list(annotations_path.rglob("*.json"))
            if not json_files:
                raise FileNotFoundError(
                    f"No JSON files found under annotations directory '{annotations_path}'"
                )

            for jf in json_files:
                with open(jf, "r") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"[WARN] Failed to parse '{jf}': {e}")
                        continue
                for k, v in data.items():
                    key = str(k)
                    # If the same key appears multiple times (train/val/test),
                    # keep the first occurrence – they should be identical.
                    if key not in annotations:
                        annotations[key] = v

            print(
                f"[INFO] Loaded {len(annotations)} annotations "
                f"from directory '{annotations_path}' ({len(json_files)} JSON files)."
            )
        else:
            raise FileNotFoundError(
                f"annotations_path '{annotations_path}' is neither a file nor a directory"
            )

        return annotations

    @staticmethod
    def _choose_bbox(ann):
        """
        Choose which bounding box to use from a HaGRID annotation entry.

        Prefer united_bbox for two-handed gestures (if present and non-null),
        otherwise fall back to the first bbox in 'bboxes'.
        """
        united = ann.get("united_bbox")
        if isinstance(united, list) and len(united) > 0 and united[0] is not None:
            bbox = united[0]
        else:
            bboxes = ann.get("bboxes")
            if not bboxes:
                raise ValueError("Annotation entry has no 'bboxes' or 'united_bbox'")
            bbox = bboxes[0]

        if len(bbox) != 4:
            raise ValueError(f"Expected bbox with 4 elements, got: {bbox}")

        return bbox  # normalized [x, y, w, h] in [0,1]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, bbox_norm = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        x, y, bw, bh = bbox_norm
        # Convert normalized coords to pixel coordinates
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + bw) * w)
        y2 = int((y + bh) * h)

        # Add a small margin around the bbox
        margin = 0.1
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        dx = int(box_w * margin)
        dy = int(box_h * margin)

        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)

        # Safety check in case something went wrong
        if x2 <= x1 or y2 <= y1:
            # Fallback to full image if bbox is degenerate
            x1, y1, x2, y2 = 0, 0, w, h

        img = img.crop((x1, y1, x2, y2))

        if self.transform is not None:
            img = self.transform(img)

        return img, label_idx
