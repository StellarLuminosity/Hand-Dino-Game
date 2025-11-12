import os
import random
import shutil
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

# Configuration
DATASET_NAME = "innominate817/hagrid-sample-30k-384p"
TARGET_CLASSES = ["palm", "peace", "fist"]
SPLIT_RATIOS = {"train": 0.7, "test": 0.15, "val": 0.15}
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
DOWNLOAD_DIR = RAW_DATA_DIR / "hagrid-sample-30k-384p"


def download_dataset():
    """Download the HaGRID dataset from Kaggle."""
    print("Downloading dataset from Kaggle...")

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    api.dataset_download_files(DATASET_NAME, path=str(RAW_DATA_DIR), unzip=True)

    print(f"Dataset downloaded to {DOWNLOAD_DIR}")


def filter_and_split():
    """Filter dataset to target classes and split into train/test/val."""
    print("Filtering and splitting dataset...")

    images_root = DOWNLOAD_DIR / "hagrid_30k"
    if not images_root.exists():
        raise FileNotFoundError(f"Expected images under {images_root}")

    # Match folders like train_val_palm, train_val_peace, train_val_fist
    class_dirs = []
    for cname in TARGET_CLASSES:
        d = images_root / f"train_val_{cname}"
        if d.exists() and d.is_dir():
            class_dirs.append(d)
        else:
            print(f"Warning: {d} not found")

    if not class_dirs:
        raise ValueError(
            f"None of the target classes {TARGET_CLASSES} were found under {images_root}"
        )

    # Create output directories
    for split in SPLIT_RATIOS.keys():
        for cname in TARGET_CLASSES:
            (DATA_DIR / split / cname).mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}

    def list_images(p: Path):
        files = []
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
        return files

    # Process each class
    rng = random.Random(42)  # reproducible
    for class_dir in class_dirs:
        cname = class_dir.name.replace("train_val_", "")
        print(f"\nProcessing class: {cname}")

        image_files = list_images(class_dir)
        if not image_files:
            print(f"  Warning: no images in {class_dir}")
            continue

        rng.shuffle(image_files)
        total = len(image_files)
        train_end = int(total * SPLIT_RATIOS["train"])
        test_end = train_end + int(total * SPLIT_RATIOS["test"])

        splits = {
            "train": image_files[:train_end],
            "test": image_files[train_end:test_end],
            "val": image_files[test_end:],
        }

        for split_name, files in splits.items():
            dest_dir = DATA_DIR / split_name / cname
            for src in files:
                shutil.copy2(src, dest_dir / src.name)
            print(f"  {split_name}: {len(files)} images")

    print("\nDataset preprocessing complete!")
    print(f"Data organized in: {DATA_DIR}")
    for split in SPLIT_RATIOS.keys():
        print(f"  {DATA_DIR / split}/")
        for cname in TARGET_CLASSES:
            count = len(list((DATA_DIR / split / cname).glob("*")))
            print(f"    {cname}/ ({count} images)")


def cleanup_raw_data():
    """Optionally remove raw downloaded data to save space."""
    response = input("\nRemove raw downloaded data? (y/n): ").strip().lower()
    if response == "y":
        shutil.rmtree(RAW_DATA_DIR)
        print("Raw data removed.")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Check if dataset already downloaded
    if not DOWNLOAD_DIR.exists():
        download_dataset()
    else:
        print(f"Dataset already exists at {DOWNLOAD_DIR}")
        response = input("Re-download? (y/n): ").strip().lower()
        if response == "y":
            shutil.rmtree(RAW_DATA_DIR)
            download_dataset()

    # Filter and split
    filter_and_split()

    # Optional cleanup
    cleanup_raw_data()
