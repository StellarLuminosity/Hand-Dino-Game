import os
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from kaggle.api.kaggle_api_extended import KaggleApi


def preprocess_dataset(
    dataset_name: str = "innominate817/hagrid-sample-30k-384p",
    data_dir: str = "data",
    cleanup_raw: bool = False,
):
    """
    Download and preprocess the HaGRID dataset from Kaggle.
    """
    target_classes = ["palm", "peace", "fist"]
    split_ratios = {"train": 0.7, "test": 0.15, "val": 0.15}

    data_dir = Path(data_dir)
    raw_data_dir = data_dir / "raw"
    download_dir = raw_data_dir / "hagrid-sample-30k-384p"
    annotations_dir = data_dir / "hagrid_annotations"

    # Download annotations (only if not already downloaded)
    if not annotations_dir.exists():
        print("Downloading HaGRID annotations...")
        annotations_url = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
        annotations_zip = raw_data_dir / "annotations.zip"

        raw_data_dir.mkdir(parents=True, exist_ok=True)

        try:
            urlretrieve(annotations_url, annotations_zip)
            print(f"Annotations downloaded to {annotations_zip}")

            # Extract annotations
            with zipfile.ZipFile(annotations_zip, "r") as zip_ref:
                zip_ref.extractall(raw_data_dir)

            # Move extracted annotations to final location
            # The zip contains an "annotations" folder
            extracted_path = raw_data_dir / "annotations"
            if extracted_path.exists():
                shutil.move(str(extracted_path), str(annotations_dir))
                print(f"Annotations extracted to {annotations_dir}")
            else:
                print(f"[WARN] Expected 'annotations' folder after extraction")

            # Clean up zip file
            annotations_zip.unlink()

        except Exception as e:
            print(f"[ERROR] Failed to download annotations: {e}")
            print(
                "Please download manually from the URL above and place in data/hagrid_annotations/"
            )
    else:
        print(f"Annotations already exist at {annotations_dir}")

    # Download dataset (only if not already downloaded)
    if not download_dir.exists():
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()

        raw_data_dir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_name, path=str(raw_data_dir), unzip=True)
        print(f"Dataset downloaded to {download_dir}")
    else:
        print(f"Dataset already exists at {download_dir}")

    # Filter and split
    print("Filtering and splitting dataset...")

    images_root = download_dir / "hagrid_30k"
    if not images_root.exists():
        raise FileNotFoundError(f"Expected images under {images_root}")

    # Match folders like train_val_palm, train_val_peace, train_val_fist
    class_dirs = []
    for cname in target_classes:
        d = images_root / f"train_val_{cname}"
        if d.exists() and d.is_dir():
            class_dirs.append(d)
        else:
            print(f"Warning: {d} not found")

    if not class_dirs:
        raise ValueError(
            f"None of the target classes {target_classes} were found under {images_root}"
        )

    # Create output directories
    for split in split_ratios.keys():
        for cname in target_classes:
            (data_dir / split / cname).mkdir(parents=True, exist_ok=True)

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
        train_end = int(total * split_ratios["train"])
        test_end = train_end + int(total * split_ratios["test"])

        splits = {
            "train": image_files[:train_end],
            "test": image_files[train_end:test_end],
            "val": image_files[test_end:],
        }

        for split_name, files in splits.items():
            dest_dir = data_dir / split_name / cname
            for src in files:
                shutil.copy2(src, dest_dir / src.name)
            print(f"  {split_name}: {len(files)} images")

    print("\nDataset preprocessing complete!")
    print(f"Data organized in: {data_dir}")
    for split in split_ratios.keys():
        print(f"  {data_dir / split}/")
        for cname in target_classes:
            count = len(list((data_dir / split / cname).glob("*")))
            print(f"    {cname}/ ({count} images)")

    # Optional cleanup
    if cleanup_raw and raw_data_dir.exists():
        shutil.rmtree(raw_data_dir)
        print("\nRaw data removed.")
