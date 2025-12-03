import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from kaggle.api.kaggle_api_extended import KaggleApi

import config


def preprocess_dataset():
    """
    Download and preprocess the HaGRID dataset from Kaggle.
    """
    dataset_name = config.dataset_name
    data_dir = Path(config.data_dir)
    target_classes = config.target_classes
    split_ratios = config.split_ratios
    raw_data_dir = data_dir / "raw"
    download_dir = raw_data_dir / "hagrid-sample-30k-384p"
    annotations_dir = data_dir / "hagrid_annotations"

    # Download annotations (only if not already downloaded)
    if not annotations_dir.exists():
        print("Downloading HaGRID annotations...")
        annotations_url = config.annotations_url
        annotations_zip = raw_data_dir / "annotations.zip"

        raw_data_dir.mkdir(parents=True, exist_ok=True)

        try:
            urlretrieve(annotations_url, annotations_zip)
            print(f"Annotations downloaded to {annotations_zip}")

            # Extract annotations to temporary location
            temp_extract_path = raw_data_dir / "annotations_temp"
            with zipfile.ZipFile(annotations_zip, "r") as zip_ref:
                zip_ref.extractall(temp_extract_path)

            # Find the actual annotations folder (might be nested)
            extracted_path = temp_extract_path / "annotations"
            if not extracted_path.exists():
                # Check if annotations are directly in temp_extract_path
                if any((temp_extract_path / "train").iterdir()):
                    extracted_path = temp_extract_path
                else:
                    raise FileNotFoundError(
                        f"Could not find annotations structure in {temp_extract_path}"
                    )

            # Create final annotations directory structure
            annotations_dir.mkdir(parents=True, exist_ok=True)
            
            # Only copy JSON files for our target classes
            for split in ["train", "val", "test"]:
                split_src = extracted_path / split
                split_dst = annotations_dir / split
                
                if split_src.exists():
                    split_dst.mkdir(parents=True, exist_ok=True)
                    
                    # Copy only JSON files for target classes
                    for cname in target_classes:
                        json_file = split_src / f"{cname}.json"
                        if json_file.exists():
                            shutil.copy2(json_file, split_dst / json_file.name)
                            print(f"  Copied {split}/{cname}.json")
                        else:
                            print(f"  [WARN] {split}/{cname}.json not found in annotations")
                else:
                    print(f"  [WARN] Split directory {split} not found in annotations")

            # Clean up temporary extraction and zip file
            shutil.rmtree(temp_extract_path)
            annotations_zip.unlink()
            print(f"Annotations filtered and saved to {annotations_dir}")

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

    def list_images(p: Path):
        files = []
        for f in p.iterdir():
            if f.is_file() and f.suffix.lower() in config.img_extensions:
                files.append(f)
        return files

    # Process each class
    for class_dir in class_dirs:
        cname = class_dir.name.replace("train_val_", "")
        print(f"\nProcessing class: {cname}")

        image_files = list_images(class_dir)
        if not image_files:
            print(f"  Warning: no images in {class_dir}")
            continue

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

    shutil.rmtree(raw_data_dir)
    print("\nRaw data removed.")

if __name__ == "__main__":
    preprocess_dataset()

