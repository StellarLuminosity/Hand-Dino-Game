# src/train.py

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import HagridBBoxImageFolder
from .model import HandGestureCNN

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_dataloaders(
    data_root: str,
    annotations_path: str,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Builds train and validation dataloaders from data_root/{train,val}
    using HagridBBoxImageFolder so that images are cropped with
    HaGRID bounding boxes before being resized.

    Assumes directory structure:
      data_root/train/{class1,class2,...}
      data_root/val/{class1,class2,...}

    and annotations_path points to:
      - A HaGRID annotations JSON file, or
      - A directory like 'hagrid_annotations/' containing *.json files.
    """
    data_root = Path(data_root)

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'val' folders under {data_root}, "
            "but one or both do not exist. Did you run preprocess_dataset.py?"
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = HagridBBoxImageFolder(
        root=str(train_dir),
        annotations_path=annotations_path,
        transform=train_transform,
    )
    val_dataset = HagridBBoxImageFolder(
        root=str(val_dir),
        annotations_path=annotations_path,
        transform=val_transform,
    )

    print("Class to index mapping:", train_dataset.class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.class_to_idx


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            running_correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_correct / total if total > 0 else 0.0

    return epoch_loss, epoch_acc


def train_model(
    data_root: str,
    annotations_path: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    num_workers: int = 4,
    output_dir: str = "checkpoints",
    device: str = "cuda",
):
    """
    Main training function that orchestrates the entire training pipeline.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders(
        data_root=data_root,
        annotations_path=annotations_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(class_to_idx)
    model = HandGestureCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"- {elapsed:.1f}s | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": class_to_idx,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  -> Saved new best model to {ckpt_path}")

    print(f"Training complete. Best validation accuracy: {best_val_acc:.3f}")
