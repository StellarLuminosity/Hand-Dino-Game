# src/train.py

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config

from .hagrid_dataset import HagridBBoxImageFolder
from .model import HandGestureCNN


def get_dataloaders():
    """
    Builds train and validation dataloaders
    """
    data_root = Path(config.data_dir)
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected 'train' and 'val' folders under {data_root}, "
            "but one or both do not exist. Did you run preprocess_dataset.py?"
        )

    train_transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ]
    )

    train_dataset = HagridBBoxImageFolder(
        root=str(train_dir),
        transform=train_transform,
    )
    val_dataset = HagridBBoxImageFolder(
        root=str(val_dir),
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


def train_model():
    """
    Main training function that orchestrates the entire training pipeline.
    """
    device = config.device
    epochs = config.epochs
    learning_rate = config.learning_rate
    output_dir = config.output_dir

    print(f"Using device: {device}")

    train_loader, val_loader, class_to_idx = get_dataloaders()

    num_classes = len(class_to_idx)
    model = HandGestureCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
