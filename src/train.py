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
    Builds train, validation, and test dataloaders
    """
    data_root = Path(config.data_dir)
    batch_size = config.batch_size
    num_workers = config.num_workers

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

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
    test_dataset = HagridBBoxImageFolder(
        root=str(test_dir),
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


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


def train(
    model,
    train_loader,
    val_loader,
    device,
    eval_every=1,
    max_val_batches=None,
):
    num_epochs = config.epochs
    learning_rate = config.learning_rate
    output_dir = config.output_dir

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0.0
    class_to_idx = train_loader.dataset.class_to_idx

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, device, max_batches=max_val_batches
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{num_epochs} "
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


def evaluate(model, data_loader, device, max_batches=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    running_loss = 0.0

    num_classes = len(data_loader.dataset.classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    avg_loss = running_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    idx_to_class = {idx: cls for cls, idx in data_loader.dataset.class_to_idx.items()}
    per_class_acc = {}
    for idx in range(num_classes):
        tp = confusion[idx, idx].item()
        total_cls = confusion[idx, :].sum().item()
        class_name = idx_to_class[idx]
        per_class_acc[class_name] = tp / total_cls if total_cls > 0 else 0.0

    return avg_loss, accuracy, per_class_acc, confusion


def evaluate_checkpoint_on_split(split_name: str = "test"):
    """
    Load the best saved CNN model (best_model.pt) and evaluate it
    on a given split (default: 'test').

    Returns:
        test_loss, test_acc
    """
    device = config.device
    data_root = Path(config.data_dir)
    annotations_path = config.annotations_path
    split_dir = data_root / split_name

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split path {split_dir} does not exist. "
            "Did you run preprocess_dataset.py?"
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(config.normalize_mean, config.normalize_std),
        ]
    )

    dataset = HagridBBoxImageFolder(
        root=str(split_dir),
        annotations_path=annotations_path,
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    ckpt_path = os.path.join(config.output_dir, "best_model.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint {ckpt_path} not found. Train the model first."
        )

    ckpt = torch.load(ckpt_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(class_to_idx)

    model = HandGestureCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, loader, criterion, device)

    print(f"\n=== CNN evaluation on '{split_name}' split ===")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.3f}")

    return test_loss, test_acc


