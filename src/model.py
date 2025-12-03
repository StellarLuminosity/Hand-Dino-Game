# src/model.py

import torch
import torch.nn as nn


class HandGestureCNN(nn.Module):
    """
    CNN architecture for hand gesture recognition.

    - Input: 3 x 64 x 64
    - Conv block 1: 3 -> 32, kernel 3x3, ReLU, MaxPool 2x2
    - Conv block 2: 32 -> 64, kernel 3x3, ReLU, MaxPool 2x2
    - Conv block 3: 64 -> 128, kernel 3x3, ReLU, MaxPool 2x2
    - FC: 128*8*8 -> 256 -> num_classes
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
