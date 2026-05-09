from typing import Optional

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int = 28 * 28, hidden: int = 256, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class MnistConvNet(nn.Module):
    """MNIST / FashionMNIST：1×28×28 灰度小卷积网（smoke 与 hetero 基线用）。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class SmallCNN(nn.Module):
    """CIFAR-10 / SVHN 等 32×32 RGB 常用小 CNN（联邦实验里常见规模）。"""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name: str, num_classes: Optional[int] = None) -> nn.Module:
    """num_classes 默认 10；CIFAR-100 / EMNIST 等由 `infer_num_classes(dataset)` 传入。"""
    name = name.lower().strip()
    nc = 10 if num_classes is None else int(num_classes)
    if name == "mlp":
        return MLP(num_classes=nc)
    if name in ("cnn", "mnist_cnn", "lenet"):
        return MnistConvNet(num_classes=nc)
    if name in ("cnn_small", "small_cnn", "cnn_cifar"):
        return SmallCNN(in_channels=3, num_classes=nc)
    raise ValueError(f"Unknown model: {name}")
