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


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP()
    raise ValueError(f"Unknown model: {name}")
