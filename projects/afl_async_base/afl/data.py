from typing import Dict, List
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def load_mnist(data_dir: str):
    tfm = transforms.Compose([transforms.ToTensor()])
    root = Path(data_dir)
    train = datasets.MNIST(root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(root, train=False, download=True, transform=tfm)
    return train, test


def non_iid_shards(labels, num_clients, num_shards):
    shard_size = len(labels) // num_shards
    usable = shard_size * num_shards
    idxs = np.argsort(labels)[:usable]

    shards = np.split(idxs, num_shards)
    rng = np.random.default_rng(42)
    rng.shuffle(shards)

    per = num_shards // num_clients
    client_map = {}
    for c in range(num_clients):
        client_map[c] = np.concatenate(shards[c * per:(c + 1) * per]).tolist()
    return client_map


def make_client_loaders(dataset_name, data_dir, num_clients, batch_size, non_iid, num_shards):
    train, test = load_mnist(data_dir)
    y = np.array(train.targets)

    if non_iid:
        client_idxs = non_iid_shards(y, num_clients, num_shards)
    else:
        idxs = np.arange(len(train))
        np.random.default_rng(42).shuffle(idxs)
        splits = np.array_split(idxs, num_clients)
        client_idxs = {i: splits[i].tolist() for i in range(num_clients)}

    loaders = {
        c: DataLoader(
            Subset(train, client_idxs[c]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        for c in range(num_clients)
    }

    test_loader = DataLoader(test, batch_size=512, shuffle=False)
    return loaders, test_loader
