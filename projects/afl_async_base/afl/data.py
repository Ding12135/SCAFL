from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _norm_dataset_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")


# EMNIST：联邦/LEAF 类基准里常用划分（torchvision split -> 类别数）
_EMNIST_META: Dict[str, Tuple[str, int]] = {
    "emnist": ("balanced", 47),
    "emnistbalanced": ("balanced", 47),
    "emnistbymerge": ("bymerge", 47),
    "emnistbyclass": ("byclass", 62),
    "emnistdigits": ("digits", 10),
    "emnistletters": ("letters", 26),
    "emnistmnist": ("mnist", 10),
}

_SUPPORTED = frozenset(
    {"mnist", "fashionmnist", "cifar10", "cifar100", "svhn", *_EMNIST_META.keys()}
)


def infer_num_classes(dataset_name: str) -> int:
    """与 `load_train_test` 一致：用于构建分类头。"""
    dn = _norm_dataset_name(dataset_name)
    if dn in _EMNIST_META:
        return _EMNIST_META[dn][1]
    if dn == "cifar100":
        return 100
    if dn in ("mnist", "fashionmnist", "cifar10", "svhn"):
        return 10
    raise ValueError(
        f"未知数据集 {dataset_name!r}，无法推断 num_classes。"
        f"支持: MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, "
        f"EMNIST(=balanced), EMNIST_BYCLASS, EMNIST_DIGITS, EMNIST_LETTERS, EMNIST_MNIST, EMNIST_BYMERGE"
    )


def _train_labels_for_partition(train) -> np.ndarray:
    """Non-IID shard 划分：兼容 .targets / .labels。"""
    if hasattr(train, "targets"):
        return np.asarray(train.targets)
    if hasattr(train, "labels"):
        return np.asarray(train.labels).reshape(-1)
    raise ValueError(f"无法从数据集 {type(train)} 读取标签用于划分")


def load_train_test(dataset_name: str, data_dir: str) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """
    按 dataset_name 加载 (train, test)。

    图像分类（联邦常用）：
    - MNIST, FashionMNIST（10 类，28×28 灰度）
    - CIFAR10, CIFAR100（32×32 RGB）
    - SVHN（10 类，32×32 RGB）
    - EMNIST*：见 `infer_num_classes` / `_EMNIST_META`
    """
    dn = _norm_dataset_name(dataset_name)
    if dn not in _SUPPORTED:
        raise ValueError(
            f"未知数据集: {dataset_name!r}。支持: "
            f"MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, "
            f"EMNIST, EMNIST_BYCLASS, EMNIST_DIGITS, EMNIST_LETTERS, EMNIST_MNIST, EMNIST_BYMERGE"
        )
    root = str(Path(data_dir))

    if dn == "mnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root, train=True, download=True, transform=tfm)
        test = datasets.MNIST(root, train=False, download=True, transform=tfm)
        return train, test

    if dn == "fashionmnist":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        test = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
        return train, test

    if dn == "cifar10":
        tfm_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        tfm_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616),
                ),
            ]
        )
        train = datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
        return train, test

    if dn == "cifar100":
        tfm_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )
        tfm_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )
        train = datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
        test = datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
        return train, test

    if dn == "svhn":
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.SVHN(root, split="train", download=True, transform=tfm)
        test = datasets.SVHN(root, split="test", download=True, transform=tfm)
        return train, test

    if dn in _EMNIST_META:
        split, _ = _EMNIST_META[dn]
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.EMNIST(root, split=split, train=True, download=True, transform=tfm)
        test = datasets.EMNIST(root, split=split, train=False, download=True, transform=tfm)
        return train, test

    raise ValueError(f"未实现的数据集分支: {dataset_name}")


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
        client_map[c] = np.concatenate(shards[c * per : (c + 1) * per]).tolist()
    return client_map


def make_client_loaders(dataset_name, data_dir, num_clients, batch_size, non_iid, num_shards):
    train, test = load_train_test(dataset_name, data_dir)
    y = _train_labels_for_partition(train)

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
