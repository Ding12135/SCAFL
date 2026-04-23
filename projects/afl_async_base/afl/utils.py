import time
import random
from pathlib import Path
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(log_root: str) -> Path:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(log_root) / run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "events").mkdir(parents=True, exist_ok=True)
    return run_dir


@torch.no_grad()
def state_dict_sub(a, b):
    """Return a - b (CPU tensors)."""
    out = {}
    for k in a.keys():
        out[k] = (a[k] - b[k]).detach().cpu()
    return out


@torch.no_grad()
def state_dict_add_inplace(dst, delta, scale: float):
    """dst += scale * delta (in-place)."""
    for k in dst.keys():
        dst[k].add_(delta[k].to(dst[k].device), alpha=scale)


def now_s():
    return time.time()
