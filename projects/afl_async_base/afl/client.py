from typing import Dict
import time
import torch
import torch.nn as nn
from torch.optim import SGD
from .utils import state_dict_sub


def train_one_client(
    client_id: int,
    base_state: Dict[str, torch.Tensor],
    loader,
    model_builder,
    local_epochs: int,
    local_lr: float,
    device: str,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    grad_clip: float = 5.0,
    simulate_hetero: bool = True,
):
    print(f"[CLIENT {client_id}] start training")

    if simulate_hetero:
        time.sleep(0.03 * (client_id % 3))

    base_state_cpu = {k: v.detach().cpu() for k, v in base_state.items()}

    model = model_builder().to(device)
    model.load_state_dict(base_state_cpu, strict=True)
    model.train()

    opt = SGD(model.parameters(), lr=local_lr, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # ===== Dynamic state info for server-side collection =====
    # These statistics are purely for measurement; they do not change the SGD/delta update logic.
    num_samples = 0
    loss_sum = 0.0
    train_started_at = time.time()

    for _ in range(local_epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)

            # Stat collection (does not affect the training algorithm)
            batch_n = int(y.size(0))
            num_samples += batch_n
            loss_sum += float(loss.item()) * batch_n
            loss.backward()

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()

    train_finished_at = time.time()
    train_loss = loss_sum / max(1, num_samples)

    local_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    delta = state_dict_sub(local_state, base_state_cpu)

    # ===== delta norm clip（异步精度关键）=====
    total_norm = 0.0
    for v in delta.values():
        total_norm += v.norm().item() ** 2
    total_norm = total_norm ** 0.5

    max_norm = 10.0
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for k in delta:
            delta[k].mul_(scale)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"[CLIENT {client_id}] finish training, send update")

    # sent_at should reflect the moment right before the update is sent to the server.
    sent_at = time.time()
    return delta, num_samples, local_epochs, train_started_at, train_finished_at, sent_at, train_loss
