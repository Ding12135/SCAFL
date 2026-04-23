import copy
import csv
import json
import os
import queue
import sys
import yaml
import torch
import torch.multiprocessing as mp
from typing import Any, Dict, List, Optional, Set, TextIO

from .model import build_model
from .data import make_client_loaders
from .aggregator import Aggregator, UpdateMsg
from .runtime_state import ClientRuntimeState, BufferedUpdate, SystemState
from .dynamic_controller import DynamicController
from .scafl_policy import (
    compute_scafl_p2_objective_for_prefix,
    compute_queue_aware_priority_scores,
    format_policy_params_for_log,
    make_policy_from_config,
)
from .scafl_types import AggregationCandidateSet
from .utils import set_seed, make_run_dir, now_s


class _TeeStdout:
    """Mirror stdout to terminal and a log file (observation only)."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self._terminal = terminal
        self._log_file = log_file

    def write(self, message: str) -> int:
        self._terminal.write(message)
        self._log_file.write(message)
        self._log_file.flush()
        return len(message)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._terminal, name)


@torch.no_grad()
def _delta_l2_norm(delta: Dict[str, torch.Tensor]) -> float:
    """Overall L2 norm of delta dict (observation only)."""
    total = 0.0
    for v in delta.values():
        total += float(v.detach().float().norm().item() ** 2)
    return total ** 0.5


def _round_metrics_delay_extras(
    candidate_set: AggregationCandidateSet,
    policy_decision,
) -> tuple:
    """
    为 round_metrics 计算 d_k 与 id/base_step 字符串；selected_indices 为 buffer 下标。
    """
    items = candidate_set.items
    n = len(items)
    all_dk = [float(it.d_k) for it in items]
    avg_candidate_dk = sum(all_dk) / n if n else 0.0
    sel = policy_decision.selected_indices if policy_decision else None
    if not sel:
        return (
            0.0,
            0.0,
            0.0,
            avg_candidate_dk,
            "",
            "",
            n,
        )
    sdk = [items[i].d_k for i in sel]
    avg_s = sum(sdk) / len(sdk)
    max_s = max(sdk)
    min_s = min(sdk)
    sel_set = set(sel)
    unsel_ids = [items[i].client_id for i in range(n) if i not in sel_set]
    base_steps = [items[i].base_step for i in sel]
    unsel_count = n - len(sel_set)
    return (
        avg_s,
        max_s,
        min_s,
        avg_candidate_dk,
        ",".join(str(x) for x in unsel_ids),
        ",".join(str(x) for x in base_steps),
        unsel_count,
    )


def _round_metrics_queue_score_extras(
    candidate_set: AggregationCandidateSet,
    policy_decision,
    policy_cfg: Dict[str, Any],
    queue_by_client_id: Optional[Dict[int, float]],
) -> tuple:
    """
    候选批内 q_k（virtual_queue）与 queue_aware 下的 priority 统计，供 round_metrics 扩展列。
    """
    items = candidate_set.items
    n = len(items)
    qs = [
        float(queue_by_client_id.get(int(it.client_id), 0.0)) if queue_by_client_id else 0.0
        for it in items
    ]
    avg_candidate_q = sum(qs) / n if n else 0.0
    sel = policy_decision.selected_indices if policy_decision else None
    if sel:
        sq = [qs[i] for i in sel]
        avg_sq = sum(sq) / len(sq)
        max_sq = max(sq)
        min_sq = min(sq)
    else:
        avg_sq = max_sq = min_sq = 0.0

    if policy_cfg.get("type") == "queue_aware":
        _, pri = compute_queue_aware_priority_scores(
            items,
            queue_by_client_id,
            float(policy_cfg["alpha"]),
            float(policy_cfg["beta"]),
        )
        if sel:
            sp = [pri[i] for i in sel]
            avg_ss = sum(sp) / len(sp)
            max_ss = max(sp)
            min_ss = min(sp)
        else:
            avg_ss = max_ss = min_ss = 0.0
    else:
        avg_ss = max_ss = min_ss = 0.0

    return (
        avg_sq,
        max_sq,
        min_sq,
        avg_candidate_q,
        avg_ss,
        max_ss,
        min_ss,
    )


def _round_metrics_drift_penalty_extras(
    candidate_set: AggregationCandidateSet,
    policy_decision,
    queue_by_client_id: Optional[Dict[int, float]],
) -> tuple:
    """
    approx_drift_penalty 的 objective / 前缀统计；其它 policy 填 0，仍写 candidate_queue_sum。
    """
    items = candidate_set.items
    n = len(items)
    qs = [
        float(queue_by_client_id.get(int(it.client_id), 0.0)) if queue_by_client_id else 0.0
        for it in items
    ]
    candidate_queue_sum = sum(qs)
    if policy_decision is None:
        return (0.0, 0.0, 0.0, 0, 0, candidate_queue_sum, 0.0)
    if policy_decision.objective_value is not None:
        uq = float(policy_decision.queue_term) if policy_decision.queue_term is not None else 0.0
        return (
            float(policy_decision.objective_value),
            float(policy_decision.time_term or 0.0),
            uq,
            int(policy_decision.evaluated_prefix_count or 0),
            int(policy_decision.selected_prefix_size or 0),
            candidate_queue_sum,
            uq,
        )
    sel = policy_decision.selected_indices
    if sel:
        unsel_q = sum(qs[i] for i in range(n) if i not in set(sel))
    else:
        unsel_q = 0.0
    return (0.0, 0.0, 0.0, 0, 0, candidate_queue_sum, unsel_q)


def _round_metrics_scafl_p2_extras(
    candidate_set: AggregationCandidateSet,
    policy_decision,
    queue_by_client_id: Optional[Dict[int, float]],
    tau_max_used: int,
    V: float,
) -> tuple:
    """
    计算 P2 对齐字段：
    - selected_objective_p2 / D_t
    - tau_max_used
    - candidate_term_sum = sum_i q_i*((tau_i+1)*(1-beta_i)-tau_max)
    - beta ones/zeros
    """
    items = candidate_set.items
    n = len(items)
    sel = list(policy_decision.selected_indices or []) if policy_decision else []
    obj_p2, d_t, term_sum, _ = compute_scafl_p2_objective_for_prefix(
        items=items,
        sorted_prefix_indices=sel,
        queue_by_client_id=queue_by_client_id,
        tau_max_used=tau_max_used,
        V=V,
    )
    candidate_clients = set(int(it.client_id) for it in items)
    selected_clients = set(int(items[i].client_id) for i in sel if 0 <= i < n)
    beta_ones = len(selected_clients)
    beta_zeros = max(len(candidate_clients) - beta_ones, 0)
    return obj_p2, d_t, int(tau_max_used), term_sum, beta_ones, beta_zeros


def _decision_debug_rows_for_round(
    *,
    logical_round: int,
    candidate_set: AggregationCandidateSet,
    selected_indices: List[int],
    queue_by_client_id: Optional[Dict[int, float]],
    tau_max_used: int,
    policy_type: str,
    policy_decision=None,
) -> List[List[Any]]:
    """
    每行一条 pending update。beta_k 为 client-level；update_in_aggregated_prefix 表示该行下标是否在
    policy selected_indices 内（update-level，可与同 client 其他行不同）。
    """
    rows: List[List[Any]] = []
    selected_idx_set: Set[int] = {int(x) for x in (selected_indices or [])}
    cand_map = {}
    if policy_decision is not None and getattr(policy_decision, "candidate_decisions", None):
        for rec in policy_decision.candidate_decisions or []:
            # 以 update_id 为唯一键，避免同一 client 多条 candidate 的歧义。
            cand_map[str(rec.update_id)] = rec

    # 客户端级 beta：只要该 client_id 的任意 pending update 被选中，则 beta_k=1
    selected_client_ids_set: Set[int] = set()
    if policy_decision is not None and policy_decision.selected_client_ids is not None:
        selected_client_ids_set = {int(x) for x in policy_decision.selected_client_ids or []}
    else:
        selected_client_ids_set = set()

    # candidate_set.logical_round 是“聚合事件发生前”的工程近似轮次：
    # 在 server 中 flush/apply 后才将 logical_round +1，因此该值通常比逻辑聚合轮次小 1。
    candidate_set_round_before_agg_approx = int(candidate_set.logical_round)
    for i, it in enumerate(candidate_set.items):
        update_in_aggregated_prefix = 1 if i in selected_idx_set else 0
        q_k = float(queue_by_client_id.get(int(it.client_id), 0.0)) if queue_by_client_id else 0.0
        tau_k = int(it.staleness)
        if selected_client_ids_set:
            beta_k = 1 if int(it.client_id) in selected_client_ids_set else 0
        else:
            beta_k = 1 if i in set(selected_indices) else 0
        term_k = q_k * (((tau_k + 1) * (1 - beta_k)) - int(tau_max_used))
        rec = cand_map.get(str(getattr(it, "update_id", "")))
        must_select = int(bool(rec.must_select)) if rec is not None else 0
        decision_reason = rec.decision_reason if rec is not None else ""

        update_id = str(getattr(it, "update_id", ""))
        # entered_buffer_round 是“消息被 append 进 Aggregator.buffer 时”的工程近似轮次。
        buffer_entry_round_approx = int(getattr(it, "entered_buffer_round", -1))
        buffer_resident_rounds_approx = (
            max(0, candidate_set_round_before_agg_approx - buffer_entry_round_approx)
            if buffer_entry_round_approx >= 0
            else 0
        )
        was_carried_over = (
            1
            if buffer_entry_round_approx >= 0
            and buffer_entry_round_approx < candidate_set_round_before_agg_approx
            else 0
        )
        rows.append(
            [
                logical_round,
                candidate_set_round_before_agg_approx,
                int(it.client_id),
                update_id,
                buffer_entry_round_approx,
                int(buffer_resident_rounds_approx),
                int(was_carried_over),
                int(it.base_step),
                float(it.d_k),
                float(it.compute_time),
                float(it.upload_delay),
                q_k,
                tau_k,
                beta_k,
                term_k,
                must_select,
                update_in_aggregated_prefix,
                decision_reason,
                policy_type,
            ]
        )
    return rows


def collect_system_state(
    client_states: Dict[int, ClientRuntimeState],
    metadata_buffer: List[BufferedUpdate],
) -> SystemState:
    """
    基于 server 自维护的 metadata_buffer（只存 BufferedUpdate）与各客户端运行时估计状态，构造 SystemState。

    该函数仅负责统计输出，不改变聚合行为。
    """

    stalenesses = [int(bu.staleness) for bu in metadata_buffer]
    buffer_size = len(stalenesses)
    avg_buffer_staleness = (sum(stalenesses) / buffer_size) if buffer_size > 0 else 0.0
    max_buffer_staleness = max(stalenesses) if buffer_size > 0 else 0

    compute_times = [
        cs.estimated_compute_time
        for cs in client_states.values()
        if cs.last_recv_step is not None and cs.last_recv_step >= 0
    ]
    upload_delays = [
        cs.estimated_upload_delay
        for cs in client_states.values()
        if cs.last_recv_step is not None and cs.last_recv_step >= 0
    ]

    avg_compute_time = (sum(compute_times) / len(compute_times)) if compute_times else 0.0
    avg_upload_delay = (sum(upload_delays) / len(upload_delays)) if upload_delays else 0.0

    if len(compute_times) >= 2:
        var = sum((t - avg_compute_time) ** 2 for t in compute_times) / len(compute_times)
        compute_heterogeneity = var ** 0.5
    else:
        compute_heterogeneity = 0.0

    return SystemState(
        avg_upload_delay=avg_upload_delay,
        avg_compute_time=avg_compute_time,
        compute_heterogeneity=compute_heterogeneity,
        buffer_size=buffer_size,
        avg_buffer_staleness=avg_buffer_staleness,
        max_buffer_staleness=max_buffer_staleness,
    )


@torch.no_grad()
def evaluate(model, test_loader, device: str):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)

        loss_sum += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

    return correct / total, loss_sum / total


def client_proc(client_id: int, shared, lock, recv_q, cfg):
    from .client import train_one_client

    model_builder = lambda: build_model(cfg["model"])
    loader = shared["client_loaders"][client_id]
    device = cfg["device"]

    for _ in range(cfg["updates_per_client"]):
        with lock:
            base_step = int(shared["global_step"])
            base_state = copy.deepcopy(shared["global_state"])

        delta, num_samples, local_epochs, train_started_at, train_finished_at, sent_at, train_loss = train_one_client(
            client_id=client_id,
            base_state=base_state,
            loader=loader,
            model_builder=model_builder,
            local_epochs=cfg["local_epochs"],
            local_lr=cfg["local_lr"],
            device=device,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 1e-4),
            grad_clip=cfg.get("grad_clip", 5.0),
            simulate_hetero=True,
        )

        msg = UpdateMsg(
            client_id=client_id,
            base_step=base_step,
            delta=delta,
            num_samples=num_samples,
            local_epochs=local_epochs,
            train_started_at=train_started_at,
            train_finished_at=train_finished_at,
            sent_at=sent_at,
            train_loss=train_loss,
        )
        recv_q.put(msg)

    print(f"[CLIENT {client_id}] process finished")


# triggered_flush / beta_k：本条消息在当前处理时刻是否触发一次全局应用（非“最终被 flush 吸收”）。
METRICS_HEADER = [
    "recv_step",
    "global_step",
    "wall_time",
    "client_id",
    "base_step",
    "staleness",
    "accepted",
    "applied",
    "triggered_flush",
    "flush_reason",
    "dropped_by_cutoff",
    "buffer_len_before",
    "buffer_len_after",
    "metadata_buffer_len",
    "buffer_target_t",
    "tau_max_t",
    "delay_score",
    "heter_score",
    "staleness_score",
    "train_loss",
    "num_samples",
    "compute_time",
    "upload_delay",
    "delta_norm",
    "client_q",
    "beta_k",
    "test_acc",
    "test_loss",
]

FLUSH_METRICS_HEADER = [
    "global_step",
    "wall_time",
    "flush_reason",
    "num_updates",
    "avg_staleness",
    "max_staleness",
    "min_staleness",
    "avg_compute_time",
    "avg_upload_delay",
    "total_samples",
    "buffer_target_t",
    "tau_max_t",
]

# 每次 buffered flush 一行：对齐论文 M_t、d_k、round(t) 的工程近似（见 main 内注释）。
#
# 语义提示（client vs update）：
# - selected_client_ids / unselected_client_ids：策略最优 prefix 上的 **client_id**（聚合语义）。
# - aggregated_update_ids：本轮 **candidate 行下标 ∈ selected_indices** 的 **update_id**（实际参与聚合的 pending 行）。
# - selected_beta_ones / selected_beta_zeros：与 P2 目标一致的 **client-level** β 计数（非 pending 行数）。
ROUND_METRICS_HEADER = [
    "logical_round",
    "global_step_before",
    "global_step_after",
    "queue_update_applied",
    "candidate_client_ids",
    "selected_count",
    "candidate_count",
    "remaining_buffer_count",
    "unselected_count",
    "flush_reason",
    "avg_selected_staleness",
    "max_selected_staleness",
    "wall_time",
    "selected_client_ids",
    "unselected_client_ids",
    "selected_base_steps",
    "avg_selected_dk",
    "max_selected_dk",
    "min_selected_dk",
    "avg_candidate_dk",
    "policy_type",
    "avg_selected_q",
    "max_selected_q",
    "min_selected_q",
    "avg_candidate_q",
    "avg_selected_score",
    "max_selected_score",
    "min_selected_score",
    "policy_params",
    "avg_candidate_q_before",
    "avg_selected_q_before",
    "avg_unselected_q_before",
    "avg_candidate_q_after",
    "avg_selected_q_after",
    "avg_unselected_q_after",
    "queue_rule_version",
    "selected_objective",
    "selected_time_term",
    "selected_queue_term",
    "selected_objective_p2",
    "selected_D_t",
    "selected_tau_max_used",
    "candidate_term_sum",
    "selected_beta_ones",
    "selected_beta_zeros",
    "selected_client_count",
    "unselected_client_count",
    "evaluated_prefix_count",
    "selected_prefix_size",
    "candidate_queue_sum",
    "unselected_queue_sum",
    "q_delta_selected_avg",
    "q_delta_unselected_avg",
    "aggregated_update_ids",
    "carried_over_update_ids",
    "dropped_update_ids",
    "buffer_update_ids_after_round",
]

# decision_debug 每行对应一条 pending candidate（update 粒度）：
# - beta_k：策略/队列意义上的 **client-level** β（与 policy selected_client_ids 或回退到 index 一致）。
# - update_in_aggregated_prefix：本条 candidate **行下标** 是否落在 policy selected_indices 内（**update-level**，可与同 client 的另一行不同）。
DECISION_DEBUG_HEADER = [
    "logical_round",
    "candidate_set_round_before_agg_approx",
    "client_id",
    "update_id",
    "buffer_entry_round_approx",
    "buffer_resident_rounds_approx",
    "was_carried_over",
    "base_step",
    "d_k",
    "compute_time",
    "upload_delay",
    "q_k_pre",
    "tau_k",
    "beta_k",
    "term_k",
    "must_select",
    "update_in_aggregated_prefix",
    "decision_reason",
    "policy_type",
]

P2_PREFIX_DIAGNOSTICS_HEADER = [
    "logical_round",
    "policy_type",
    "candidate_count",
    "prefix_size",
    "is_selected_prefix",
    "selected_indices",
    "selected_client_ids",
    "unselected_client_ids",
    "D_t_prefix",
    "tau_max_used",
    "objective_p2",
    "candidate_term_sum",
    "beta_ones",
    "beta_zeros",
    "selected_dks",
    "selected_qs",
    "selected_taus",
    "unselected_qs",
    "unselected_taus",
    "policy_params",
    "wall_time",
]

QUEUE_TRACE_HEADER = [
    "logical_round",
    "client_id",
    "was_candidate",
    "queue_beta_client",
    "q_before",
    "q_after",
    "staleness",
    "d_k",
    "beta_k",
    "policy_type",
]

QUEUE_RULE_VERSION = "v1_event_candidate_beta"


def _avg_or_zero(vals: List[float]) -> float:
    return (sum(vals) / len(vals)) if vals else 0.0


def apply_queue_update_for_aggregation_event(
    *,
    logical_round: int,
    candidate_set: AggregationCandidateSet,
    selected_indices: Optional[List[int]],
    client_states: Dict[int, ClientRuntimeState],
    tau_max_t: int,
    policy_type: str,
) -> Dict[str, Any]:
    """
    在一次真实聚合事件（logical_round）后更新 virtual_queue（工程版 Q_k）。

    当前规则（queue_rule_version=v1_event_candidate_beta）：
    - 仅更新本轮 candidate clients；非 candidate 客户端本轮不变。
    - 对 candidate i：
        beta_i = 1 (selected) else 0
        q_i <- max( q_i + (staleness_i + 1) * (1 - beta_i) - tau_max_t, 0 )
    - 直观上：
        selected 客户端会因 -tau_max_t 项下降（或归零）；
        unselected 客户端会额外累积 (staleness+1) 惩罚，体现“本轮未服务”的压力。

    这实现了论文队列递推的“工程事件边界版”：在每个真实 aggregation event
   （logical_round）之后，把每个 candidate client 的 Q_k(t) 更新为 Q_k(t+1)。

    与论文仍可能存在的差异主要来自：
    - 我们用 server 端计算得到的 staleness 近似 tau_k(t)；
    - 当前仅对 candidate 集合里的 client 更新 Q_k（非 candidate 保持不变）。
    """
    items = candidate_set.items
    n = len(items)
    selected_set: Set[int] = set(selected_indices or [])
    tau_eff = max(int(tau_max_t), 0)

    # 按 client_id 去重：在同一 logical_round 内，每个客户端只更新一次 Q_k(t)->Q_k(t+1)
    items_by_client: Dict[int, List[Tuple[int, Any]]] = {}
    for idx, it in enumerate(items):
        cid = int(it.client_id)
        items_by_client.setdefault(cid, []).append((idx, it))

    q_before_by_client: Dict[int, float] = {}
    q_after_by_client: Dict[int, float] = {}
    beta_by_client: Dict[int, int] = {}

    # 先计算每个客户端在本轮的 beta（只要该客户端的任何 pending 被 selected_indices 选入，则 beta=1）
    for cid, lst in items_by_client.items():
        beta_client = 1 if any(idx in selected_set for idx, _it in lst) else 0
        beta_by_client[cid] = beta_client
        cs = client_states[cid]
        q_prev = float(cs.virtual_queue)

        if beta_client == 1:
            # beta=1 时 (staleness+1)*(1-beta)=0，tau 取什么都不影响递推项
            q_new = q_prev - tau_eff
        else:
            # beta=0 时使用该客户端在本轮候选中的最坏 staleness
            tau_k_use = max(int(it.staleness) for _idx, it in lst)
            q_new = q_prev + (tau_k_use + 1) - tau_eff
        q_new = max(float(q_new), 0.0)

        q_before_by_client[cid] = q_prev
        q_after_by_client[cid] = float(q_new)
        cs.virtual_queue = float(q_new)

    # 生成 trace_rows：每个 pending update 复用同一个客户端级 Q_k(t)/Q_k(t+1)
    trace_rows: List[List[Any]] = []
    candidate_clients = sorted(items_by_client.keys())
    candidate_q_before = [q_before_by_client[cid] for cid in candidate_clients]
    candidate_q_after = [q_after_by_client[cid] for cid in candidate_clients]

    selected_q_before: List[float] = []
    selected_q_after: List[float] = []
    unselected_q_before: List[float] = []
    unselected_q_after: List[float] = []
    selected_deltas: List[float] = []
    unselected_deltas: List[float] = []

    for cid in candidate_clients:
        beta_client = beta_by_client[cid]
        q_prev = q_before_by_client[cid]
        q_new = q_after_by_client[cid]
        if beta_client == 1:
            selected_q_before.append(q_prev)
            selected_q_after.append(q_new)
            selected_deltas.append(q_new - q_prev)
        else:
            unselected_q_before.append(q_prev)
            unselected_q_after.append(q_new)
            unselected_deltas.append(q_new - q_prev)

    for idx, it in enumerate(items):
        cid = int(it.client_id)
        beta_client = beta_by_client.get(cid, 0)
        trace_rows.append(
            [
                logical_round,
                cid,
                1,
                beta_client,
                q_before_by_client.get(cid, 0.0),
                q_after_by_client.get(cid, 0.0),
                int(it.staleness),
                float(it.d_k),
                beta_client,
                policy_type,
            ]
        )

    return {
        "queue_update_applied": 1 if len(items) > 0 else 0,
        "q_before_by_idx": {},  # deprecated: now stored per-client only
        "q_after_by_idx": {},  # deprecated
        "avg_candidate_q_before": _avg_or_zero(candidate_q_before),
        "avg_selected_q_before": _avg_or_zero(selected_q_before),
        "avg_unselected_q_before": _avg_or_zero(unselected_q_before),
        "avg_candidate_q_after": _avg_or_zero(candidate_q_after),
        "avg_selected_q_after": _avg_or_zero(selected_q_after),
        "avg_unselected_q_after": _avg_or_zero(unselected_q_after),
        "q_delta_selected_avg": _avg_or_zero(selected_deltas),
        "q_delta_unselected_avg": _avg_or_zero(unselected_deltas),
        "candidate_queue_sum": sum(candidate_q_before),
        "unselected_queue_sum": sum(
            q_before_by_client[cid]
            for cid in candidate_clients
            if beta_by_client.get(cid, 0) == 0
        ),
        "trace_rows": trace_rows,
        "queue_rule_version": QUEUE_RULE_VERSION,
    }


def main():
    cfg_path = os.environ.get("AFL_CONFIG", "").strip()
    if not cfg_path:
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
    cfg_path = os.path.abspath(cfg_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    run_dir = make_run_dir(cfg["log_root"])

    events_dir = run_dir / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = events_dir / "debug.log"
    _orig_stdout = sys.stdout
    _debug_fp = open(debug_log_path, "w", encoding="utf-8")
    sys.stdout = _TeeStdout(_orig_stdout, _debug_fp)

    try:
        print("[SERVER] starting async federated learning")
        print(f"[SERVER] device={cfg['device']} | async_mode={cfg['async_mode']}")

        train_policy, policy_cfg = make_policy_from_config(cfg)
        print(f"[SERVER] policy.type={policy_cfg['type']}")

        controller = DynamicController(cfg)

        with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        client_loaders, test_loader = make_client_loaders(
            dataset_name=cfg["dataset"],
            data_dir=cfg["data_dir"],
            num_clients=cfg["num_clients"],
            batch_size=cfg["batch_size"],
            non_iid=cfg["non_iid"],
            num_shards=cfg["num_shards"],
        )

        device = cfg["device"]
        model = build_model(cfg["model"]).to(device)
        global_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

        agg = Aggregator(
            server_lr=cfg["server_lr"],
            async_mode=cfg["async_mode"],
            buffer_size=cfg["buffer_size"],
            staleness_weight=cfg["staleness_weight"],
            staleness_lambda=cfg["staleness_lambda"],
            staleness_cutoff=cfg["staleness_cutoff"],
        )

        client_states: Dict[int, ClientRuntimeState] = {
            cid: ClientRuntimeState(
                client_id=cid,
                last_base_step=0,
                last_recv_step=-1,
                current_staleness=0,
                estimated_compute_time=0.0,
                estimated_upload_delay=0.0,
                virtual_queue=0.0,
            )
            for cid in range(cfg["num_clients"])
        }

        metadata_buffer: List[BufferedUpdate] = []

        manager = mp.Manager()
        lock = manager.Lock()
        recv_q = manager.Queue(maxsize=500)

        shared = manager.dict()
        shared["global_state"] = global_state
        shared["global_step"] = 0
        shared["client_loaders"] = client_loaders

        print("[SERVER] spawning client processes")
        procs = []
        for cid in range(cfg["num_clients"]):
            p = mp.Process(
                target=client_proc,
                args=(cid, shared, lock, recv_q, cfg),
                daemon=True,
            )
            p.start()
            procs.append(p)
            print(f"[SERVER] client {cid} started")

        metrics_path = run_dir / "metrics.csv"
        flush_metrics_path = run_dir / "flush_metrics.csv"
        round_metrics_path = run_dir / "round_metrics.csv"
        queue_trace_path = run_dir / "queue_trace.csv"
        decision_debug_path = run_dir / "decision_debug.csv"
        p2_prefix_diag_path = run_dir / "p2_prefix_diagnostics.csv"

        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(METRICS_HEADER)

        with open(flush_metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(FLUSH_METRICS_HEADER)

        with open(round_metrics_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(ROUND_METRICS_HEADER)
        with open(queue_trace_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(QUEUE_TRACE_HEADER)
        with open(decision_debug_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(DECISION_DEBUG_HEADER)
        with open(p2_prefix_diag_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(P2_PREFIX_DIAGNOSTICS_HEADER)

        start_t = now_s()
        total_msgs_target = cfg["num_clients"] * cfg["updates_per_client"]
        recv_step = 0
        applied_updates = 0
        update_seq = 0  # for update_id lifecycle tracking

        last_test_acc = -1.0
        last_test_loss = -1.0

        staleness_sum = 0.0
        staleness_max = 0
        accepted_count = 0
        flush_count = 0
        flush_size_sum = 0
        compute_time_sum = 0.0
        upload_delay_sum = 0.0
        max_client_q = 0.0

        # logical_round（统一语义）：
        # - 仅表示“一次实际发生的 buffered 聚合事件（flush/apply subset）”；
        # - message arrival 本身不计入 round；
        # - 仅当 buffered 且 applied=1 时 logical_round += 1。
        completed_flush_rounds = 0

        while recv_step < total_msgs_target:
            try:
                msg: UpdateMsg = recv_q.get(timeout=5.0)
            except queue.Empty:
                continue

            recv_step += 1
            recv_at = now_s()
            msg.recv_at = recv_at

            with lock:
                global_step_before_apply = int(shared["global_step"])
                gs_before = global_step_before_apply
                staleness_raw = gs_before - int(msg.base_step)
                if staleness_raw < 0:
                    print(
                        f"[WARNING] staleness_raw < 0, clamp: client={msg.client_id} raw={staleness_raw}"
                    )
                staleness = max(0, staleness_raw)

            staleness_sum += float(staleness)
            staleness_max = max(staleness_max, int(staleness))

            compute_time_raw = float(msg.train_finished_at - msg.train_started_at)
            if compute_time_raw < 0:
                print(
                    f"[WARNING] compute_time < 0, clamp: client={msg.client_id} raw={compute_time_raw}"
                )
            compute_time = max(0.0, compute_time_raw)

            upload_delay_raw = float(recv_at - msg.sent_at)
            if upload_delay_raw < 0:
                print(
                    f"[WARNING] upload_delay < 0, clamp: client={msg.client_id} raw={upload_delay_raw}"
                )
            upload_delay = max(0.0, upload_delay_raw)

            num_samples_raw = int(msg.num_samples)
            if num_samples_raw < 0:
                print(
                    f"[WARNING] num_samples < 0, clamp: client={msg.client_id} raw={num_samples_raw}"
                )
            num_samples = max(0, num_samples_raw)

            compute_time_sum += compute_time
            upload_delay_sum += upload_delay

            # === update_id：只在该条消息会进入 Aggregator buffer 时生成 ===
            # 进入条件：staleness 通过 Aggregator 的 cutoff 接受逻辑。
            if cfg["async_mode"] == "buffered" and agg._accept(int(staleness)):
                update_seq += 1
                msg.update_id = (
                    f"u{update_seq:08d}-c{int(msg.client_id)}-bs{int(msg.base_step)}"
                )
                # 语义：该条消息首次进入 buffer 的 logical_round（工程近似为当前 completed_flush_rounds）。
                msg.entered_buffer_round = int(completed_flush_rounds)
            else:
                msg.update_id = ""
                msg.entered_buffer_round = -1

            bu = BufferedUpdate(
                client_id=msg.client_id,
                base_step=int(msg.base_step),
                arrival_step=int(gs_before),
                staleness=int(staleness),
                num_samples=num_samples,
                train_loss=msg.train_loss,
                compute_time=compute_time,
                upload_delay=upload_delay,
                delta=msg.delta,
            )

            cs = client_states[msg.client_id]
            cs.last_base_step = int(msg.base_step)
            cs.last_recv_step = int(gs_before)
            cs.current_staleness = int(staleness)
            cs.estimated_compute_time = compute_time
            cs.estimated_upload_delay = upload_delay

            system_state = collect_system_state(client_states, metadata_buffer)

            delta_norm = _delta_l2_norm(msg.delta)

            print(
                f"[UPDATE] recv_step={recv_step} client_id={msg.client_id} base_step={int(msg.base_step)} "
                f"global_step={global_step_before_apply} staleness={int(staleness)} "
                f"train_loss={msg.train_loss} compute_time={compute_time:.6f} upload_delay={upload_delay:.6f}"
            )

            ctrl = controller.compute(system_state)
            print(
                f"[CONTROL] tau_max_t={ctrl.tau_max_t} buffer_target_t={ctrl.buffer_target_t} "
                f"delay_score={ctrl.delay_score:.6f} heter_score={ctrl.heter_score:.6f} "
                f"staleness_score={ctrl.staleness_score:.6f}"
            )

            buffer_target_override = ctrl.buffer_target_t if cfg["async_mode"] == "buffered" else None
            tau_max_override = ctrl.tau_max_t if cfg["async_mode"] == "buffered" else None

            # client_id -> virtual_queue（Q_k 工程近似）；供 queue_aware policy 显式使用，后续对齐论文 Q_k(t)。
            queue_by_client_id: Dict[int, float] = {
                int(cid): float(cs.virtual_queue) for cid, cs in client_states.items()
            }

            policy_decision = None
            last_candidate_count = 0
            candidate_set = None
            if cfg["async_mode"] == "buffered":
                candidate_set = agg.preview_aggregation_candidate_set(
                    msg,
                    staleness,
                    completed_flush_rounds,
                    gs_before,
                )
                last_candidate_count = len(candidate_set.items)
                target_sz = (
                    int(buffer_target_override)
                    if buffer_target_override is not None
                    else int(agg.buffer_size)
                )
                policy_decision = train_policy.decide(
                    candidate_set,
                    system_state,
                    target_size=target_sz,
                    tau_max_override=tau_max_override,
                    queue_by_client_id=queue_by_client_id,
                )

            with lock:
                buffer_len_before = len(agg.buffer)
                max_staleness_agg_before = max((s for _, s in agg.buffer), default=0)

                step_result = agg.step(
                    shared["global_state"],
                    msg,
                    staleness,
                    buffer_target_override=buffer_target_override,
                    tau_max_override=tau_max_override,
                    policy_decision=policy_decision,
                    logical_round=completed_flush_rounds,
                    global_step=gs_before,
                )
                applied = step_result.applied
                accepted = step_result.accepted
                triggered_flush = step_result.triggered_flush
                buffer_len_after = len(agg.buffer)
                max_staleness_agg_after = max((s for _, s in agg.buffer), default=0)

                if applied:
                    shared["global_step"] = gs_before + 1
                    applied_updates += 1

                global_step_after = int(shared["global_step"])

            if accepted:
                accepted_count += 1

            dropped_by_cutoff = 0 if accepted else 1
            if dropped_by_cutoff:
                print(
                    f"[WARNING] update dropped_by_cutoff=1 client_id={msg.client_id} "
                    f"staleness={int(staleness)} cutoff={agg.staleness_cutoff}"
                )

            flush_reason_str = step_result.flush_reason if step_result.flush_reason else ""

            if cfg["async_mode"] == "buffered":
                target_sz = int(buffer_target_override) if buffer_target_override is not None else int(agg.buffer_size)
                print(
                    f"[BUFFER] buffer_len_before={buffer_len_before} buffer_len_after={buffer_len_after} "
                    f"target_buffer_size={target_sz} max_staleness_before={max_staleness_agg_before} "
                    f"max_staleness_after={max_staleness_agg_after} applied={int(applied)} "
                    f"triggered_flush={int(triggered_flush)} flush_reason={flush_reason_str!r}"
                )
            else:
                print(
                    f"[BUFFER] buffer_len_before={buffer_len_before} buffer_len_after={buffer_len_after} "
                    f"applied={int(applied)} triggered_flush={int(triggered_flush)} "
                    f"flush_reason={flush_reason_str!r}"
                )

            if cfg["async_mode"] == "buffered":
                if applied:
                    metadata_buffer.clear()
                else:
                    if buffer_len_after > buffer_len_before:
                        metadata_buffer.append(bu)

            metadata_buffer_len = len(metadata_buffer)

            if (
                cfg["async_mode"] == "buffered"
                and applied
                and step_result.flush_snapshot is not None
            ):
                snap = step_result.flush_snapshot
                flush_count += 1
                flush_size_sum += snap.num_updates
                flush_wall_time = now_s() - start_t
                with open(flush_metrics_path, "a", newline="", encoding="utf-8") as ff:
                    csv.writer(ff).writerow(
                        [
                            global_step_after,
                            flush_wall_time,
                            flush_reason_str or "",
                            snap.num_updates,
                            snap.avg_staleness,
                            snap.max_staleness,
                            snap.min_staleness,
                            snap.avg_compute_time,
                            snap.avg_upload_delay,
                            snap.total_samples,
                            int(ctrl.buffer_target_t),
                            int(ctrl.tau_max_t),
                        ]
                    )

                flush_round_id = completed_flush_rounds + 1
                completed_flush_rounds = flush_round_id
                sel_ids = policy_decision.selected_client_ids if policy_decision else []
                ids_str = ",".join(str(x) for x in (sel_ids or []))
                round_wall_time = now_s() - start_t
                candidate_ids_str = ""
                queue_extras = {
                    "queue_update_applied": 0,
                    "avg_candidate_q_before": 0.0,
                    "avg_selected_q_before": 0.0,
                    "avg_unselected_q_before": 0.0,
                    "avg_candidate_q_after": 0.0,
                    "avg_selected_q_after": 0.0,
                    "avg_unselected_q_after": 0.0,
                    "q_delta_selected_avg": 0.0,
                    "q_delta_unselected_avg": 0.0,
                    "candidate_queue_sum": 0.0,
                    "unselected_queue_sum": 0.0,
                    "trace_rows": [],
                    "queue_rule_version": QUEUE_RULE_VERSION,
                }
                if candidate_set is not None and policy_decision is not None:
                    (
                        avg_sel_dk,
                        max_sel_dk,
                        min_sel_dk,
                        avg_cand_dk,
                        unsel_ids_str,
                        sel_base_steps_str,
                        unselected_count_rm,
                    ) = _round_metrics_delay_extras(candidate_set, policy_decision)
                    (
                        avg_sel_q,
                        max_sel_q,
                        min_sel_q,
                        avg_cand_q,
                        avg_sel_score,
                        max_sel_score,
                        min_sel_score,
                    ) = _round_metrics_queue_score_extras(
                        candidate_set,
                        policy_decision,
                        policy_cfg,
                        queue_by_client_id,
                    )
                    candidate_ids_str = ",".join(
                        str(int(it.client_id)) for it in candidate_set.items
                    )
                    queue_extras = apply_queue_update_for_aggregation_event(
                        logical_round=flush_round_id,
                        candidate_set=candidate_set,
                        selected_indices=policy_decision.selected_indices,
                        client_states=client_states,
                        tau_max_t=int(ctrl.tau_max_t),
                        policy_type=policy_cfg["type"],
                    )
                    if queue_extras["trace_rows"]:
                        with open(queue_trace_path, "a", newline="", encoding="utf-8") as qf:
                            wq = csv.writer(qf)
                            for row in queue_extras["trace_rows"]:
                                wq.writerow(row)
                else:
                    avg_sel_dk = max_sel_dk = min_sel_dk = avg_cand_dk = 0.0
                    unsel_ids_str = ""
                    sel_base_steps_str = ""
                    unselected_count_rm = 0
                    avg_sel_q = max_sel_q = min_sel_q = avg_cand_q = 0.0
                    avg_sel_score = max_sel_score = min_sel_score = 0.0
                policy_params_str = format_policy_params_for_log(policy_cfg)
                if candidate_set is not None:
                    (
                        sel_obj,
                        sel_time_term,
                        sel_queue_term,
                        eval_prefix_cnt,
                        sel_prefix_sz,
                        cand_q_sum,
                        unsel_q_sum,
                    ) = _round_metrics_drift_penalty_extras(
                        candidate_set,
                        policy_decision,
                        queue_by_client_id,
                    )
                else:
                    sel_obj = sel_time_term = sel_queue_term = 0.0
                    eval_prefix_cnt = sel_prefix_sz = 0
                    cand_q_sum = unsel_q_sum = 0.0
                if candidate_set is not None and policy_decision is not None:
                    tau_used_rm = int(ctrl.tau_max_t)
                    (
                        sel_obj_p2,
                        sel_D_t,
                        sel_tau_max_used,
                        cand_term_sum,
                        beta_ones,
                        beta_zeros,
                    ) = _round_metrics_scafl_p2_extras(
                        candidate_set,
                        policy_decision,
                        queue_by_client_id,
                        tau_used_rm,
                        float(policy_cfg.get("V", 1.0)),
                    )
                    dbg_rows = _decision_debug_rows_for_round(
                        logical_round=flush_round_id,
                        candidate_set=candidate_set,
                        selected_indices=list(policy_decision.selected_indices or []),
                        queue_by_client_id=queue_by_client_id,
                        tau_max_used=tau_used_rm,
                        policy_type=str(policy_cfg["type"]),
                        policy_decision=policy_decision,
                    )
                    if dbg_rows:
                        with open(decision_debug_path, "a", newline="", encoding="utf-8") as df:
                            wd = csv.writer(df)
                            for row in dbg_rows:
                                wd.writerow(row)
                    pfx = policy_decision.prefix_evaluations
                    if policy_cfg["type"] in ("scafl_p2", "scafl_skeleton", "scafl") and pfx:
                        with open(p2_prefix_diag_path, "a", newline="", encoding="utf-8") as pf:
                            wp = csv.writer(pf)
                            selected_prefix_size = int(policy_decision.selected_prefix_size or -1)
                            for rec in pfx:
                                wp.writerow(
                                    [
                                        flush_round_id,
                                        policy_cfg["type"],
                                        rec.candidate_count,
                                        rec.prefix_size,
                                        int(rec.prefix_size == selected_prefix_size),
                                        "|".join(str(x) for x in rec.selected_indices),
                                        "|".join(str(x) for x in rec.selected_client_ids),
                                        "|".join(str(x) for x in rec.unselected_client_ids),
                                        rec.D_t,
                                        rec.tau_max_used,
                                        rec.objective_value,
                                        rec.candidate_term_sum,
                                        rec.beta_ones,
                                        rec.beta_zeros,
                                        "|".join(str(x) for x in rec.selected_dks),
                                        "|".join(str(x) for x in rec.selected_qs),
                                        "|".join(str(x) for x in rec.selected_taus),
                                        "|".join(str(x) for x in rec.unselected_qs),
                                        "|".join(str(x) for x in rec.unselected_taus),
                                        policy_params_str,
                                        round_wall_time,
                                    ]
                                )
                else:
                    sel_obj_p2 = sel_D_t = cand_term_sum = 0.0
                    sel_tau_max_used = int(ctrl.tau_max_t)
                    beta_ones = beta_zeros = 0
                selected_client_count_rm = 0
                unselected_client_count_rm = 0
                if candidate_set is not None and policy_decision is not None:
                    candidate_clients_rm = set(int(it.client_id) for it in candidate_set.items)
                    selected_clients_rm = set(
                        int(x) for x in (policy_decision.selected_client_ids or [])
                    )
                    selected_client_count_rm = len(selected_clients_rm)
                    unselected_client_count_rm = max(
                        len(candidate_clients_rm) - selected_client_count_rm, 0
                    )

                # === update_id lifecycle 追踪（骨架版子集聚合验证用）===
                aggregated_update_ids_str = ""
                carried_over_update_ids_str = ""
                dropped_update_ids_str = ""
                buffer_update_ids_after_round_str = "|".join(
                    str(getattr(m, "update_id", ""))
                    for (m, _s) in agg.buffer
                    if getattr(m, "update_id", "")
                )
                if (
                    candidate_set is not None
                    and policy_decision is not None
                    and policy_decision.selected_indices is not None
                ):
                    sel_set = set(policy_decision.selected_indices)
                    candidate_set_round_before_agg_approx = int(
                        candidate_set.logical_round
                    )
                    aggregated_update_ids = [
                        str(it.update_id)
                        for idx, it in enumerate(candidate_set.items)
                        if idx in sel_set and getattr(it, "update_id", "")
                    ]
                    carried_over_update_ids = [
                        str(it.update_id)
                        for idx, it in enumerate(candidate_set.items)
                        if idx not in sel_set
                        and int(getattr(it, "entered_buffer_round", -1)) >= 0
                        and int(getattr(it, "entered_buffer_round", -1))
                        < candidate_set_round_before_agg_approx
                        and getattr(it, "update_id", "")
                    ]
                    aggregated_update_ids_str = "|".join(aggregated_update_ids)
                    carried_over_update_ids_str = "|".join(carried_over_update_ids)

                with open(round_metrics_path, "a", newline="", encoding="utf-8") as rf:
                    csv.writer(rf).writerow(
                        [
                            flush_round_id,
                            gs_before,
                            global_step_after,
                            int(queue_extras["queue_update_applied"]),
                            candidate_ids_str,
                            step_result.selected_count,
                            last_candidate_count,
                            step_result.remaining_buffer_count,
                            unselected_count_rm,
                            flush_reason_str or "",
                            snap.avg_staleness,
                            snap.max_staleness,
                            round_wall_time,
                            ids_str,
                            unsel_ids_str,
                            sel_base_steps_str,
                            avg_sel_dk,
                            max_sel_dk,
                            min_sel_dk,
                            avg_cand_dk,
                            policy_cfg["type"],
                            avg_sel_q,
                            max_sel_q,
                            min_sel_q,
                            avg_cand_q,
                            avg_sel_score,
                            max_sel_score,
                            min_sel_score,
                            policy_params_str,
                            queue_extras["avg_candidate_q_before"],
                            queue_extras["avg_selected_q_before"],
                            queue_extras["avg_unselected_q_before"],
                            queue_extras["avg_candidate_q_after"],
                            queue_extras["avg_selected_q_after"],
                            queue_extras["avg_unselected_q_after"],
                            queue_extras["queue_rule_version"],
                            sel_obj,
                            sel_time_term,
                            sel_queue_term,
                            sel_obj_p2,
                            sel_D_t,
                            sel_tau_max_used,
                            cand_term_sum,
                            beta_ones,
                            beta_zeros,
                            selected_client_count_rm,
                            unselected_client_count_rm,
                            eval_prefix_cnt,
                            sel_prefix_sz,
                            queue_extras["candidate_queue_sum"],
                            queue_extras["unselected_queue_sum"],
                            queue_extras["q_delta_selected_avg"],
                            queue_extras["q_delta_unselected_avg"],
                            aggregated_update_ids_str,
                            carried_over_update_ids_str,
                            dropped_update_ids_str,
                            buffer_update_ids_after_round_str,
                        ]
                    )
            cs2 = client_states[msg.client_id]
            beta_k = 1 if triggered_flush else 0
            max_client_q = max(
                max_client_q, *(float(c.virtual_queue) for c in client_states.values())
            )

            print(
                f"[QUEUE] client_id={msg.client_id} q_now={float(cs2.virtual_queue):.6f} "
                f"event_applied={int(cfg['async_mode']=='buffered' and applied and step_result.flush_snapshot is not None)} "
                f"beta_k={int(beta_k)}"
            )

            test_acc_row = last_test_acc
            test_loss_row = last_test_loss
            if applied and (global_step_after % cfg["eval_every"] == 0):
                with lock:
                    model.load_state_dict(shared["global_state"], strict=True)
                model.to(device)
                test_acc_row, test_loss_row = evaluate(model, test_loader, device)
                last_test_acc = test_acc_row
                last_test_loss = test_loss_row
                print(f"[EVAL] global_step={global_step_after} test_acc={test_acc_row:.4f} test_loss={test_loss_row:.4f}")

            wall_time = now_s() - start_t

            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        recv_step,
                        global_step_after,
                        wall_time,
                        msg.client_id,
                        int(msg.base_step),
                        int(staleness),
                        int(accepted),
                        int(applied),
                        int(triggered_flush),
                        flush_reason_str,
                        int(dropped_by_cutoff),
                        buffer_len_before,
                        buffer_len_after,
                        metadata_buffer_len,
                        int(ctrl.buffer_target_t),
                        int(ctrl.tau_max_t),
                        ctrl.delay_score,
                        ctrl.heter_score,
                        ctrl.staleness_score,
                        msg.train_loss if msg.train_loss is not None else "",
                        num_samples,
                        compute_time,
                        upload_delay,
                        delta_norm,
                        float(cs2.virtual_queue),
                        int(beta_k),
                        test_acc_row,
                        test_loss_row,
                    ]
                )

        for p in procs:
            p.join(timeout=2.0)

        total_wall = now_s() - start_t
        n = recv_step
        summary = {
            "total_wall_time": total_wall,
            "total_received_updates": n,
            "total_applied_steps": int(shared["global_step"]),
            "final_accuracy": last_test_acc,
            "avg_staleness": (staleness_sum / n) if n else 0.0,
            "max_staleness": staleness_max,
            "accept_ratio": (accepted_count / n) if n else 0.0,
            # buffered 模式下为“真实 flush 次数”；与 per-row triggered_flush 不同，后者只看当条消息是否触发当次应用。
            "flush_count": flush_count,
            "avg_flush_size": (flush_size_sum / flush_count) if flush_count else 0.0,
            "avg_compute_time": (compute_time_sum / n) if n else 0.0,
            "avg_upload_delay": (upload_delay_sum / n) if n else 0.0,
            "max_client_q": max_client_q,
            "run_dir": str(run_dir),
            "metric_definitions": {
                "triggered_flush": (
                    "本条接收消息在当前处理时刻是否触发了一次全局模型应用："
                    "immediate 下 accepted 且 applied 为 1；buffered 下仅触发本次 flush 的当前消息为 1，"
                    "此前已在缓冲中的消息在该步仍为 0（不回填）。"
                ),
                "beta_k": (
                    "与 triggered_flush 同义写入 CSV：是否因到达当下触发应用，"
                    "不是“是否最终被某次 flush 吸收”。"
                ),
                "virtual_queue_Q_k": (
                    "当前工程语义按 logical_round（真实聚合事件）更新，仅更新本轮 candidate clients："
                    "selected(beta=1) 与 unselected(beta=0) 区分处理；"
                    "非 candidate 客户端本轮保持不变。"
                ),
            },
        }

        with open(run_dir / "summary.json", "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2, ensure_ascii=False)

        print(f"[DONE] run_dir={run_dir}")
        print(
            f"received={n}, applied_updates={applied_updates}, "
            f"final_global_step={int(shared['global_step'])}"
        )
    finally:
        sys.stdout = _orig_stdout
        _debug_fp.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
