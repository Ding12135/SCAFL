"""
SC-AFL 聚合策略（第一版：默认兼容 Legacy 全缓冲 flush）。

后续论文策略（drift-plus-penalty、按 d_k 排序选集）将实现新类并替换 decide() 逻辑；
LegacyFullBufferPolicy 保留为 baseline 与回归对照。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .runtime_state import SystemState
from .scafl_types import (
    AggregationCandidateSet,
    CandidateDecisionRecord,
    PendingUpdate,
    PolicyDecision,
    PrefixEvaluationRecord,
)


def compute_d_k(compute_time: float, upload_delay: float) -> float:
    """
    统一延迟标量 d_k（占位定义，供 PendingUpdate 与各 policy 使用）。

    当前实现：d_k = compute_time + upload_delay（与客户端一次本地训练耗时 + 上传等待一致）。
    论文若定义不同（如仅上行、或加权），在此替换即可；Aggregator 构造 PendingUpdate 时应调用本函数。
    """
    return max(0.0, float(compute_time)) + max(0.0, float(upload_delay))


def parse_policy_config(cfg: Any) -> Dict[str, Any]:
    """从顶层 yaml 读取 policy 段；缺省为 legacy。"""
    raw = cfg.get("policy") if isinstance(cfg, dict) else None
    if not isinstance(raw, dict):
        raw = {}
    ptype = str(raw.get("type", "legacy")).strip().lower()
    return {
        "type": ptype,
        "min_select_size": int(raw.get("min_select_size", 4)),
        "select_size": int(raw.get("select_size", 4)),
        "max_select_size": int(raw.get("max_select_size", 6)),
        "alpha": float(raw.get("alpha", 1.0)),
        "beta": float(raw.get("beta", 1.0)),
        "V": float(raw.get("V", 1.0)),
        "d_zero_eps": float(raw.get("d_zero_eps", 1e-12)),
    }


def make_policy_from_config(cfg: Any) -> Tuple[object, Dict[str, Any]]:
    """
    根据配置实例化策略对象与解析后的参数字典（用于日志 policy_type）。
    """
    pc = parse_policy_config(cfg)
    ptype = pc["type"]
    if ptype == "sorted_subset":
        return (
            SortedSubsetPolicy(
                min_select_size=int(pc["min_select_size"]),
                select_size=int(pc["select_size"]),
            ),
            pc,
        )
    if ptype in ("scafl_skeleton", "scafl"):
        return (
            SCAFLPolicy(
                min_select_size=int(pc["min_select_size"]),
                d_zero_eps=float(pc.get("d_zero_eps", 1e-12)),
                V=float(pc["V"]),
            ),
            pc,
        )
    if ptype == "queue_aware":
        return (
            QueueAwareSubsetPolicy(
                min_select_size=int(pc["min_select_size"]),
                select_size=int(pc["select_size"]),
                alpha=float(pc["alpha"]),
                beta=float(pc["beta"]),
            ),
            pc,
        )
    if ptype == "approx_drift_penalty":
        return (
            ApproxDriftPlusPenaltyPolicy(
                min_select_size=int(pc["min_select_size"]),
                max_select_size=int(pc["max_select_size"]),
                V=float(pc["V"]),
            ),
            pc,
        )
    if ptype == "scafl_p2":
        return (
            SCAFLP2Policy(
                min_select_size=int(pc["min_select_size"]),
                max_select_size=int(pc["max_select_size"]),
                V=float(pc["V"]),
            ),
            pc,
        )
    if ptype == "legacy":
        return LegacyFullBufferPolicy(), pc
    raise ValueError(
        f"Unknown policy.type: {ptype!r} (use 'legacy', 'scafl_skeleton', "
        f"'sorted_subset', 'queue_aware', 'approx_drift_penalty', or 'scafl_p2')"
    )


def format_policy_params_for_log(pc: Dict[str, Any]) -> str:
    """写入 round_metrics policy_params 列；与 yaml 对齐便于复现实验。"""
    t = str(pc.get("type", "legacy"))
    parts = [f"type={t}"]
    if t in ("sorted_subset", "queue_aware"):
        parts.append(f"min_select_size={pc.get('min_select_size', '')}")
        parts.append(f"select_size={pc.get('select_size', '')}")
    if t in ("scafl_skeleton", "scafl"):
        parts.append(f"min_select_size={pc.get('min_select_size', '')}")
        parts.append(f"d_zero_eps={pc.get('d_zero_eps', '')}")
        parts.append(f"V={pc.get('V', '')}")
    if t == "queue_aware":
        parts.append(f"alpha={pc.get('alpha', '')}")
        parts.append(f"beta={pc.get('beta', '')}")
    if t == "approx_drift_penalty":
        parts.append(f"min_select_size={pc.get('min_select_size', '')}")
        parts.append(f"max_select_size={pc.get('max_select_size', '')}")
        parts.append(f"V={pc.get('V', '')}")
    if t == "scafl_p2":
        parts.append(f"min_select_size={pc.get('min_select_size', '')}")
        parts.append(f"max_select_size={pc.get('max_select_size', '')}")
        parts.append(f"V={pc.get('V', '')}")
    return ";".join(parts)


def compute_scafl_p2_objective_for_prefix(
    *,
    items: List[PendingUpdate],
    sorted_prefix_indices: List[int],
    queue_by_client_id: Optional[Dict[int, float]],
    tau_max_used: int,
    V: float,
) -> Tuple[float, float, float, float]:
    """
    SC-AFL P2 对齐工程式（client-level beta 语义）：
      objective = V * D_t + sum_i q_i * ((tau_i + 1) * (1 - beta_i) - tau_max)
    其中 i 遍历当前 candidate clients（按 client_id 去重），beta_i=1 表示该客户端在
    selected prefix 中至少有一条 update 被选中。
    返回 (objective, D_t, candidate_term_sum, unselected_term_sum)。

    工程近似（详见 docs/paper_symbol_mapping.md）：
    - D_t：prefix 上 max(d_k)，d_k = compute_time + upload_delay（单条 pending 的延迟标量）。
    - tau_i：该 client 多条 pending 时取 max(staleness)；staleness 为 server 侧工程计数。
    - tau_max：本轮使用 tau_max_override，缺省则策略内回退为 buffer target_size。
    - Q：virtual_queue 快照 queue_by_client_id[client_id]，递推为工程事件边界版。
    """
    n = len(items)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    sel = set(sorted_prefix_indices)
    D_t = max(float(items[i].d_k) for i in sorted_prefix_indices) if sorted_prefix_indices else 0.0
    candidate_clients: Set[int] = set(int(it.client_id) for it in items)
    selected_clients: Set[int] = set(
        int(items[i].client_id) for i in sorted_prefix_indices if 0 <= i < n
    )
    term_sum = 0.0
    unselected_term_sum = 0.0
    tau_m = int(tau_max_used)
    for cid in candidate_clients:
        q_i = float(queue_by_client_id.get(cid, 0.0)) if queue_by_client_id else 0.0
        beta_i = 1 if cid in selected_clients else 0
        tau_i = max(int(it.staleness) for it in items if int(it.client_id) == cid)
        term_i = q_i * (((tau_i + 1) * (1 - beta_i)) - tau_m)
        term_sum += term_i
        if beta_i == 0:
            unselected_term_sum += term_i
    obj = float(V) * D_t + term_sum
    return obj, D_t, term_sum, unselected_term_sum


def _minmax_norm_list(vals: List[float]) -> List[float]:
    """候选批内 min-max 到 [0,1]；若全相等则全 0（与 SortedSubset 的平局行为一致）。"""
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        return [0.0] * len(vals)
    rng = vmax - vmin
    return [(v - vmin) / rng for v in vals]


def compute_queue_aware_priority_scores(
    items: List[PendingUpdate],
    queue_by_client_id: Optional[Dict[int, float]],
    alpha: float,
    beta: float,
) -> Tuple[List[float], List[float]]:
    """
    QueueAwareSubsetPolicy 与 round_metrics 共用的打分（工程近似，非论文 Lyapunov 控制律）。

    对当前候选批：
      d_norm_i = (d_i - d_min) / (d_max - d_min)  （全相等时为 0）
      q_norm_i = (q_i - q_min) / (q_max - q_min)  （q_i 为 ClientRuntimeState.virtual_queue）
      priority_i = beta * q_norm_i - alpha * d_norm_i

    越大越优先入选（q 大优先缓解 backlog，d 大惩罚慢更新）。
    queue_by_client_id: client_id -> Q_k 当前值；为 None 时 q 全按 0。
    """
    n = len(items)
    ds = [float(it.d_k) for it in items]
    qs: List[float] = []
    for it in items:
        if queue_by_client_id is None:
            qs.append(0.0)
        else:
            qs.append(float(queue_by_client_id.get(int(it.client_id), 0.0)))
    d_n = _minmax_norm_list(ds)
    q_n = _minmax_norm_list(qs)
    a, b = float(alpha), float(beta)
    pri = [b * q_n[i] - a * d_n[i] for i in range(n)]
    return qs, pri


class LegacyFullBufferPolicy:
    """
    复刻改造前 buffered 行为：
    - flush 当且仅当 (len >= target_size) 或 (max_staleness >= tau_max_override)
    - 触发时选中 append 后 buffer 内**全部**下标 0..n-1
    - 不改变 staleness 加权公式（公式在 Aggregator 内）
    """

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = (system_state, queue_by_client_id)  # 当前 Legacy 不使用；留给后续 SC-AFL
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
            )

        max_s = max(pu.staleness for pu in items)
        size_trigger = n >= target_size
        staleness_trigger = (
            tau_max_override is not None and max_s >= int(tau_max_override)
        )
        should_flush = size_trigger or staleness_trigger

        if not should_flush:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="no_trigger",
                drop_unselected=False,
            )

        if size_trigger:
            reason = "size"
        elif staleness_trigger:
            reason = "staleness"
        else:
            reason = "legacy"

        indices = list(range(n))
        return PolicyDecision(
            should_flush=True,
            selected_indices=indices,
            selected_client_ids=[items[i].client_id for i in indices],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason=reason,
            drop_unselected=False,
        )


class SCAFLPolicy:
    """
    SC-AFL P2 objective 驱动最小版（在 skeleton 框架内替换 top-Q）：
    - 不直接更新模型，仅返回聚合子集 M_t；
    - 候选来源为 append 后 buffer；
    - 规则：
        1) n < min_select_size -> 不 flush；
        2) 使用稳定排序键构造 candidate 顺序：按 d_k 升序，再按 client_id/索引；
        3) 枚举 prefix size = 1..n，对每个 prefix 计算工程版 P2 objective；
        4) 选择 objective 最小的 prefix 作为 selected subset（平局取更小 prefix）。
    - 这一步是结构落地（从“全 buffer 聚合”到“子集聚合”），非论文最终控制律。
    """

    def __init__(self, min_select_size: int, d_zero_eps: float = 1e-12, V: float = 1.0):
        self.min_select_size = int(min_select_size)
        # 保留 d_zero_eps 字段仅为兼容旧配置。
        self.d_zero_eps = float(d_zero_eps)
        self.V = float(V)

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = (system_state, tau_max_override, queue_by_client_id)
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
                candidate_decisions=[],
            )
        if n < self.min_select_size:
            recs = [
                CandidateDecisionRecord(
                    client_id=int(it.client_id),
                    update_id=str(getattr(it, "update_id", "")),
                    base_step=int(it.base_step),
                    staleness=int(it.staleness),
                    compute_time=float(it.compute_time),
                    upload_delay=float(it.upload_delay),
                    num_samples=int(it.num_samples),
                    d_k=float(it.d_k),
                    must_select=False,
                    selected=False,
                    decision_reason="below_min_select",
                )
                for it in items
            ]
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="below_min_select",
                drop_unselected=False,
                candidate_decisions=recs,
            )

        # 稳定排序键：先按 d_k 升序，再按 client_id/原始下标。
        ranked = sorted(
            range(n),
            key=lambda i: (float(items[i].d_k), int(items[i].client_id), i),
        )
        tau_used = int(tau_max_override) if tau_max_override is not None else int(target_size)
        best_j = 1
        best_obj = float("inf")
        best_D = 0.0
        best_term = 0.0
        prefix_records: List[PrefixEvaluationRecord] = []

        for j in range(1, n + 1):
            sel = ranked[:j]
            obj, D_t, term_sum, _ = compute_scafl_p2_objective_for_prefix(
                items=items,
                sorted_prefix_indices=sel,
                queue_by_client_id=queue_by_client_id,
                tau_max_used=tau_used,
                V=self.V,
            )
            sel_set = set(sel)
            sel_ids = [int(items[i].client_id) for i in sel]
            unsel_ids = [int(items[i].client_id) for i in range(n) if i not in sel_set]
            sel_dks = [float(items[i].d_k) for i in sel]
            sel_qs = [
                float(queue_by_client_id.get(int(items[i].client_id), 0.0))
                if queue_by_client_id
                else 0.0
                for i in sel
            ]
            sel_taus = [int(items[i].staleness) for i in sel]
            unsel_qs = [
                float(queue_by_client_id.get(int(items[i].client_id), 0.0))
                if queue_by_client_id
                else 0.0
                for i in range(n)
                if i not in sel_set
            ]
            unsel_taus = [int(items[i].staleness) for i in range(n) if i not in sel_set]
            beta_ones = len(set(sel_ids))
            beta_zeros = len(set(int(it.client_id) for it in items)) - beta_ones
            prefix_records.append(
                PrefixEvaluationRecord(
                    prefix_size=int(j),
                    candidate_count=int(n),
                    selected_indices=[int(x) for x in sel],
                    selected_client_ids=sel_ids,
                    unselected_client_ids=unsel_ids,
                    D_t=float(D_t),
                    tau_max_used=int(tau_used),
                    candidate_term_sum=float(term_sum),
                    objective_value=float(obj),
                    beta_ones=int(beta_ones),
                    beta_zeros=int(max(beta_zeros, 0)),
                    selected_dks=sel_dks,
                    selected_qs=sel_qs,
                    selected_taus=sel_taus,
                    unselected_qs=unsel_qs,
                    unselected_taus=unsel_taus,
                )
            )
            if obj < best_obj - 1e-12 or (abs(obj - best_obj) <= 1e-12 and j < best_j):
                best_j = j
                best_obj = obj
                best_D = D_t
                best_term = term_sum

        selected = ranked[:best_j]
        selected_set = set(selected)

        recs: List[CandidateDecisionRecord] = []
        for i, it in enumerate(items):
            is_sel = i in selected_set
            if is_sel:
                reason = "p2_prefix_selected"
            else:
                reason = "p2_prefix_unselected"
            recs.append(
                CandidateDecisionRecord(
                    client_id=int(it.client_id),
                    update_id=str(getattr(it, "update_id", "")),
                    base_step=int(it.base_step),
                    staleness=int(it.staleness),
                    compute_time=float(it.compute_time),
                    upload_delay=float(it.upload_delay),
                    num_samples=int(it.num_samples),
                    d_k=float(it.d_k),
                    must_select=False,
                    selected=is_sel,
                    decision_reason=reason,
                )
            )

        return PolicyDecision(
            should_flush=True,
            selected_indices=selected,
            selected_client_ids=[int(items[i].client_id) for i in selected],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason="scafl_p2_prefix_subset",
            drop_unselected=False,
            objective_value=float(best_obj),
            time_term=float(best_D),
            queue_term=float(best_term),
            selected_prefix_size=int(best_j),
            evaluated_prefix_count=int(n),
            prefix_evaluations=prefix_records,
            candidate_decisions=recs,
        )


class SortedSubsetPolicy:
    """
    SC-AFL 结构近似版 / MVP（非论文最终控制律）：

    - 候选集 items 的下标 i 与 append 后 buffer 中下标一一对应（0..n-1）。
    - 若 n < min_select_size：不 flush（should_flush=False）。
    - 若 n >= min_select_size：flush，并令 m = min(select_size, n)。
    - 按 d_k 升序排序，取 d_k 最小的 m 个**原始下标**作为 selected_indices（同 d_k 时按下标升序打破平局）。
    - 未选中下标对应更新保留在 buffer（由 Aggregator 删除已选行实现）。

    注意：本策略**不使用** Legacy 的 buffer_size / tau_max 触发条件；target_size/tau_max_override 仅作占位传入。
    """

    def __init__(self, min_select_size: int, select_size: int):
        self.min_select_size = int(min_select_size)
        self.select_size = int(select_size)

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = (system_state, target_size, tau_max_override, queue_by_client_id)
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
            )

        if n < self.min_select_size:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="below_min_select",
                drop_unselected=False,
            )

        m = min(self.select_size, n)
        # 按 d_k 升序，同值按下标升序，保证 selected_indices 是 buffer 中的真实下标
        order = sorted(range(n), key=lambda i: (float(items[i].d_k), i))
        selected_indices = order[:m]

        return PolicyDecision(
            should_flush=True,
            selected_indices=selected_indices,
            selected_client_ids=[items[i].client_id for i in selected_indices],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason="sorted_subset_top_m",
            drop_unselected=False,
        )


class QueueAwareSubsetPolicy:
    """
    Queue + delay 联合启发式子集策略（工程近似版，非论文 drift-plus-penalty 最终控制律）。

    目的：验证 virtual_queue（Q_k 工程近似）参与聚合决策的整条链路；后续可替换为定理化目标。

    - 与 SortedSubset 相同：n < min_select_size 不 flush；否则 m = min(select_size, n)。
    - 入选顺序由 compute_queue_aware_priority_scores 的 priority 降序决定（同 priority 按下标升序）。
    - 须由 server 传入 queue_by_client_id（client_id -> virtual_queue），勿在策略内读全局状态。
    """

    def __init__(
        self,
        min_select_size: int,
        select_size: int,
        alpha: float,
        beta: float,
    ):
        self.min_select_size = int(min_select_size)
        self.select_size = int(select_size)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = (system_state, target_size, tau_max_override)
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
            )

        if n < self.min_select_size:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="below_min_select",
                drop_unselected=False,
            )

        _, pri = compute_queue_aware_priority_scores(
            items, queue_by_client_id, self.alpha, self.beta
        )
        m = min(self.select_size, n)
        order = sorted(range(n), key=lambda i: (-pri[i], i))
        selected_indices = order[:m]

        return PolicyDecision(
            should_flush=True,
            selected_indices=selected_indices,
            selected_client_ids=[items[i].client_id for i in selected_indices],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason="queue_aware_top_m",
            drop_unselected=False,
        )


class ApproxDriftPlusPenaltyPolicy:
    """
    近似 drift-plus-penalty 子集策略（工程版，非论文严格 Lyapunov 控制律）。

    与 QueueAwareSubsetPolicy 的区别：不对单点打分取 top-m，而是对**若干候选子集**
    计算标量 objective，再取 objective 最小者（结构上接近「对 M_t 优化目标」）。

    枚举范围（避免组合爆炸）：先将候选按 (d_k 升序, buffer 下标升序) 排序，仅枚举
    **前缀子集** M_1..M_{m_max}，其中 m_max = min(max_select_size, n)。

    ---------------------------------------------------------------------------
    第一版 objective（注释即规范）：

      对前缀大小 j（1 <= j <= m_max），令 S_j 为排序后前 j 个候选的 buffer 下标集合。

      time_term_j = max_{i in S_j} d_k(i)
          解释：子集中最慢更新耗时上界，类比一轮 wall-time / D_t 惩罚项。

      queue_term_j = sum_{i notin S_j} q_k(i)
          其中 q_k(i) = queue_by_client_id[client_id]，即当前 virtual_queue（Q_k 工程近似）。
          解释：未被本轮聚合的候选，其队列压力留在系统中，作为「漂移/积压」惩罚。

      objective_j = V * time_term_j + queue_term_j
          V 为可配置权重，对标 Lyapunov 中「时间项 vs 队列项」的 trade-off（仅为形态近似）。

    选择：objective_j 最小的 j；平局取更小 j，再平局保持排序稳定性。

    flush 门限：仅当 n >= min_select_size 时 should_flush=True；否则不 flush。
    ---------------------------------------------------------------------------
    """

    def __init__(self, min_select_size: int, max_select_size: int, V: float):
        self.min_select_size = int(min_select_size)
        self.max_select_size = int(max_select_size)
        self.V = float(V)

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = (system_state, target_size, tau_max_override)
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
            )

        if n < self.min_select_size:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="below_min_select",
                drop_unselected=False,
            )

        ranked = sorted(range(n), key=lambda i: (float(items[i].d_k), i))
        m_max = min(self.max_select_size, n)
        evaluated = m_max

        ds = [float(items[i].d_k) for i in range(n)]
        qs = [
            float(queue_by_client_id.get(int(items[i].client_id), 0.0))
            if queue_by_client_id
            else 0.0
            for i in range(n)
        ]

        best_j: Optional[int] = None
        best_time = 0.0
        best_queue = 0.0
        best_obj = float("inf")

        for j in range(1, m_max + 1):
            prefix_idx = ranked[:j]
            time_t = max(ds[i] for i in prefix_idx)
            in_prefix = set(prefix_idx)
            queue_t = sum(qs[i] for i in range(n) if i not in in_prefix)
            obj = self.V * time_t + queue_t
            if best_j is None or obj < best_obj - 1e-12 or (
                abs(obj - best_obj) <= 1e-12 and j < best_j
            ):
                best_j = j
                best_obj = obj
                best_time = time_t
                best_queue = queue_t

        assert best_j is not None
        selected_indices = ranked[:best_j]

        return PolicyDecision(
            should_flush=True,
            selected_indices=selected_indices,
            selected_client_ids=[items[i].client_id for i in selected_indices],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason="approx_drift_penalty_prefix",
            drop_unselected=False,
            objective_value=float(best_obj),
            time_term=float(best_time),
            queue_term=float(best_queue),
            selected_prefix_size=int(best_j),
            evaluated_prefix_count=int(evaluated),
        )


class SCAFLP2Policy:
    """
    SC-AFL P2 对齐版策略（第一版，工程实现，不是论文严格复现版）。

    目标形式对齐论文 P2：
      min over beta(t):
        V * D_t + sum_k Q_k(t) * ((tau_k(t) + 1) * (1 - beta_k^t) - tau_max)

    工程映射：
    - Q_k(t): queue_by_client_id[client_id]（决策前 virtual_queue）
    - tau_k(t): candidate_set 中该更新的 staleness
    - beta_k^t: selected=1, unselected=0（仅对 candidate_set 定义）
    - D_t: selected prefix 的 max(d_k)
    - tau_max: 使用本轮 tau_max_override（若缺省则回退 target_size，仅作稳定回退）

    搜索方式：先按 d_k 升序排序，仅枚举 prefix S_1..S_m（m=min(max_select_size,n)）。
    """

    def __init__(self, min_select_size: int, max_select_size: int, V: float):
        self.min_select_size = int(min_select_size)
        self.max_select_size = int(max_select_size)
        self.V = float(V)

    def decide(
        self,
        candidates: AggregationCandidateSet,
        system_state: Optional[SystemState] = None,
        *,
        target_size: int,
        tau_max_override: Optional[int],
        queue_by_client_id: Optional[Dict[int, float]] = None,
    ) -> PolicyDecision:
        _ = system_state
        items = candidates.items
        n = len(items)
        if n == 0:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="empty_candidates",
                drop_unselected=False,
            )
        if n < self.min_select_size:
            return PolicyDecision(
                should_flush=False,
                selected_indices=None,
                selected_client_ids=None,
                tau_max=tau_max_override,
                buffer_target=target_size,
                reason="below_min_select",
                drop_unselected=False,
            )

        ranked = sorted(range(n), key=lambda i: (float(items[i].d_k), i))
        m_max = min(self.max_select_size, n)
        tau_used = int(tau_max_override) if tau_max_override is not None else int(target_size)

        best_j = 1
        best_obj = float("inf")
        best_D = 0.0
        best_term = 0.0
        prefix_records: List[PrefixEvaluationRecord] = []

        for j in range(1, m_max + 1):
            sel = ranked[:j]
            obj, D_t, term_sum, _ = compute_scafl_p2_objective_for_prefix(
                items=items,
                sorted_prefix_indices=sel,
                queue_by_client_id=queue_by_client_id,
                tau_max_used=tau_used,
                V=self.V,
            )
            sel_set = set(sel)
            sel_ids = [int(items[i].client_id) for i in sel]
            unsel_ids = [int(items[i].client_id) for i in range(n) if i not in sel_set]
            sel_dks = [float(items[i].d_k) for i in sel]
            sel_qs = [
                float(queue_by_client_id.get(int(items[i].client_id), 0.0))
                if queue_by_client_id
                else 0.0
                for i in sel
            ]
            sel_taus = [int(items[i].staleness) for i in sel]
            unsel_qs = [
                float(queue_by_client_id.get(int(items[i].client_id), 0.0))
                if queue_by_client_id
                else 0.0
                for i in range(n)
                if i not in sel_set
            ]
            unsel_taus = [int(items[i].staleness) for i in range(n) if i not in sel_set]
            prefix_records.append(
                PrefixEvaluationRecord(
                    prefix_size=int(j),
                    candidate_count=int(n),
                    selected_indices=[int(x) for x in sel],
                    selected_client_ids=sel_ids,
                    unselected_client_ids=unsel_ids,
                    D_t=float(D_t),
                    tau_max_used=int(tau_used),
                    candidate_term_sum=float(term_sum),
                    objective_value=float(obj),
                    beta_ones=int(len(sel)),
                    beta_zeros=int(n - len(sel)),
                    selected_dks=sel_dks,
                    selected_qs=sel_qs,
                    selected_taus=sel_taus,
                    unselected_qs=unsel_qs,
                    unselected_taus=unsel_taus,
                )
            )
            if obj < best_obj - 1e-12 or (abs(obj - best_obj) <= 1e-12 and j < best_j):
                best_j = j
                best_obj = obj
                best_D = D_t
                best_term = term_sum

        selected_indices = ranked[:best_j]
        return PolicyDecision(
            should_flush=True,
            selected_indices=selected_indices,
            selected_client_ids=[items[i].client_id for i in selected_indices],
            tau_max=tau_max_override,
            buffer_target=target_size,
            reason="scafl_p2_prefix",
            drop_unselected=False,
            objective_value=float(best_obj),
            time_term=float(best_D),
            queue_term=float(best_term),
            selected_prefix_size=int(best_j),
            evaluated_prefix_count=int(m_max),
            prefix_evaluations=prefix_records,
        )
