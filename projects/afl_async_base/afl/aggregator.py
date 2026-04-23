from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import math
import torch
from .utils import state_dict_add_inplace
from .scafl_types import AggregationCandidateSet, PendingUpdate, PolicyDecision
from .scafl_policy import LegacyFullBufferPolicy, compute_d_k


@dataclass
class FlushSnapshot:
    """
    Observation-only snapshot of agg.buffer taken immediately before a buffered flush
    (after append, before weighted aggregation). Does not affect aggregation math.
    """

    num_updates: int
    avg_staleness: float
    max_staleness: int
    min_staleness: int
    avg_compute_time: float
    avg_upload_delay: float
    total_samples: int


@dataclass
class UpdateMsg:
    client_id: int
    base_step: int
    delta: Dict[str, torch.Tensor]

    # ===== Dynamic state info for server-side collection (used by "first innovation point") =====
    # Default values are provided so existing call sites can omit these fields.
    num_samples: int = 0
    local_epochs: int = 0
    train_started_at: float = 0.0
    train_finished_at: float = 0.0
    sent_at: float = 0.0
    recv_at: float | None = None
    train_loss: float | None = None
    # SC-AFL skeleton/debug：稳定的更新标识与其进入 buffer 的 logical_round。
    # 由 server 在消息进入 Aggregator 缓冲区前生成/填充。
    update_id: str = ""
    entered_buffer_round: int = -1


@dataclass
class StepResult:
    """
    Lightweight return type for Aggregator.step().

    triggered_flush: whether *this* message, at this processing step, triggered one
    global model apply (immediate apply or buffered flush). It is not retroactive for
    older buffered rows (buffered peers stay 0 on the flush step).

    Cases:
    - rejected: accepted=False, triggered_flush=False
    - accepted, still only in buffer (buffered) / N/A (immediate): applied=False,
      triggered_flush=False
    - accepted, global apply now: applied=True, triggered_flush=True
    """

    applied: bool
    accepted: bool
    triggered_flush: bool
    flush_reason: Optional[str] = None
    # Populated only for buffered mode when a real flush occurs; observation only.
    flush_snapshot: Optional[FlushSnapshot] = None
    flushed_count: int = 0
    selected_count: int = 0
    remaining_buffer_count: int = 0
    policy_logical_round: Optional[int] = None


class Aggregator:
    def __init__(
        self,
        server_lr: float,
        async_mode: str = "buffered",
        buffer_size: int = 8,
        staleness_weight: str = "exp",
        staleness_lambda: float = 0.2,
        staleness_cutoff: int = 8,
    ):
        self.server_lr = server_lr
        self.async_mode = async_mode
        self.buffer_size = buffer_size
        self.staleness_weight = staleness_weight
        self.staleness_lambda = staleness_lambda
        self.staleness_cutoff = staleness_cutoff
        self.buffer: List[Tuple[UpdateMsg, int]] = []

    def _weight(self, s: int):
        if self.staleness_weight == "none":
            return 1.0
        if self.staleness_weight == "inv":
            return 1.0 / (1.0 + s)
        if self.staleness_weight == "exp":
            return math.exp(-self.staleness_lambda * s)
        raise ValueError("Unknown staleness_weight")

    def _accept(self, s: int):
        return self.staleness_cutoff < 0 or s <= self.staleness_cutoff

    def _pairs_to_pending(self, pairs: List[Tuple[UpdateMsg, int]]) -> List[PendingUpdate]:
        """将 (UpdateMsg, staleness) 列表转为 PendingUpdate（d_k 为占位定义）。"""
        out: List[PendingUpdate] = []
        for m, s in pairs:
            ct = max(0.0, float(m.train_finished_at - m.train_started_at))
            ra = m.recv_at
            ud = max(0.0, float(ra - m.sent_at)) if ra is not None else 0.0
            arrival = float(ra) if ra is not None else 0.0
            d_k = compute_d_k(ct, ud)
            out.append(
                PendingUpdate(
                    msg=m,
                    staleness=int(s),
                    compute_time=ct,
                    upload_delay=ud,
                    arrival_time=arrival,
                    client_id=int(m.client_id),
                    base_step=int(m.base_step),
                    num_samples=int(m.num_samples),
                    d_k=d_k,
                    update_id=str(getattr(m, "update_id", "")),
                    entered_buffer_round=int(
                        getattr(m, "entered_buffer_round", -1)
                    ),
                )
            )
        return out

    def preview_aggregation_candidate_set(
        self,
        msg: UpdateMsg,
        staleness: int,
        logical_round: int,
        global_step: int,
    ) -> AggregationCandidateSet:
        """
        在未 append 前构造「旧 buffer + 本条」候选集，供策略 decide 使用。
        须与随后 step() 内先 append 再决策的状态一致。
        """
        virtual = list(self.buffer) + [(msg, staleness)]
        items = self._pairs_to_pending(virtual)
        return AggregationCandidateSet(
            items=items,
            source="buffer_plus_incoming",
            logical_round=logical_round,
            global_step=global_step,
        )

    def _build_legacy_decision_after_append(
        self,
        logical_round: int,
        global_step: int,
        target_size: int,
        tau_max_override: Optional[int],
    ) -> PolicyDecision:
        """当前兼容：append 后 buffer 与 preview 一致，用 Legacy 生成决策。"""
        items = self._pairs_to_pending(self.buffer)
        cs = AggregationCandidateSet(
            items=items,
            source="buffer_plus_incoming",
            logical_round=logical_round,
            global_step=global_step,
        )
        return LegacyFullBufferPolicy().decide(
            cs,
            None,
            target_size=target_size,
            tau_max_override=tau_max_override,
        )

    def step(
        self,
        global_state,
        msg: UpdateMsg,
        staleness: int,
        buffer_target_override: int | None = None,
        tau_max_override: int | None = None,
        policy_decision: PolicyDecision | None = None,
        logical_round: int = 0,
        global_step: int = 0,
    ) -> StepResult:
        if not self._accept(staleness):
            print(f"[AGG] drop update from client {msg.client_id}, staleness={staleness}")
            return StepResult(
                applied=False,
                accepted=False,
                triggered_flush=False,
                flush_reason=None,
                flush_snapshot=None,
            )

        if self.async_mode == "immediate":
            w = max(self._weight(staleness), 0.05)
            print(f"[AGG] apply immediate update, w={w:.3f}")
            state_dict_add_inplace(global_state, msg.delta, self.server_lr * w)
            return StepResult(
                applied=True,
                accepted=True,
                triggered_flush=True,
                flush_reason=None,
                flush_snapshot=None,
            )

        self.buffer.append((msg, staleness))
        target_size = (
            buffer_target_override if buffer_target_override is not None else self.buffer_size
        )
        max_buffer_staleness = max(s for _, s in self.buffer) if self.buffer else 0
        print(
            f"[AGG] buffer update {len(self.buffer)}/{target_size} max_staleness={max_buffer_staleness}"
        )

        if policy_decision is None:
            policy_decision = self._build_legacy_decision_after_append(
                logical_round=logical_round,
                global_step=global_step,
                target_size=target_size,
                tau_max_override=tau_max_override,
            )

        if not policy_decision.should_flush:
            return StepResult(
                applied=False,
                accepted=True,
                triggered_flush=False,
                flush_reason=None,
                flush_snapshot=None,
            )

        buf_len = len(self.buffer)
        raw_sel = policy_decision.selected_indices
        if raw_sel is None or len(raw_sel) == 0:
            selected_indices = list(range(buf_len))
        else:
            selected_indices = list(raw_sel)

        valid: Set[int] = set(range(buf_len))
        if not set(selected_indices).issubset(valid):
            print(
                f"[AGG] invalid selected_indices, fallback to full buffer: {selected_indices!r}"
            )
            selected_indices = list(range(buf_len))

        # 去重保序（策略应给唯一索引；若重复则保留首次出现顺序）
        seen: Set[int] = set()
        uniq: List[int] = []
        for i in selected_indices:
            if i not in seen:
                seen.add(i)
                uniq.append(i)
        selected_indices = uniq

        last_idx = buf_len - 1
        triggered_flush = last_idx in set(selected_indices)

        selected_pairs = [self.buffer[i] for i in selected_indices]
        stalenesses = [int(s) for _, s in selected_pairs]
        n_u = len(stalenesses)
        computes: List[float] = []
        uploads: List[float] = []
        total_samples = 0
        for m, _s in selected_pairs:
            ct = max(0.0, float(m.train_finished_at - m.train_started_at))
            computes.append(ct)
            ra = m.recv_at
            if ra is not None:
                uploads.append(max(0.0, float(ra - m.sent_at)))
            else:
                uploads.append(0.0)
            total_samples += int(m.num_samples)

        flush_snapshot = FlushSnapshot(
            num_updates=n_u,
            avg_staleness=float(sum(stalenesses) / n_u) if n_u else 0.0,
            max_staleness=max(stalenesses) if n_u else 0,
            min_staleness=min(stalenesses) if n_u else 0,
            avg_compute_time=float(sum(computes) / n_u) if n_u else 0.0,
            avg_upload_delay=float(sum(uploads) / n_u) if n_u else 0.0,
            total_samples=int(total_samples),
        )

        keys = global_state.keys()
        agg = {k: torch.zeros_like(global_state[k]) for k in keys}
        wsum = 0.0

        for m, s in selected_pairs:
            w = max(self._weight(s), 0.05)
            wsum += w
            for k in keys:
                agg[k].add_(m.delta[k], alpha=w)

        for k in keys:
            agg[k].div_(wsum)

        print("[AGG] apply buffered update (subset-capable)")
        state_dict_add_inplace(global_state, agg, self.server_lr)

        for i in sorted(selected_indices, reverse=True):
            del self.buffer[i]

        remaining = len(self.buffer)
        flush_reason = policy_decision.reason or None

        return StepResult(
            applied=True,
            accepted=True,
            triggered_flush=triggered_flush,
            flush_reason=flush_reason,
            flush_snapshot=flush_snapshot,
            flushed_count=n_u,
            selected_count=n_u,
            remaining_buffer_count=remaining,
            policy_logical_round=logical_round,
        )
