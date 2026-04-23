"""
SC-AFL 策略与聚合决策的轻量数据结构（基础设施层）。

当前为「兼容现有 FedBuff 式异步聚合」的接口占位；后续论文级 SC-AFL 的
drift-plus-penalty、按 d_k 子集搜索等将替换/扩展 PolicyDecision 的生成逻辑，
本文件字段语义应尽量保持稳定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class PendingUpdate:
    """
    单条待聚合更新在策略层的视图（与 Aggregator 内部 (UpdateMsg, staleness) 对应）。

    msg:
        类型为 aggregator.UpdateMsg（为避免循环导入此处标注为 Any）。
    d_k:
        第一版占位：d_k = compute_time + upload_delay（论文中定义可后续替换）。
    arrival_time:
        使用服务端收到时刻（与 msg.recv_at 一致）。
    """

    msg: Any
    staleness: int
    compute_time: float
    upload_delay: float
    arrival_time: float
    client_id: int
    base_step: int
    num_samples: int
    d_k: float
    # 稳定唯一标识：用于跨多 logical_round 追踪该更新是否被选入/删除。
    update_id: str
    # update_id 第一次进入 buffer 的 logical_round（工程语义近似）。
    entered_buffer_round: int
    # SC-AFL skeleton 决策辅助字段（默认占位，策略可覆盖）
    must_select: bool = False
    selected: bool = False
    decision_reason: str = ""


@dataclass
class AggregationCandidateSet:
    """
    一次「可能发生 flush」时的候选集合快照。

    source:
        说明候选来自何处；当前实现为 append 前的「旧 buffer + 本条」虚拟拼接。
    logical_round:
        工程近似：已完成 flush 次数（见 server 注释）；后续可替换为论文 round(t)。
    """

    items: List[PendingUpdate]
    source: str
    logical_round: int
    global_step: int


@dataclass
class PolicyDecision:
    """
    聚合策略输出：后续 SC-AFL 将在此接入 drift-plus-penalty / 子集搜索。

    should_flush:
        是否与当前 Aggregator 的 flush 触发条件一致（由 Legacy 策略复刻）。
    selected_indices:
        相对「append 后 buffer」下标的子集；None 表示不 flush 或未选择。
    drop_unselected:
        True 时未来可丢弃未选中项；当前基础设施阶段默认 False（未选中保留在 buffer）。
    """

    should_flush: bool
    selected_indices: Optional[List[int]]
    selected_client_ids: Optional[List[int]]
    tau_max: Optional[int]
    buffer_target: Optional[int]
    reason: str
    drop_unselected: bool = False
    # 以下由 ApproxDriftPlusPenaltyPolicy 等填充；旧策略保持 None。
    objective_value: Optional[float] = None
    time_term: Optional[float] = None
    queue_term: Optional[float] = None
    selected_prefix_size: Optional[int] = None
    evaluated_prefix_count: Optional[int] = None
    # 仅供 scafl_p2 等“子集枚举策略”写入；其它策略保持 None。
    prefix_evaluations: Optional[List["PrefixEvaluationRecord"]] = None
    # skeleton/candidate-level 诊断（其它策略可留空）
    candidate_decisions: Optional[List["CandidateDecisionRecord"]] = None


@dataclass
class PrefixEvaluationRecord:
    """
    单个 prefix 子集在一次 policy 决策中的评估明细（用于 P2 诊断视图）。
    """

    prefix_size: int
    candidate_count: int
    selected_indices: List[int]
    selected_client_ids: List[int]
    unselected_client_ids: List[int]
    D_t: float
    tau_max_used: int
    candidate_term_sum: float
    objective_value: float
    beta_ones: int
    beta_zeros: int
    selected_dks: List[float]
    selected_qs: List[float]
    selected_taus: List[int]
    unselected_qs: List[float]
    unselected_taus: List[int]


@dataclass
class CandidateDecisionRecord:
    """
    单个 candidate 在某轮 policy 决策后的结果（用于结构化日志）。
    """

    client_id: int
    update_id: str
    base_step: int
    staleness: int
    compute_time: float
    upload_delay: float
    num_samples: int
    d_k: float
    must_select: bool
    selected: bool
    decision_reason: str
