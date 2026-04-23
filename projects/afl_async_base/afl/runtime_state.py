from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass
class ClientRuntimeState:
    """
    Server 侧对某个客户端的运行时估计状态，用于后续动态控制的观测量。

    注意：
    - 本结构只保存“观测/估计”信息，不参与训练算法本身。
    - estimated_* 字段在每次 server 收到来自该客户端的更新时刷新。
    """

    client_id: int

    # 最近一次上报该客户端更新时，对应的基准步/接收步
    last_base_step: int
    last_recv_step: int

    # 到达时的滞后程度（arrival_step - base_step），不对聚合做任何裁剪
    current_staleness: int

    # 训练计算耗时估计（train_finished_at - train_started_at）
    estimated_compute_time: float

    # 上传延迟估计（recv_at - sent_at）
    estimated_upload_delay: float

    # 虚拟队列长度 Q_k（用于 SC-AFL 论文 queue 递推的工程状态）。
    # 该字段在 server 侧“真实 aggregation event / logical_round”之后更新：
    # - candidate 集合内：
    #     - selected: beta_k=1，Q_k 往下（减去 tau_max_t）
    #     - unselected: beta_k=0，Q_k 往上（累积 staleness 压力）
    # - 非 candidate 客户端：本轮不变（工程边界约定）。
    virtual_queue: float


@dataclass
class BufferedUpdate:
    """
    把 UpdateMsg 映射为“更方便服务端采集/统计”的结构。
    该结构包含在 server 端用于计算观测量所需的字段。
    """

    client_id: int
    base_step: int
    arrival_step: int
    staleness: int

    num_samples: int
    train_loss: Optional[float]

    compute_time: float
    upload_delay: float

    # 模型参数增量
    delta: Dict[str, torch.Tensor]


@dataclass
class SystemState:
    """
    全局系统状态（server 端观测量汇总），用于动态控制策略的输入。
    本轮需求约束：只打印，不改变训练/聚合逻辑。
    """

    avg_upload_delay: float
    avg_compute_time: float
    compute_heterogeneity: float

    buffer_size: int
    avg_buffer_staleness: float
    max_buffer_staleness: int

