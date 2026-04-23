# 论文符号 ↔ 工程变量 / 日志字段（严格对齐版说明）

本文档描述 **SC-AFL P2 形态** 在当前代码库中的对应关系，并标出 **仍为工程近似** 的部分。不引入新算法，仅固定符号与可复现实验配置。

---

## 1. 可配置：Lyapunov 权重 \(V\)

| 论文 / 文中记法 | 工程配置 | 默认值 | 日志中出现位置 |
|-----------------|----------|--------|----------------|
| \(V\)（时间项 \(V \cdot D_t\) 与队列项的相对权重） | `policy.V`（顶层 YAML `policy:` 段） | `1.0` | `round_metrics.policy_params` 字符串中的 `V=...`；`p2_prefix_diagnostics.policy_params` 同结构 |

**实现说明**

- `parse_policy_config()` 读取 `V`，缺省为 `1.0`。
- **`scafl_p2`**、**`approx_drift_penalty`**：策略类持有 `self.V`。
- **`scafl_skeleton` / `scafl`**：`SCAFLPolicy` 持有 `self.V`，枚举 prefix 时传入 `compute_scafl_p2_objective_for_prefix(..., V=self.V)`（与 `scafl_p2` 一致，可论文对齐调参）。

---

## 2. P2 目标（论文形式 ↔ 工程实现）

论文中用于决策的 P2 形态（符号与 `scafl_policy.compute_scafl_p2_objective_for_prefix` 注释一致）：

\[
\text{objective} = V \cdot D_t + \sum_{k \in \mathcal{C}} Q_k(t)\,\Big( (\tau_k(t)+1)(1-\beta_k^t) - \tau_{\max} \Big)
\]

其中 \(\mathcal{C}\) 为**当前候选批**中的 **client** 集合（client-level \(\beta\)）。

| 论文符号 | 含义（论文意图） | 工程变量 / 计算 | 主要日志字段 |
|----------|------------------|-----------------|--------------|
| \(V\) | 时间 vs 队列权衡 | `policy.V` → 传入目标函数 | 见 §1 |
| \(D_t\) | 与本轮“时间跨度”相关的项 | `max(d_k)` 在 **选中 prefix 的 candidate 行**上取最大；`d_k = compute_time + upload_delay` | `p2_prefix_diagnostics.D_t_prefix`；`round_metrics.selected_D_t`；`selected_objective_p2` 分解中与 \(D_t\) 一致部分见 `selected_time_term`（server 对 P2 策略写入的 `time_term`） |
| \(Q_k(t)\) | 虚拟队列状态 | `ClientRuntimeState.virtual_queue`，决策前快照 `queue_by_client_id[client_id]` | `decision_debug.q_k_pre`；prefix 行中 `selected_qs` / `unselected_qs` |
| \(\tau_k(t)\) | 过时量 | 见 **§3（工程近似）** | `decision_debug.tau_k`（**每条 pending 一行**）；按 client 聚合时在目标里用 **该 client 下多条 pending 的 staleness 取 max** |
| \(\tau_{\max}\) | 阈值 / 上界项 | 本轮 `tau_max_override`（来自 dynamic controller 的 `tau_max_t`）；若缺失则策略内 **回退为 `target_size`**（buffer 目标长度） | `p2_prefix_diagnostics.tau_max_used`；`round_metrics.selected_tau_max_used` |
| \(\beta_k^t\) | 是否服务 client \(k\) | client-level：`1` 若该 client 在选中 prefix 的 **indices 对应行**中至少出现一次 | `selected_beta_ones` / `selected_beta_zeros`（**client 计数**）；`decision_debug.beta_k`（client-level） |

**仍为工程近似（P2 objective 整体）**

- 候选集仅为 **当前 buffer 快照** 的 pending 行，不是全文理想化下的全体客户端。
- **Prefix 搜索**：`scafl_p2` / skeleton 在 **按 \(d_k\) 排序后的序列上枚举前缀** \(S_1,\ldots,S_m\)，不是任意子集上的全局最优 \(\beta\)。
- **`candidate_term_sum`** 与 **`objective_p2`**：与上式一致实现，但 \(\tau\)、\(Q\) 的来源为 **离散事件、工程 staleness**，见 §3–§4。

---

## 3. \(\tau_k\) 与 \(D_t\)：工程近似要点

### 3.1 \(\tau_k(t)\)（staleness）

| 项目 | 说明 |
|------|------|
| **每条 pending 记录** | `PendingUpdate.staleness`，由聚合器在候选集构建时写入，表示 **工程意义上的落后步数/间隔**（离散计数）。 |
| **论文 \(\tau_k(t)\)** | 连续时间或轮次下的“过时”；本实现 **不保证** 与论文同一测度。 |
| **P2 目标中 per-client \(\tau_i\)** | 若同一 client 有多条 pending，取 **`max(staleness)`** 在该 client 的各行上（见 `compute_scafl_p2_objective_for_prefix`）。 |
| **日志** | `decision_debug.tau_k` 为 **行级**；与论文“每客户端一个 \(\tau_k\)”对照时，应对同一 `client_id` 做 **max** 再比。 |

**结论**：\(\tau_k\) **对齐意图**，**不对齐**连续时间论文中的严格定义 → 标为 **工程近似**。

### 3.2 \(D_t\)（时间 / 延迟聚合项）

| 项目 | 说明 |
|------|------|
| **定义** | 对选中 prefix 内所有 candidate **行**的 `d_k` 取 **`max`**。 |
| **`d_k`** | `compute_d_k(compute_time, upload_delay)` → `compute_time + upload_delay`（非负截断）。 |
| **论文** | 常与一轮 wall-time 或调度跨度相关；此处 **显式绑定为“单条更新观测到的计算+上传延迟之和”**。 |

**结论**：\(D_t\) **形态**对应论文“本轮时间惩罚”，**物理含义**为工程延迟上界 → **半论文对齐、半工程近似**。

---

## 4. 队列 \(Q_k\)：工程事件边界版

虚拟队列在 `apply_queue_update_for_aggregation_event`（`server.py`）中按 **aggregation event** 更新：`queue_rule_version=v1_event_candidate_beta`。同一 logical round 内 **按 client 去重** 只更新一次 \(Q_k\)。

**与论文 Lyapunov 推导的差异**：事件边界、仅更新 candidate 子集、\(\tau_{\max}\) 取工程 `tau_max_t` 等，均属 **工程近似**；详见该函数文档字符串。

---

## 5. 快速索引：日志 CSV 列名

| 文件 | 与 P2 / \(V\) 强相关列 |
|------|-------------------------|
| `round_metrics.csv` | `policy_params`（含 `V`），`selected_objective_p2`，`selected_D_t`，`selected_tau_max_used`，`candidate_term_sum`，`selected_beta_ones`，`selected_beta_zeros` |
| `p2_prefix_diagnostics.csv` | `objective_p2`，`D_t_prefix`，`tau_max_used`，`candidate_term_sum`，`beta_ones`，`beta_zeros`，`policy_params` |
| `decision_debug.csv` | `q_k_pre`，`tau_k`，`beta_k`，`term_k`（单行的队列项贡献分解） |

---

## 6. 修订记录

- 引入 **`scafl_skeleton` 的 `V` 与 `scafl_p2` 同源配置**（`policy.V`），便于与论文 P2 权重一致调参。
