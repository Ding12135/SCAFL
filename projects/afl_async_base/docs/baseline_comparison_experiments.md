# Baseline 对比实验说明（用于论文 / 报告）

本文档描述 `projects/afl_async_base` 中**已实现、可作为对照组**的 baseline 设定，便于与创新方法（例如引入端侧状态感知后的方法）做**公平、可复现**的对比。

---

## 1. 实验目的与对照关系

| 角色 | 说明 |
|------|------|
| **Baseline（本文档覆盖）** | 在**不显式使用**端侧 CPU / 内存 / 带宽 / 电量 / 能耗等**资源状态**作为控制输入的前提下，完成异步联邦学习中的**陈旧度约束、缓冲与（可选）子集聚合**。 |
| **创新方法（对照组之外）** | 在相同训练与数据设定下，增加端侧状态观测，并用于动态调整 `τ`、`buffer` 或其它控制律；与 baseline 的**差异应可归因于「状态感知控制」**。 |

对比时应固定：**数据集划分、`seed`、模型、本地 epoch、学习率、客户端数、每客户端更新次数、`server_lr`** 等；仅切换 **yaml 配置**（`async_mode`、`policy`、`dynamic_control`、`staleness_cutoff`、`heterogeneity` 等）。

---

## 2. 符号与工程语义（与代码对齐）

- **`global_step`**：服务器侧全局模型版本计数；客户端 `base_step` 为拉取模型时的 `global_step`。
- **`staleness`**：单条 update 到达时 `max(0, global_step − base_step)`（工程计数）。
- **`τ`（tau）**：陈旧上界；**固定 baseline** 下 `τ = staleness_cutoff`（`dynamic_control.enabled: false`）；**动态 baseline** 下由 `DynamicController` 输出 `tau_max_t`（见 `dynamic_threshold_baseline_hetero.yaml`）。
- **`candidate_set`**：`preview_aggregation_candidate_set` 得到 **旧 buffer（按当前 `global_step` 刷新陈旧度）+ incoming**，供策略与队列诊断使用。
- **`d_k`（工程近似）**：已入 buffer 的旧 update 在候选中 **`d_k = 0`**（表示本轮不再等待其完成）；**incoming** 为 **`compute_time + upload_delay`**（见 `aggregator._pairs_to_pending` 注释）。更细的符号映射见 `docs/paper_symbol_mapping.md`。
- **`Q_k`（virtual queue）**：`ClientRuntimeState.virtual_queue`；在每次真实 flush 后按 `apply_queue_update_for_aggregation_event` 更新（仅 candidate 客户端）。

---

## 3. 已提供的 Baseline 配置一览

以下文件均在 `configs/` 目录；**同一套代码** `python -m afl.server`，通过 `AFL_CONFIG` 切换。

| 配置名 | 文件 | 异步模式 | 陈旧阈值 / 动态控制 | 聚合策略 | 异构时延 |
|--------|------|----------|----------------------|----------|----------|
| **SC-AFL-like（主 baseline）** | `configs/scafl_baseline_hetero.yaml` | `buffered` | 固定 `staleness_cutoff`；`dynamic_control.enabled: false` | `scafl_p2`（P2-like 子集 + 队列） | 可选 `heterogeneity` |
| FedAsync 式 | `configs/fedasync_hetero.yaml` | `immediate` | 固定 cutoff；动态控制关 | `legacy`（占位） | 可选 |
| FedBuff 全缓冲式 | `configs/fedbuff_static_hetero.yaml` | `buffered` | `staleness_cutoff: -1`；动态控制关 | `legacy`（满 buffer flush） | 可选 |
| 固定阈值 + 动态阈值策略 | `configs/static_threshold_fedbuff_hetero.yaml` | `buffered` | 固定 cutoff；动态控制关 | `dynamic_threshold` | 可选 |
| 动态阈值控制器 | `configs/dynamic_threshold_baseline_hetero.yaml` | `buffered` | `dynamic_control.enabled: true` | `dynamic_threshold` | 可选 |
| 动态控制 + 异构（另一对照） | `configs/baseline_dynamic_staleness.yaml` | `buffered` | 动态 `τ` / `buffer_target` | `dynamic_threshold` | 可按需打开 |

**主对比建议**：创新方法 vs **`scafl_baseline_hetero`**；其余 yaml 作为 **FedAsync / FedBuff / 固定阈值 / 全动态阈值** 的横向参照。

---

## 4. SC-AFL-like 主 Baseline 的控制逻辑（摘要）

1. 客户端异步上传 update；服务器计算 **`staleness`**，维护 **buffer**。
2. 构造 **`candidate_set`**；其中 **incoming** 须满足 **`staleness ≤ τ`**，否则丢弃（`drop_incoming_by_tau`）；buffer 内超过 **τ** 的条目在后续步骤中被裁剪。
3. **`tau_used = staleness_cutoff`**（动态控制关闭时）；**不参与**端侧 CPU/内存/带宽/电量决策。
4. **`scafl_p2` policy**：在 **固定 τ** 下，结合 **`Q_k`** 与 **P2-like 目标** 在候选前缀中选择子集；聚合后对 **virtual queue** 更新。
5. 聚合权重仍支持 **`staleness_weight`**（如 `exp`）与 **`staleness_lambda`**。

异构：**仅**通过 `hetero_simulator.py` 增加 **compute / upload 的 wall-time**，以产生到达间隔差异；**不**把「档位 / 因子」写入 policy 或 DynamicController。

---

## 5. 可复现性与批量实验

- **配置种子**：`yaml` 中 `seed`；可用环境变量 **`AFL_SEED`** 覆盖（见 `server.py`）。
- **批量运行**：`scripts/run_baseline_suite.sh`（多 seed × 多配置）。
- **汇总表**：`python tools/summarize_baselines.py --log-root <日志根目录>` → `results/baseline_summary.csv`。
- **曲线图**：`python tools/plot_baseline_curves.py <run_dir> ... --out-dir results/figures`。

每次运行会在 `log_root` 下生成带时间戳的 **`run_dir`**，内含 `config.yaml`、`summary.json`、`metrics.csv`、`flush_metrics.csv`、`round_metrics.csv`、`queue_trace.csv`、`decision_debug.csv`、`events/debug.log` 等，便于写论文「实验设置」与附录。

---

## 6. 撰写对比实验章节时可用的表述要点

1. **任务与模型**：默认 MNIST + `mlp`；亦可 `FashionMNIST` / `EMNIST*` + `mlp`，`CIFAR10` / `CIFAR100` / `SVHN` + `cnn_small`（类别数由 `infer_num_classes(dataset)` 决定，见 `afl/data.py`、`afl/model.py`）。Non-IID 由 `non_iid`、`num_shards` 控制。
2. **Baseline 定义**：固定陈旧上界 **τ**、服务器缓冲、**无**端侧资源状态反馈下的异步聚合；主 baseline 采用 **SC-AFL 思想**的 **队列 + P2-like 子集选择**（`scafl_p2`）。
3. **异构性**：仅时延异构，用于模拟异步环境，**不作为控制输入**。
4. **公平性**：与创新方法共享同一数据路径、同一 `seed` 策略、同一评估间隔 `eval_every`；仅替换配置中的 **控制与策略段**。
5. **报告指标**：最终 / 最佳精度、墙钟时间、`accept_ratio`、flush 次数与平均 flush 规模、`metrics` / `round_metrics` 中的队列与 P2 相关列（若启用）。

---

## 7. 单条运行命令示例

```bash
cd projects/afl_async_base
export PYTHONPATH=.
AFL_CONFIG=configs/scafl_baseline_hetero.yaml python -m afl.server
```

从仓库根：

```bash
export PYTHONPATH=/path/to/SCAFL
AFL_CONFIG=projects/afl_async_base/configs/scafl_baseline_hetero.yaml python -m projects.afl_async_base.afl.server
```

---

*若论文符号与工程字段需一一对应，请与 `docs/paper_symbol_mapping.md` 交叉核对后定稿。*
