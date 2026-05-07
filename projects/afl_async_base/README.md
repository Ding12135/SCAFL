# AFL Async Base (`projects/afl_async_base`)

异步联邦学习实验代码（MNIST + MLP 等），含缓冲聚合、动态/固定陈旧阈值、SC-AFL-like 策略与队列诊断日志。

## SC-AFL-like Baseline

本 baseline 用于**端侧状态创新点之前**的论文对比实验，定位如下：

1. **对比对象**：后续将引入「端侧状态感知」的方法与本 baseline 对照。
2. **异构环境**：可通过 `heterogeneity` 配置在客户端进程里模拟**不同的计算与上传时延**（`hetero_simulator.py`），用于产生异步到达模式；这些量**不作为** DynamicController / policy 的显式状态输入。
3. **控制输入边界**：控制器与策略**不**读取 CPU、内存、带宽、电量、能耗等端侧资源状态；控制依据为 **staleness、virtual queue、candidate set、compute_time、upload_delay、buffer 状态** 等服务器可观测信息。
4. **SC-AFL 思想（工程近似）**：在**固定陈旧上界** `tau_max = cfg["staleness_cutoff"]`（`dynamic_control.enabled: false`）下，维护虚拟队列 `Q_k(t)`，并由 **`scafl_p2` policy** 基于 **P2-like 目标**在每轮从 candidate set 中动态选择聚合子集。
5. **后续扩展**：state-aware 方法可在本 baseline 之上加入端侧状态，并驱动 `tau_max(t)`、`buffer_target(t)` 等（当前仓库中的 `dynamic_control` 路径为独立对照）。

### 运行示例

```bash
cd projects/afl_async_base
export PYTHONPATH=.
AFL_CONFIG=configs/scafl_baseline_hetero.yaml python -m afl.server
```

从仓库根目录（需将仓库根加入 `PYTHONPATH`）：

```bash
export PYTHONPATH=/path/to/SCAFL
AFL_CONFIG=projects/afl_async_base/configs/scafl_baseline_hetero.yaml python -m projects.afl_async_base.afl.server
```

可选环境变量 `AFL_SEED` 覆盖配置中的 `seed`。

### Baseline 批量脚本与汇总

- `scripts/run_baseline_suite.sh`：多 seed、多配置批量运行。
- `tools/summarize_baselines.py`：扫描 `log_root` 下各 `run_dir` 生成 `results/baseline_summary.csv`。
- `tools/plot_baseline_curves.py`：从 `metrics.csv` 等绘制英文标注曲线到 `results/figures/`。
