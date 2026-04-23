# PPT 汇报用：参数对比与出图说明

## 1. 选了哪几个「最重要」的参数？

在其它条件尽量一致的前提下，用 **4 组配置** 覆盖三类对比（适合明天汇报一页「实验设计」+ 一页「结果图」）：

| 文件 | 变动的关键参数 | 汇报时一句话 |
|------|----------------|--------------|
| `01_buffered_dynamic.yaml` | **基线**：`async_mode: buffered`，`staleness_weight: exp`，`dynamic_control.enabled: true` | 缓冲聚合 + 指数 staleness 加权 + 动态缓冲/τ |
| `02_immediate.yaml` | **`async_mode: immediate`** | 与缓冲模式对比「异步更新形态」 |
| `03_buffered_no_dynamic.yaml` | **`dynamic_control.enabled: false`** | 与基线对比「是否开启动态调节」 |
| `04_staleness_none.yaml` | **`staleness_weight: none`** | 与 exp 对比「陈旧度是否参与加权」 |

共同设定（公平对比）：同一 `seed`、`num_clients`、`updates_per_client`、数据路径、本地训练超参、`server_lr` 等。当前默认 **`num_clients: 5`**、**`updates_per_client: 80`**（共 **400** 条服务端接收更新/实验；运行时间显著长于小规模 smoke，可按需改四个 yaml 里同一数值）。

**请自行检查**：各文件中的 `data_dir`、`log_root`、`device`（`cuda`/`cpu`）是否符合你机器路径与显卡情况。

---

## 2. 如何跑实验？

在 **`projects/afl_async_base`** 目录下：

```bash
export PYTHONPATH="$PWD"
python scripts/run_ppt_experiments.py
```

脚本会按顺序执行四次 `python -m afl.server`，并把每次的 `run_dir` 写入：

`configs/ppt/last_runs_manifest.json`

若只想跑其中某一组，可照旧使用：

```bash
export PYTHONPATH="$PWD"
export AFL_CONFIG="$PWD/configs/ppt/01_buffered_dynamic.yaml"
python -m afl.server
```

---

## 3. 如何画图？

需安装 **matplotlib**：

```bash
pip install -r requirements-ppt.txt
```

（`requirements-ppt.txt` 位于 `projects/afl_async_base/` 目录。）

```bash
cd projects/afl_async_base
python scripts/plot_ppt_results.py
```

默认读取 `configs/ppt/last_runs_manifest.json`，图片输出到：

`SCAFL/logs/ppt_prepared/figures/`（即仓库根下 `logs/ppt_prepared/figures/`）

生成文件：

- `ppt_test_accuracy.png`：测试准确率 vs `global_step`
- `ppt_test_loss.png`：测试损失 vs `global_step`
- `ppt_summary_bar.png`：各次运行的 `final_accuracy` 与 `total_wall_time`

**手动指定多次运行目录**（未用 manifest 时）：

```bash
python scripts/plot_ppt_results.py \
  --out-dir /path/to/figures \
  /path/to/run_a /path/to/run_b \
  --labels "基线" "immediate"
```

---

## 4. 汇报时可讲的「验证点」

- **① vs ②**：缓冲 flush 与即时更新对收敛/精度的影响。  
- **① vs ③**：动态 `buffer_target_t` / `tau_max_t` 是否带来差异（需结合 `metrics.csv` 中控制量列）。  
- **① vs ④**：staleness 指数衰减加权 vs 不加权。

---

## 5. 是否确定采用本套代码？

当前新增内容仅限：`configs/ppt/*.yaml`、两个脚本、本说明。若你希望 **改对比轴**（例如改成 `non_iid: true`、加大客户端数），只需编辑对应 yaml 或复制新文件并在 `scripts/run_ppt_experiments.py` 的 `PPT_RUNS` 列表中注册。
