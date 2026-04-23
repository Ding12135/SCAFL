#!/usr/bin/env python3
"""
读取 PPT 实验的 metrics.csv / summary.json 绘制对比图（用于汇报幻灯片）。

默认读取 configs/ppt/last_runs_manifest.json（由 run_ppt_experiments.py 生成）。
也可手动指定多个 run 目录。

依赖: matplotlib

用法:
  cd projects/afl_async_base
  python scripts/plot_ppt_results.py
  python scripts/plot_ppt_results.py --manifest configs/ppt/last_runs_manifest.json --out-dir logs/ppt_prepared/figures

输出:
  - ppt_test_accuracy.png   测试精度 vs global_step
  - ppt_test_loss.png       测试损失 vs global_step
  - ppt_summary_bar.png     各次运行 final_accuracy / wall_time 柱状图
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def setup_matplotlib_chinese() -> None:
    """
    配置支持中文的字体，避免标题/图例/坐标轴出现方框。
    依次尝试：已注册 CJK 字体名、常见 Linux 字体文件、WSL 访问 Windows 字体。
    """
    import os

    import matplotlib as mpl
    from matplotlib import font_manager

    mpl.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII，避免变方块

    # 1) 已安装在系统中的字体名（fontManager 能解析的 family name）
    preferred_names = (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "STHeiti",
        "PingFang SC",
    )
    known = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_names:
        if name in known:
            mpl.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            print(f"[plot] 中文字体: {name}")
            return

    # 2) 通过字体文件注册（WSL 读 Windows、Linux 常见路径）
    font_paths = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/msyhbd.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/mnt/c/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for fp in font_paths:
        if not fp or not os.path.isfile(fp):
            continue
        try:
            font_manager.fontManager.addfont(fp)
            prop = font_manager.FontProperties(fname=fp)
            fam = prop.get_name()
            mpl.rcParams["font.sans-serif"] = [fam, "DejaVu Sans", "sans-serif"]
            print(f"[plot] 中文字体: {fam} ({fp})")
            return
        except Exception:
            continue

    # 3) 兜底：仍设一串常见名，部分环境可能后续会解析到
    mpl.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
        "sans-serif",
    ]
    print(
        "[plot] 未找到专用 CJK 字体文件，已使用兜底字体列表；若中文仍显示为方框，请安装 fonts-noto-cjk 或将 Windows 字体路径加入脚本。",
        file=sys.stderr,
    )


def load_metrics_rows(metrics_path: Path) -> list[dict]:
    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def forward_fill_metric(rows: list[dict], key: str, invalid: set[str | float]) -> list[tuple[int, float]]:
    """按 global_step 去重保留最后一条；test_acc/test_loss 对无效值前向填充。"""
    last_val: float | None = None
    by_step: dict[int, float] = {}
    for row in rows:
        try:
            gs = int(float(row["global_step"]))
        except (KeyError, ValueError):
            continue
        raw = row.get(key, "")
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v in invalid or v < 0:
            if last_val is not None:
                by_step[gs] = last_val
            continue
        last_val = v
        by_step[gs] = v
    return sorted(by_step.items())


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:
        print(
            "缺少 matplotlib。请执行: pip install -r requirements-ppt.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from e


def plot_curves(
    series: list[tuple[str, Path]],
    y_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
    invalid_acc: set[float],
) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for label, run_dir in series:
        mp = run_dir / "metrics.csv"
        if not mp.is_file():
            print(f"[skip] 无 metrics.csv: {mp}", file=sys.stderr)
            continue
        rows = load_metrics_rows(mp)
        pts = forward_fill_metric(rows, y_key, invalid_acc if y_key == "test_acc" else {-1.0})
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label=label)

    plt.xlabel("global_step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[write] {out_path}")


def plot_summary_bars(series: list[tuple[str, Path]], out_path: Path) -> None:
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    labels: list[str] = []
    accs: list[float] = []
    times: list[float] = []
    for label, run_dir in series:
        sp = run_dir / "summary.json"
        if not sp.is_file():
            print(f"[skip] 无 summary.json: {sp}", file=sys.stderr)
            continue
        with open(sp, "r", encoding="utf-8") as f:
            s = json.load(f)
        # 长标签换行，避免与中文混排时挤在一起
        labels.append(label.replace(" ", "\n") if len(label) > 18 else label)
        accs.append(float(s.get("final_accuracy", 0.0)))
        times.append(float(s.get("total_wall_time", 0.0)))

    if not labels:
        print("[error] 无有效 summary，跳过柱状图", file=sys.stderr)
        return

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = range(len(labels))
    w = 0.35
    ax1.bar([i - w / 2 for i in x], accs, width=w, label="最终测试精度", color="steelblue")
    ax1.set_ylabel("最终测试精度")
    ax1.set_ylim(0, max(1.0, max(accs) * 1.1) if accs else 1.0)

    ax2 = ax1.twinx()
    ax2.bar([i + w / 2 for i in x], times, width=w, label="总耗时 (秒)", color="coral")
    ax2.set_ylabel("总耗时 (秒)")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8, rotation=0)
    ax1.set_title("PPT 实验汇总（精度与总耗时）")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[write] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制 PPT 对比图")
    ap.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "ppt" / "last_runs_manifest.json",
        help="run_ppt_experiments.py 生成的 manifest",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT.parent.parent / "logs" / "ppt_prepared" / "figures",
        help="输出 PNG 目录（默认 SCAFL/logs/ppt_prepared/figures）",
    )
    ap.add_argument(
        "run_dirs",
        nargs="*",
        help="可选：手动指定 run_dir 列表（若给则忽略 manifest，需与 --labels 成对）",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        help="与 run_dirs 一一对应的图例标签",
    )
    args = ap.parse_args()

    series: list[tuple[str, Path]] = []
    if args.run_dirs:
        labels = args.labels or [f"run{i}" for i in range(len(args.run_dirs))]
        if len(labels) != len(args.run_dirs):
            print("[error] --labels 数量须与手动 run_dir 数量一致", file=sys.stderr)
            sys.exit(1)
        series = list(zip(labels, [Path(p).resolve() for p in args.run_dirs]))
    else:
        if not args.manifest.is_file():
            print(f"[error] 找不到 manifest: {args.manifest}", file=sys.stderr)
            sys.exit(1)
        with open(args.manifest, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            series.append((item["label"], Path(item["run_dir"])))

    invalid_acc = {-1.0}

    _ensure_matplotlib()
    setup_matplotlib_chinese()

    plot_curves(
        series,
        "test_acc",
        "测试集准确率 vs 全局步",
        "test_acc",
        args.out_dir / "ppt_test_accuracy.png",
        invalid_acc,
    )
    plot_curves(
        series,
        "test_loss",
        "测试集损失 vs 全局步",
        "test_loss",
        args.out_dir / "ppt_test_loss.png",
        {-1.0},
    )
    plot_summary_bars(series, args.out_dir / "ppt_summary_bar.png")

    print(f"\n[done] 图表已输出到: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
