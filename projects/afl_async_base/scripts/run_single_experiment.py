#!/usr/bin/env python3
"""
只跑一个 yaml 配置，结束后自动对本次 run_dir 调用 plot_ppt_results.py 出图。

用法（在 projects/afl_async_base 下）:
  export PYTHONPATH="$PWD"
  python scripts/run_single_experiment.py configs/ppt/01_buffered_dynamic.yaml

可选:
  python scripts/run_single_experiment.py configs/ppt/01_buffered_dynamic.yaml \\
    --label "① buffered+动态" \\
    --out-dir /home/kamila/SCAFL/logs/ppt_prepared/figures_single
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="单次实验 + 自动画图")
    ap.add_argument(
        "config",
        type=Path,
        help="yaml 配置路径（相对 ROOT 或绝对路径）",
    )
    ap.add_argument(
        "--label",
        type=str,
        default="",
        help="图例标签（默认用配置文件名，不含 .yaml）",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="PNG 输出目录（默认 ROOT/../../logs/ppt_prepared/figures_single）",
    )
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        print(f"[error] 找不到配置: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log_root = Path(cfg["log_root"]).resolve()

    label = args.label.strip() or cfg_path.stem

    before: set[Path] = set()
    if log_root.exists():
        before = {p for p in log_root.iterdir() if p.is_dir()}

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["AFL_CONFIG"] = str(cfg_path)

    print(f"[run] AFL_CONFIG={cfg_path}")
    r = subprocess.run(
        [sys.executable, "-m", "afl.server"],
        cwd=str(ROOT),
        env=env,
    )
    if r.returncode != 0:
        print(f"[error] afl.server 退出码 {r.returncode}", file=sys.stderr)
        sys.exit(r.returncode)

    after: set[Path] = set()
    if log_root.exists():
        after = {p for p in log_root.iterdir() if p.is_dir()}

    new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
    if not new_dirs:
        print(f"[error] 未在 {log_root} 下检测到新 run 目录", file=sys.stderr)
        sys.exit(1)

    run_dir = new_dirs[-1]
    print(f"[ok] run_dir={run_dir}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT.parent.parent / "logs" / "ppt_prepared" / "figures_single"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_script = ROOT / "scripts" / "plot_ppt_results.py"
    print(f"[plot] out_dir={out_dir}")
    r2 = subprocess.run(
        [
            sys.executable,
            str(plot_script),
            "--out-dir",
            str(out_dir),
            str(run_dir),
            "--labels",
            label,
        ],
        cwd=str(ROOT),
        env=env,
    )
    if r2.returncode != 0:
        sys.exit(r2.returncode)
    print(f"\n[done] 图表: {out_dir}")


if __name__ == "__main__":
    main()
