#!/usr/bin/env python3
"""
顺序运行 PPT 预设的若干 yaml，并在 configs/ppt/last_runs_manifest.json 记录
每次运行的 run_dir，供 plot_ppt_results.py 画图。

用法（在项目根 projects/afl_async_base 下）:
  python scripts/run_ppt_experiments.py

环境: 需已能运行 python -m afl.server（CUDA/数据路径与 yaml 一致）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# (展示用短标签, 相对 ROOT 的配置路径)
PPT_RUNS: list[tuple[str, str]] = [
    ("① buffered + exp + 动态控制", "configs/ppt/01_buffered_dynamic.yaml"),
    ("② immediate + exp", "configs/ppt/02_immediate.yaml"),
    ("③ buffered + 关动态", "configs/ppt/03_buffered_no_dynamic.yaml"),
    ("④ buffered + staleness none", "configs/ppt/04_staleness_none.yaml"),
]

MANIFEST_PATH = ROOT / "configs" / "ppt" / "last_runs_manifest.json"


def main() -> None:
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    manifest: list[dict] = []

    for label, rel in PPT_RUNS:
        cfg_path = ROOT / rel
        if not cfg_path.is_file():
            print(f"[skip] 找不到配置: {cfg_path}", file=sys.stderr)
            continue

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        log_root = Path(cfg["log_root"])

        before: set[Path] = set()
        if log_root.exists():
            before = {p for p in log_root.iterdir() if p.is_dir()}

        print(f"\n========== {label} ==========")
        print(f"AFL_CONFIG={cfg_path}")
        env_run = env.copy()
        env_run["AFL_CONFIG"] = str(cfg_path.resolve())

        r = subprocess.run(
            [sys.executable, "-m", "afl.server"],
            cwd=str(ROOT),
            env=env_run,
        )
        if r.returncode != 0:
            print(f"[error] 运行失败 returncode={r.returncode}，标签: {label}", file=sys.stderr)
            continue

        after: set[Path] = set()
        if log_root.exists():
            after = {p for p in log_root.iterdir() if p.is_dir()}

        new_dirs = sorted(after - before, key=lambda p: p.stat().st_mtime)
        if not new_dirs:
            print(f"[warn] 未检测到新 run 目录 under {log_root}", file=sys.stderr)
            continue

        run_dir = new_dirs[-1]
        manifest.append(
            {
                "label": label,
                "config": rel.replace("\\", "/"),
                "run_dir": str(run_dir.resolve()),
            }
        )
        print(f"[ok] run_dir={run_dir}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 已写入 manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
