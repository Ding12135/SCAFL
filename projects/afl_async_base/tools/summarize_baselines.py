#!/usr/bin/env python3
"""
Aggregate baseline runs under log_root into results/baseline_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _time_to_acc(rows: List[Dict[str, str]], thr: float) -> str:
    t_best: Optional[float] = None
    for r in rows:
        acc = _safe_float(r.get("test_acc"))
        wt = _safe_float(r.get("wall_time"))
        if acc is None or wt is None:
            continue
        if acc < 0:
            continue
        if acc >= thr:
            if t_best is None or wt < t_best:
                t_best = wt
    return "" if t_best is None else str(t_best)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    sum_path = run_dir / "summary.json"
    if not cfg_path.is_file() or not sum_path.is_file():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(sum_path, encoding="utf-8") as f:
        summary = json.load(f)

    metrics = _read_csv(run_dir / "metrics.csv")
    flush_m = _read_csv(run_dir / "flush_metrics.csv")

    accs = [
        float(r["test_acc"])
        for r in metrics
        if _safe_float(r.get("test_acc")) is not None and float(r["test_acc"]) >= 0
    ]
    best_acc = max(accs) if accs else ""

    rem_buf = [
        _safe_float(r.get("remaining_buffer_count"))
        for r in metrics
        if _safe_float(r.get("remaining_buffer_count")) is not None
    ]
    avg_rem = sum(rem_buf) / len(rem_buf) if rem_buf else ""

    drop_stale = [
        _safe_float(r.get("dropped_stale_count"))
        for r in metrics
        if _safe_float(r.get("dropped_stale_count")) is not None
    ]
    avg_drop_stale = sum(drop_stale) / len(drop_stale) if drop_stale else ""

    accepted_n = sum(1 for r in metrics if str(r.get("accepted", "0")) == "1")
    recv_n = len(metrics) or 1
    accept_ratio = accepted_n / recv_n
    drop_ratio = 1.0 - accept_ratio

    method = f"{cfg.get('async_mode', '')}+{cfg.get('policy', {}).get('type', '')}"

    return {
        "method": method,
        "config_name": run_dir.name,
        "seed": str(cfg.get("seed", "")),
        "final_accuracy": summary.get("final_accuracy", ""),
        "best_accuracy": best_acc,
        "total_wall_time": summary.get("total_wall_time", ""),
        "total_received_updates": summary.get("total_received_updates", ""),
        "total_applied_steps": summary.get("total_applied_steps", ""),
        "avg_staleness": summary.get("avg_staleness", ""),
        "max_staleness": summary.get("max_staleness", ""),
        "accept_ratio": accept_ratio,
        "drop_ratio": drop_ratio,
        "flush_count": summary.get("flush_count", ""),
        "avg_flush_size": summary.get("avg_flush_size", ""),
        "avg_compute_time": summary.get("avg_compute_time", ""),
        "avg_upload_delay": summary.get("avg_upload_delay", ""),
        "avg_remaining_buffer_count": avg_rem,
        "avg_dropped_stale_count": avg_drop_stale,
        "time_to_accuracy_50": _time_to_acc(metrics, 0.5),
        "time_to_accuracy_60": _time_to_acc(metrics, 0.6),
        "time_to_accuracy_70": _time_to_acc(metrics, 0.7),
        "time_to_accuracy_80": _time_to_acc(metrics, 0.8),
        "run_dir": str(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log-root",
        type=Path,
        default=Path("/home/kamila/SCAFL/logs/afl_async_base"),
        help="Directory containing experiment run_* subfolders",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "baseline_summary.csv",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    if args.log_root.is_dir():
        for child in sorted(args.log_root.iterdir()):
            if not child.is_dir():
                continue
            row = summarize_run(child)
            if row:
                rows.append(row)

    if not rows:
        print("No runs found under", args.log_root)
        return

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Wrote", args.out, "rows=", len(rows))


if __name__ == "__main__":
    main()
