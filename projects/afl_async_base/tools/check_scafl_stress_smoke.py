#!/usr/bin/env python3
"""
读取一次 stress smoke 的 run 目录，检查 SC-AFL-like baseline 机制是否在日志中体现。

用法:
  python tools/check_scafl_stress_smoke.py --log-root /path/to/logs/afl_async_base
  python tools/check_scafl_stress_smoke.py --run-dir /path/to/logs/afl_async_base/20260101-120000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

EPS = 1e-12


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        v = float(str(x).strip())
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return (list(r.fieldnames or []), rows)


def _parse_pipe_list(raw: Any) -> List[str]:
    s = "" if raw is None else str(raw).strip()
    if not s:
        return []
    return [p for p in s.split("|") if p != ""]


def _find_latest_run_dir(log_root: Path) -> Optional[Path]:
    if not log_root.is_dir():
        return None
    candidates: List[Path] = []
    for child in log_root.iterdir():
        if child.is_dir() and (child / "round_metrics.csv").is_file():
            candidates.append(child)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="SC-AFL-like stress smoke log checks")
    ap.add_argument(
        "--log-root",
        type=str,
        default="",
        help="含多次 run 子目录（时间戳命名）的根路径；与 --run-dir 二选一",
    )
    ap.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="单次实验 run 目录（内含 round_metrics.csv 等）",
    )
    args = ap.parse_args()

    run_dir: Optional[Path] = None
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.log_root:
        lr = Path(args.log_root).resolve()
        run_dir = _find_latest_run_dir(lr)
        if run_dir is None:
            print(f"ERROR: no run subdirectory with round_metrics.csv under: {lr}", file=sys.stderr)
            return 2
    else:
        print("ERROR: provide --log-root or --run-dir", file=sys.stderr)
        return 2

    assert run_dir is not None
    required = {
        "decision_debug": run_dir / "decision_debug.csv",
        "queue_trace": run_dir / "queue_trace.csv",
        "p2_prefix_diagnostics": run_dir / "p2_prefix_diagnostics.csv",
        "round_metrics": run_dir / "round_metrics.csv",
    }
    missing = [name for name, p in required.items() if not p.is_file()]
    if missing:
        print("SC-AFL-like Stress Smoke Check", file=sys.stderr)
        print("================================", file=sys.stderr)
        for name in missing:
            print(f"ERROR: missing log file: {required[name]}", file=sys.stderr)
        return 2

    _, dbg = _read_csv(required["decision_debug"])
    _, qtr = _read_csv(required["queue_trace"])
    _, p2r = _read_csv(required["p2_prefix_diagnostics"])
    _, rmr = _read_csv(required["round_metrics"])

    lines: List[str] = []
    hard_fail = False
    has_warn = False

    def emit(status: str, title: str, detail: Dict[str, Any], *, is_warn: bool = False) -> None:
        nonlocal has_warn
        lines.append(f"[{status}] {title}")
        for k, v in detail.items():
            lines.append(f"       {k}: {v}")
        if is_warn:
            has_warn = True

    print("SC-AFL-like Stress Smoke Check")
    print("================================")

    # --- 1 candidate_count > selected_count ---
    max_cand = 0
    max_sel = 0
    n_gt = 0
    for row in rmr:
        cc = _to_int(row.get("candidate_count"))
        sc = _to_int(row.get("selected_count"))
        if cc is not None:
            max_cand = max(max_cand, cc)
        if sc is not None:
            max_sel = max(max_sel, sc)
        if cc is not None and sc is not None and cc > sc:
            n_gt += 1
    ok1 = n_gt >= 1
    if not ok1:
        hard_fail = True
    emit(
        "PASS" if ok1 else "FAIL",
        "candidate_count > selected_count observed",
        {
            "max_candidate_count": max_cand,
            "max_selected_count": max_sel,
            "num_rounds_candidate_gt_selected": n_gt,
        },
    )

    # --- 2 beta_k 0 and 1 ---
    betas: List[int] = []
    for row in dbg:
        b = _to_int(row.get("beta_k"))
        if b is not None:
            betas.append(b)
    ub = sorted(set(betas))
    n0 = sum(1 for x in betas if x == 0)
    n1 = sum(1 for x in betas if x == 1)
    ok2 = 0 in ub and 1 in ub
    if not ok2:
        hard_fail = True
    emit(
        "PASS" if ok2 else "FAIL",
        "beta_k contains both 0 and 1",
        {"unique_beta_values": ub, "num_beta_0": n0, "num_beta_1": n1},
    )

    # --- 3 update_in_aggregated_prefix ---
    prefs: List[int] = []
    for row in dbg:
        v = _to_int(row.get("update_in_aggregated_prefix"))
        if v is not None:
            prefs.append(v)
    up = sorted(set(prefs))
    pf0 = sum(1 for x in prefs if x == 0)
    pf1 = sum(1 for x in prefs if x == 1)
    ok3 = 0 in up and 1 in up
    if not ok3:
        hard_fail = True
    emit(
        "PASS" if ok3 else "FAIL",
        "update_in_aggregated_prefix contains both 0 and 1",
        {
            "unique_update_in_aggregated_prefix_values": up,
            "num_prefix_flag_0": pf0,
            "num_prefix_flag_1": pf1,
        },
    )

    # --- 4 buffer carry-over ---
    n_co = 0
    n_buf = 0
    max_buf_sz = 0
    for row in rmr:
        co = _parse_pipe_list(row.get("carried_over_update_ids"))
        bf = _parse_pipe_list(row.get("buffer_update_ids_after_round"))
        if co:
            n_co += 1
        if bf:
            n_buf += 1
            max_buf_sz = max(max_buf_sz, len(bf))
    ok4 = n_co >= 1 or n_buf >= 1
    if not ok4:
        hard_fail = True
    emit(
        "PASS" if ok4 else "FAIL",
        "buffer carry-over observed",
        {
            "num_nonempty_carried_over_rounds": n_co,
            "num_nonempty_buffer_after_rounds": n_buf,
            "max_buffer_after_round_size": max_buf_sz,
        },
    )

    # --- 5 queue ---
    max_qb = 0.0
    max_qa = 0.0
    n_chg = 0
    n_inc = 0
    n_dec = 0
    for row in qtr:
        qb = _to_float(row.get("q_before"))
        qa = _to_float(row.get("q_after"))
        if qb is not None:
            max_qb = max(max_qb, qb)
        if qa is not None:
            max_qa = max(max_qa, qa)
        if qb is not None and qa is not None and abs(qa - qb) > EPS:
            n_chg += 1
            if qa > qb:
                n_inc += 1
            elif qa < qb:
                n_dec += 1
    ok5 = n_chg >= 1
    if not ok5:
        emit(
            "WARN",
            "virtual queue did not change",
            {
                "max_q_before": max_qb,
                "max_q_after": max_qa,
                "num_queue_changed_rows": n_chg,
                "num_queue_increase_rows": n_inc,
                "num_queue_decrease_rows": n_dec,
                "suggestion": "try staleness_cutoff=1, buffer_size=5, updates_per_client=16",
            },
            is_warn=True,
        )
    else:
        emit(
            "PASS",
            "virtual queue changed (queue_trace)",
            {
                "max_q_before": max_qb,
                "max_q_after": max_qa,
                "num_queue_changed_rows": n_chg,
                "num_queue_increase_rows": n_inc,
                "num_queue_decrease_rows": n_dec,
            },
        )

    # --- 6 multiple prefix_size per logical_round ---
    by_lr: Dict[int, Set[int]] = {}
    for row in p2r:
        lr = _to_int(row.get("logical_round"))
        ps = _to_int(row.get("prefix_size"))
        if lr is None or ps is None:
            continue
        by_lr.setdefault(lr, set()).add(ps)
    multi_lr = [lr for lr, s in by_lr.items() if len(s) > 1]
    num_multi = len(multi_lr)
    max_pfx_per = max((len(s) for s in by_lr.values()), default=0)
    ex_lr = min(multi_lr) if multi_lr else None
    ok6 = num_multi >= 1
    if not ok6:
        hard_fail = True
    emit(
        "PASS" if ok6 else "FAIL",
        "multiple prefix_size values observed under same logical_round",
        {
            "num_rounds_with_multiple_prefix_size": num_multi,
            "max_prefix_size_count_per_round": max_pfx_per,
            "example_logical_round_with_multiple_prefix_size": ex_lr,
        },
    )

    # --- 7 selected_objective_p2 ---
    objs: List[float] = []
    for row in rmr:
        o = _to_float(row.get("selected_objective_p2"))
        if o is not None:
            objs.append(o)
    n_valid = len(objs)
    ok7 = n_valid >= 1
    if not ok7:
        hard_fail = True
    o_min = min(objs) if objs else None
    o_max = max(objs) if objs else None
    emit(
        "PASS" if ok7 else "FAIL",
        "selected_objective_p2 is valid",
        {
            "num_valid_selected_objective_p2": n_valid,
            "min_selected_objective_p2": o_min,
            "max_selected_objective_p2": o_max,
        },
    )

    for ln in lines:
        print(ln)

    if hard_fail:
        overall = "FAIL"
    elif has_warn:
        overall = "PASS_WITH_WARNINGS"
    else:
        overall = "PASS"
    print("")
    print(f"Overall result:\n{overall}")
    print(f"(run_dir: {run_dir})")
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
