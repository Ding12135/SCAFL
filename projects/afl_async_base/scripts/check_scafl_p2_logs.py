#!/usr/bin/env python3
"""
Validate SC-AFL P2 diagnostic logs consistency for one run_dir.

This is an engineering validation utility for reproducibility/debugging.
It is NOT part of the SC-AFL paper algorithm itself.

Checker semantics:
- **Pure P2** (default): diagnostics assume one pending candidate row corresponds to one
  distinct client in the usual way, so ``beta_ones + beta_zeros`` is compared to
  ``candidate_count`` (pending row count), and ``decision_debug`` client ids are compared
  as ordered multisets (sorted lists, duplicates allowed).
- **scafl_skeleton** (and alias ``scafl``): the implementation uses **client-level** beta
  aggregation, so multiple pending rows may refer to the same client. For skeleton,
  ``beta_ones + beta_zeros`` is compared to the **number of unique candidate clients**
  derived from ``selected_client_ids`` ∪ ``unselected_client_ids`` on each prefix row, and
  ``decision_debug_match`` compares **deduplicated** client ids to the selected prefix
  (rows where ``beta_k==1`` are **client-level** policy selection; legacy CSV may also
  list duplicate ``was_selected`` equal to ``beta_k``).
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EPS = 1e-8
REPORT_NAME = "p2_validation_report.csv"


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None or x == "":
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(str(x).strip())
    except Exception:
        return None


def _parse_list_field(raw: Any) -> List[str]:
    s = "" if raw is None else str(raw).strip()
    if not s:
        return []
    if "|" in s:
        return [p for p in s.split("|") if p != ""]
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip() != ""]
    return [s]


def _parse_int_list(raw: Any) -> List[int]:
    out: List[int] = []
    for p in _parse_list_field(raw):
        iv = _to_int(p)
        if iv is not None:
            out.append(iv)
    return out


def _parse_policy_params(params: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for piece in (params or "").split(";"):
        piece = piece.strip()
        if not piece or "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _is_skeleton_policy(policy_type: str) -> bool:
    pt = (policy_type or "").strip().lower()
    return pt in ("scafl_skeleton", "scafl")


def _candidate_client_count_unique(row: Dict[str, str]) -> Optional[int]:
    """Unique client count for this prefix row from selected ∪ unselected id lists."""
    sel = _parse_int_list(row.get("selected_client_ids"))
    uns = _parse_int_list(row.get("unselected_client_ids"))
    if not sel and not uns:
        return None
    return len(set(sel) | set(uns))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _has_cols(rows: List[Dict[str, str]], cols: Iterable[str]) -> bool:
    if not rows:
        return False
    keys = set(rows[0].keys())
    return all(c in keys for c in cols)


def _report(
    sink: List[Dict[str, Any]],
    *,
    logical_round: Optional[int],
    check_type: str,
    status: str,
    detail: str,
    expected: str = "",
    actual: str = "",
) -> None:
    sink.append(
        {
            "logical_round": "" if logical_round is None else logical_round,
            "check_type": check_type,
            "status": status,
            "detail": detail,
            "expected": expected,
            "actual": actual,
        }
    )


def validate_run_dir(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    report: List[Dict[str, Any]] = []

    p2_path = run_dir / "p2_prefix_diagnostics.csv"
    round_path = run_dir / "round_metrics.csv"
    debug_path = run_dir / "decision_debug.csv"

    missing = [p for p in [p2_path, round_path, debug_path] if not p.is_file()]
    if missing:
        for p in missing:
            _report(
                report,
                logical_round=None,
                check_type="input_files",
                status="fail",
                detail=f"missing required file: {p}",
            )
        return report, {"total_rounds_checked": 0, "total_prefix_rows_checked": 0, "total_failures": len(missing)}

    p2_rows = _read_csv_rows(p2_path)
    round_rows = _read_csv_rows(round_path)
    debug_rows = _read_csv_rows(debug_path)

    p2_by_round: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in p2_rows:
        lr = _to_int(r.get("logical_round"))
        if lr is not None:
            p2_by_round[lr].append(r)

    round_by_lr: Dict[int, Dict[str, str]] = {}
    for r in round_rows:
        lr = _to_int(r.get("logical_round"))
        if lr is not None:
            round_by_lr[lr] = r

    debug_by_lr: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for r in debug_rows:
        lr = _to_int(r.get("logical_round"))
        if lr is not None:
            debug_by_lr[lr].append(r)

    all_rounds = sorted(set(p2_by_round.keys()) | set(round_by_lr.keys()) | set(debug_by_lr.keys()))

    # Rule B: per-prefix checks
    for row in p2_rows:
        lr = _to_int(row.get("logical_round"))
        params = _parse_policy_params(row.get("policy_params", ""))
        v = _to_float(params.get("V"))
        obj = _to_float(row.get("objective_p2"))
        d_t = _to_float(row.get("D_t_prefix"))
        csum = _to_float(row.get("candidate_term_sum"))
        if v is None or obj is None or d_t is None or csum is None:
            _report(
                report,
                logical_round=lr,
                check_type="objective_formula",
                status="skipped",
                detail="missing V/objective_p2/D_t_prefix/candidate_term_sum",
            )
        else:
            rhs = v * d_t + csum
            ok = abs(obj - rhs) <= EPS
            _report(
                report,
                logical_round=lr,
                check_type="objective_formula",
                status="pass" if ok else "fail",
                detail="objective_p2 == V * D_t_prefix + candidate_term_sum",
                expected=f"{rhs:.10f}",
                actual=f"{obj:.10f}",
            )

        b1 = _to_int(row.get("beta_ones"))
        b0 = _to_int(row.get("beta_zeros"))
        cc = _to_int(row.get("candidate_count"))
        policy_type = row.get("policy_type", "")
        if b1 is None or b0 is None or cc is None:
            _report(
                report,
                logical_round=lr,
                check_type="beta_count_consistency",
                status="skipped",
                detail="missing beta_ones/beta_zeros/candidate_count",
            )
        elif _is_skeleton_policy(policy_type):
            ucu = _candidate_client_count_unique(row)
            if ucu is None:
                _report(
                    report,
                    logical_round=lr,
                    check_type="beta_count_consistency",
                    status="skipped",
                    detail="scafl_skeleton: cannot derive candidate_client_count_unique (empty selected/unselected id lists)",
                )
            else:
                ok = (b1 + b0) == ucu
                diag = (
                    f"candidate_row_count={cc} candidate_client_count_unique={ucu} "
                    f"beta_ones={b1} beta_zeros={b0}"
                )
                _report(
                    report,
                    logical_round=lr,
                    check_type="beta_count_consistency",
                    status="pass" if ok else "fail",
                    detail=(
                        "scafl_skeleton: beta_ones + beta_zeros == candidate_client_count_unique "
                        f"(pending rows=candidate_count; pure P2 uses candidate_count only) | {diag}"
                    ),
                    expected=str(ucu),
                    actual=str(b1 + b0),
                )
        else:
            ok = (b1 + b0) == cc
            _report(
                report,
                logical_round=lr,
                check_type="beta_count_consistency",
                status="pass" if ok else "fail",
                detail="beta_ones + beta_zeros == candidate_count (pure P2)",
                expected=str(cc),
                actual=str(b1 + b0),
            )

    # Rules A/C/D: per-round checks
    for lr in all_rounds:
        prow = p2_by_round.get(lr, [])
        if not prow:
            _report(
                report,
                logical_round=lr,
                check_type="prefix_selected_uniqueness",
                status="fail",
                detail="no prefix rows for this logical_round",
            )
            continue

        sel_rows = [r for r in prow if _to_int(r.get("is_selected_prefix")) == 1]
        if len(sel_rows) == 1:
            _report(
                report,
                logical_round=lr,
                check_type="prefix_selected_uniqueness",
                status="pass",
                detail="exactly one selected prefix row",
            )
        else:
            _report(
                report,
                logical_round=lr,
                check_type="prefix_selected_uniqueness",
                status="fail",
                detail="selected prefix row count is not 1",
                expected="1",
                actual=str(len(sel_rows)),
            )

        if len(sel_rows) != 1:
            _report(
                report,
                logical_round=lr,
                check_type="prefix_optimality",
                status="skipped",
                detail="cannot evaluate optimality without unique selected row",
            )
            continue

        selected = sel_rows[0]
        selected_obj = _to_float(selected.get("objective_p2"))
        all_objs = [_to_float(r.get("objective_p2")) for r in prow]
        all_objs_valid = [x for x in all_objs if x is not None]
        if selected_obj is None or not all_objs_valid:
            _report(
                report,
                logical_round=lr,
                check_type="prefix_optimality",
                status="skipped",
                detail="missing objective_p2 values",
            )
        else:
            min_obj = min(all_objs_valid)
            ok = abs(selected_obj - min_obj) <= EPS
            _report(
                report,
                logical_round=lr,
                check_type="prefix_optimality",
                status="pass" if ok else "fail",
                detail="selected prefix objective equals minimum objective in round",
                expected=f"{min_obj:.10f}",
                actual=f"{selected_obj:.10f}",
            )

        rr = round_by_lr.get(lr)
        if rr is None:
            _report(
                report,
                logical_round=lr,
                check_type="round_summary_match",
                status="skipped",
                detail="round_metrics row missing for this logical_round",
            )
        else:
            checks = [
                ("selected_prefix_size", "selected_prefix_size"),
                ("selected_objective_p2", "selected_objective_p2"),
                ("selected_D_t", "selected_D_t"),
                ("candidate_term_sum", "candidate_term_sum"),
                ("candidate_count", "candidate_count"),
            ]
            for c_round, c_p2 in checks:
                rv = rr.get(c_round, "")
                pv = selected.get(c_p2, "")
                if rv == "" or pv == "":
                    _report(
                        report,
                        logical_round=lr,
                        check_type="round_summary_match",
                        status="skipped",
                        detail=f"missing column value for {c_round}/{c_p2}",
                    )
                    continue
                rfv = _to_float(rv)
                pfv = _to_float(pv)
                if rfv is None or pfv is None:
                    ok = str(rv) == str(pv)
                    _report(
                        report,
                        logical_round=lr,
                        check_type="round_summary_match",
                        status="pass" if ok else "fail",
                        detail=f"{c_round} equals selected prefix field {c_p2}",
                        expected=str(pv),
                        actual=str(rv),
                    )
                else:
                    ok = abs(rfv - pfv) <= EPS
                    _report(
                        report,
                        logical_round=lr,
                        check_type="round_summary_match",
                        status="pass" if ok else "fail",
                        detail=f"{c_round} equals selected prefix field {c_p2}",
                        expected=f"{pfv:.10f}",
                        actual=f"{rfv:.10f}",
                    )

        drows = debug_by_lr.get(lr, [])
        if not drows:
            _report(
                report,
                logical_round=lr,
                check_type="decision_debug_match",
                status="skipped",
                detail="no decision_debug rows for this logical_round",
            )
        else:
            cc = _to_int(selected.get("candidate_count"))
            if cc is None:
                _report(
                    report,
                    logical_round=lr,
                    check_type="decision_debug_match",
                    status="skipped",
                    detail="candidate_count missing in selected prefix row",
                )
            else:
                ok = len(drows) == cc
                _report(
                    report,
                    logical_round=lr,
                    check_type="decision_debug_match",
                    status="pass" if ok else "fail",
                    detail="decision_debug row count equals candidate_count",
                    expected=str(cc),
                    actual=str(len(drows)),
                )

            dbg_keys = set(drows[0].keys()) if drows else set()
            if {"client_id", "beta_k"}.issubset(dbg_keys):
                # client-level: beta_k==1（旧版 was_selected 与 beta_k 重复写入，已弃用列名）
                multiset_dbg = [
                    _to_int(r.get("client_id"))
                    for r in drows
                    if _to_int(r.get("beta_k")) == 1
                ]
                multiset_dbg = [x for x in multiset_dbg if x is not None]
                sel_dbg_sorted = sorted(multiset_dbg)
                pol = selected.get("policy_type", "")
                if _is_skeleton_policy(pol):
                    selected_ids_p2_set = sorted(set(_parse_int_list(selected.get("selected_client_ids"))))
                    selected_ids_dbg_unique = sorted(set(multiset_dbg))
                    ok = selected_ids_dbg_unique == selected_ids_p2_set
                    _report(
                        report,
                        logical_round=lr,
                        check_type="decision_debug_match",
                        status="pass" if ok else "fail",
                        detail=(
                            "scafl_skeleton: deduped client ids match selected prefix "
                            "(pure P2 uses multiset equality on sorted lists)"
                        ),
                        expected="|".join(str(x) for x in selected_ids_p2_set),
                        actual="|".join(str(x) for x in selected_ids_dbg_unique),
                    )
                    _report(
                        report,
                        logical_round=lr,
                        check_type="decision_debug_match_multiset_diag",
                        status="pass",
                        detail=(
                            "scafl_skeleton: multiset pipe string for same filter (diagnostic; "
                            "pass/fail above uses set semantics)"
                        ),
                        expected="|".join(str(x) for x in selected_ids_p2_set),
                        actual="|".join(str(x) for x in sel_dbg_sorted),
                    )
                else:
                    selected_ids_p2 = sorted(_parse_int_list(selected.get("selected_client_ids")))
                    ok = sel_dbg_sorted == selected_ids_p2
                    _report(
                        report,
                        logical_round=lr,
                        check_type="decision_debug_match",
                        status="pass" if ok else "fail",
                        detail="selected client multiset in decision_debug matches selected prefix (pure P2)",
                        expected="|".join(str(x) for x in selected_ids_p2),
                        actual="|".join(str(x) for x in sel_dbg_sorted),
                    )
            else:
                _report(
                    report,
                    logical_round=lr,
                    check_type="decision_debug_match",
                    status="skipped",
                    detail="required columns missing in decision_debug (need client_id/beta_k)",
                )

    failures = sum(1 for r in report if r["status"] == "fail")
    stats = {
        "total_rounds_checked": len(all_rounds),
        "total_prefix_rows_checked": len(p2_rows),
        "total_failures": failures,
    }
    return report, stats


def write_report(path: Path, rows: List[Dict[str, Any]]) -> None:
    header = ["logical_round", "check_type", "status", "detail", "expected", "actual"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate SC-AFL P2 logs consistency for one run_dir")
    ap.add_argument("run_dir", type=Path, help="path to one run directory")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"run_dir does not exist or is not a directory: {run_dir}")
        return
    report_path = run_dir / REPORT_NAME

    report, stats = validate_run_dir(run_dir)
    write_report(report_path, report)

    fail_rows = [r for r in report if r["status"] == "fail"]
    fail_by_type = Counter(r["check_type"] for r in fail_rows)

    print(f"total_rounds_checked={stats['total_rounds_checked']}")
    print(f"total_prefix_rows_checked={stats['total_prefix_rows_checked']}")
    print(f"total_failures={stats['total_failures']}")
    print(f"failures_by_type={dict(fail_by_type)}")
    print(f"report_path={report_path}")

    if not fail_rows:
        print("All checks passed")
    else:
        print("Top failures:")
        for r in fail_rows[:10]:
            print(
                f"- round={r['logical_round']} type={r['check_type']} detail={r['detail']} "
                f"expected={r['expected']} actual={r['actual']}"
            )


if __name__ == "__main__":
    main()
