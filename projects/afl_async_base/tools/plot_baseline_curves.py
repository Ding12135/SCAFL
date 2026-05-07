#!/usr/bin/env python3
"""Plot baseline curves from one or more run directories (matplotlib only, English labels)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def _read(path: Path) -> Tuple[List[str], List[List[str]]]:
    if not path.is_file():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _col(header: List[str], name: str) -> int:
    return header.index(name)


def plot_metrics(run_dir: Path, out_dir: Path) -> None:
    hdr, rows = _read(run_dir / "metrics.csv")
    if not hdr or not rows:
        return
    def col(name: str) -> List[float]:
        i = _col(hdr, name)
        out: List[float] = []
        for r in rows:
            try:
                out.append(float(r[i]))
            except (ValueError, IndexError):
                out.append(float("nan"))
        return out

    wall = col("wall_time")
    acc = col("test_acc")
    gs = col("global_step")
    loss = col("test_loss")
    stal = col("staleness")
    blen = col("buffer_len_after")

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = run_dir.name

    plt.figure()
    plt.plot(wall, acc, marker=".", linestyle="-")
    plt.xlabel("Wall Time (s)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs Wall Time ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_accuracy_vs_wall_time.png")
    plt.close()

    plt.figure()
    plt.plot(gs, loss, marker=".", linestyle="-")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Global Step ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_loss_vs_global_step.png")
    plt.close()

    plt.figure()
    plt.plot(gs, stal, marker=".", linestyle="-")
    plt.xlabel("Global Step")
    plt.ylabel("Average Staleness")
    plt.title(f"Staleness vs Global Step ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_avg_staleness_vs_global_step.png")
    plt.close()

    recv = list(range(1, len(blen) + 1))
    plt.figure()
    plt.plot(recv, blen, marker=".", linestyle="-")
    plt.xlabel("Receive Step")
    plt.ylabel("Buffer Length")
    plt.title(f"Buffer Length vs Receive Step ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_buffer_len_vs_recv_step.png")
    plt.close()


def plot_flush_round(run_dir: Path, out_dir: Path) -> None:
    hdr, rows = _read(run_dir / "flush_metrics.csv")
    if not hdr or not rows:
        return
    i_n = _col(hdr, "num_updates")
    sizes = [float(r[i_n]) for r in rows]
    rnd = list(range(1, len(sizes) + 1))
    tag = run_dir.name
    plt.figure()
    plt.plot(rnd, sizes, marker="o", linestyle="-")
    plt.xlabel("Flush Round")
    plt.ylabel("Flush Size")
    plt.title(f"Flush Size vs Round ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_flush_size_vs_round.png")
    plt.close()


def plot_queue_round(run_dir: Path, out_dir: Path) -> None:
    hdr, rows = _read(run_dir / "queue_trace.csv")
    if not hdr or not rows:
        return
    i_q = _col(hdr, "q_after")
    i_r = _col(hdr, "logical_round")
    by_round: dict[int, float] = {}
    for r in rows:
        try:
            lr = int(float(r[i_r]))
            qv = float(r[i_q])
        except (ValueError, IndexError):
            continue
        by_round[lr] = max(by_round.get(lr, 0.0), qv)
    if not by_round:
        return
    rnds = sorted(by_round.keys())
    vals = [by_round[k] for k in rnds]
    tag = run_dir.name
    plt.figure()
    plt.plot(rnds, vals, marker="o", linestyle="-")
    plt.xlabel("Round")
    plt.ylabel("Max Queue After Update")
    plt.title(f"Queue Max vs Round ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_queue_max_vs_round.png")
    plt.close()


def plot_drop_bar(run_dir: Path, out_dir: Path) -> None:
    hdr, rows = _read(run_dir / "metrics.csv")
    if not hdr or not rows:
        return
    i_a = _col(hdr, "accepted")
    acc = sum(1 for r in rows if r[i_a] == "1")
    drop = len(rows) - acc
    tag = run_dir.name
    plt.figure()
    plt.bar(["Accepted", "Dropped"], [acc, drop])
    plt.ylabel("Count")
    plt.title(f"Drop Ratio Bar ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_drop_ratio_bar.png")
    plt.close()


def plot_tta_bar(run_dir: Path, out_dir: Path) -> None:
    hdr, rows = _read(run_dir / "metrics.csv")
    if not hdr or not rows:
        return
    i_acc = _col(hdr, "test_acc")
    i_wt = _col(hdr, "wall_time")

    def tta(th: float) -> float:
        best: float | None = None
        for r in rows:
            try:
                a = float(r[i_acc])
                w = float(r[i_wt])
            except (ValueError, IndexError):
                continue
            if a < 0:
                continue
            if a >= th and (best is None or w < best):
                best = w
        return float(best) if best is not None else float("nan")

    ths = [0.5, 0.6, 0.7, 0.8]
    vals = [tta(t) for t in ths]
    tag = run_dir.name
    plt.figure()
    plt.bar([str(t) for t in ths], vals)
    plt.xlabel("Accuracy Threshold")
    plt.ylabel("Wall Time (s)")
    plt.title(f"Time-to-Accuracy ({tag})")
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_time_to_accuracy_bar.png")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path, help="Experiment run directories")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "figures",
    )
    args = ap.parse_args()
    for rd in args.run_dirs:
        if not rd.is_dir():
            continue
        plot_metrics(rd, args.out_dir)
        plot_flush_round(rd, args.out_dir)
        plot_queue_round(rd, args.out_dir)
        plot_drop_bar(rd, args.out_dir)
        plot_tta_bar(rd, args.out_dir)
    print("Figures written to", args.out_dir)


if __name__ == "__main__":
    main()
