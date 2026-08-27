"""
Distribution of per-request KVC prefix-hit tokens, from ``raw_kvc_hit_tokens``
stored in ``stage_<i>/opal_stats.json``.

Each value is the number of KV-cache tokens that were already present (prefix
hit) for a request, saving those tokens from being recomputed during prefill.
A value of 0 means no prefix was reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_stage_stats(run_dir: Path, stage_id: int | None) -> List[dict]:
    stats = []
    for json_path in sorted(run_dir.glob("stage_*/opal_stats.json")):
        sid = int(json_path.parent.name.split("_")[-1])
        if stage_id is not None and sid != stage_id:
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_stage_id"] = sid
        stats.append(data)
    return stats


def plot_kvc_hit_tokens(stats: List[dict], output_path: Path, title: str, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    all_values: List[int] = []
    for data in stats:
        raw = data.get("raw_kvc_hit_tokens", [])
        if not raw:
            continue
        values = np.array(raw, dtype=float)
        all_values.extend(values.tolist())
        label = f"stage {data['_stage_id']}" if len(stats) > 1 else None
        ax.hist(values, bins=bins, alpha=0.6, label=label, density=False)

    if not all_values:
        print("No raw_kvc_hit_tokens data found.")
        return

    arr = np.array(all_values)
    mean_val = float(arr.mean())
    median_val = float(np.median(arr))
    p99_val = float(np.percentile(arr, 99))
    zero_pct = float((arr == 0).mean()) * 100.0
    hit_pct = 100.0 - zero_pct

    ax.axvline(mean_val, color="black", linestyle=":", linewidth=1.5, label=f"mean = {mean_val:,.0f} tokens")
    ax.axvline(median_val, color="blue", linestyle="--", linewidth=1.5, label=f"median = {median_val:,.0f} tokens")
    ax.axvline(p99_val, color="red", linestyle="--", linewidth=1.5, label=f"P99 = {p99_val:,.0f} tokens")

    ax.set_xlabel("KVC prefix-hit tokens per request", fontsize=12)
    ax.set_ylabel("Number of requests", fontsize=12)
    ax.set_title(
        f"{title}\nKVC prefix-hit token distribution  " f"(n={len(arr):,}, {hit_pct:.1f}% cache hits)",
        fontsize=13,
        pad=10,
    )
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Plot per-request KVC prefix-hit token distribution from stage_*/opal_stats.json."
    )
    p.add_argument("run_dir", help="Simulation run directory (containing stage_*/opal_stats.json).")
    p.add_argument(
        "--stage-id", type=int, default=None, help="Restrict to a single stage_id. Default: all stages combined."
    )
    p.add_argument("--bins", type=int, default=60, help="Number of histogram bins (default: 60).")
    p.add_argument("--output-dir", default=None, help="Output directory for the plot. Defaults to <run-dir>/plots/.")
    p.add_argument("--title", default=None, help="Plot title prefix. Defaults to the run directory name.")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    stats = load_stage_stats(run_dir, args.stage_id)
    if not stats:
        print(f"No opal_stats.json found under {run_dir}.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    title = args.title or run_dir.name
    suffix = f"_stage{args.stage_id}" if args.stage_id is not None else ""

    plot_kvc_hit_tokens(
        stats,
        output_dir / f"{run_dir.name}{suffix}_kvc_hit_tokens.jpg",
        title,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
