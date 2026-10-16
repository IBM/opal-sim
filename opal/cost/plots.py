# SPDX-License-Identifier: Apache-2.0
"""Plots of the cost layer, written as PDF next to opal_cost.json (wiki/Cost-Model.md, "Plots")."""

import os
import statistics

from matplotlib.figure import Figure

from opal.cost.cost_model import LINE_NAMES

# One fixed colour per bill line, the same in every plot; anything else (sessions) gets its own, so a session gap
# never looks like GPU idle.
COLORS = {"gpu_busy": "#2a78d6", "gpu_idle": "#eb6834", "gpu_unused": "#1baf7a", "billing_rounding": "#eda100"}
OTHER_COLOR = "#6250d6"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def save_figure(fig, path: str):
    fig.savefig(path, format="pdf", dpi=300, bbox_inches="tight")  # same as opal.stats.plot.save_plot


def usd(value: float) -> str:
    if value == 0:
        return "0 USD"
    return f"{value:.2f} USD" if abs(value) >= 1 else f"{value:.4f} USD"


def new_figure(title: str, rows: int = 1, cols: int = 1, size=(5.0, 3.6)):
    fig = Figure(figsize=(size[0] * cols, size[1] * rows), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(rows, cols, squeeze=False).flatten()
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(color=MUTED, labelcolor=INK, labelsize=8)
    fig.suptitle(title, color=INK, fontsize=10)
    return fig, axes


def plot_cdf(ax, values, color, xlabel: str, ylabel: str):
    values = sorted(values)
    ax.step(values, [100 * (i + 1) / len(values) for i in range(len(values))], where="post", color=color, linewidth=2)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_ylim(0, 101)


def gpu_busy_per_second(stage: dict) -> list[float]:
    """Share of each simulated second the workers' GPUs ran a batch (%). A batch counts in the second it ends, so
    one longer than a second shows up as one full second: values are capped at 100."""
    busy_pct = []
    for busy_s, workers in zip(stage["per_unit_gpu_busy_time"], stage["per_unit_workers"], strict=True):
        busy_pct.append(min(100.0, 100 * busy_s / workers) if workers else 0.0)
    return busy_pct


def plot_gpu_busy(stage: dict, report: dict, path: str):
    busy_pct = gpu_busy_per_second(stage)
    if not busy_pct:
        return  # a stage shorter than one simulated second has no samples
    u = report["usage"]
    fig, (over_time, cdf) = new_figure(
        f"stage_{report['stage']}: GPU busy over time"
        f" (workers {u['mean_workers']:g} x tp {u['gpus_per_worker']}, {u['busy_pct']:.0f}% busy overall)",
        cols=2,
    )
    # Long stages: mean over windows of `window` seconds, so the line stays readable (at most ~300 points).
    window = next((w for w in (1, 5, 10, 30, 60, 120, 300, 600, 1800) if len(busy_pct) / w <= 300), 3600)
    means = [statistics.fmean(busy_pct[i : i + window]) for i in range(0, len(busy_pct), window)]
    over_time.plot([window * (i + 1) for i in range(len(means))], means, color=COLORS["gpu_busy"], linewidth=1)
    over_time.set_xlabel("simulated time in the stage (s)", color=INK, fontsize=9)
    per = "in that second" if window == 1 else f"mean over {window} s"
    over_time.set_ylabel(f"GPU busy, {per} (%)", color=INK, fontsize=9)
    over_time.set_ylim(0, 105)
    plot_cdf(cdf, busy_pct, COLORS["gpu_busy"], "GPU busy in a second (%)", "seconds with at most that (%)")
    cdf.set_xlim(0, 100)
    no_batch_pct = 100 * busy_pct.count(0.0) / len(busy_pct)
    if no_batch_pct:
        cdf.annotate(
            f"{no_batch_pct:.0f}% of seconds with no batch",
            (0, no_batch_pct),
            xytext=(6, -14),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    cdf.axvline(u["busy_pct"], color=MUTED, linewidth=1)
    cdf.annotate(
        f"overall {u['busy_pct']:.0f}%",
        (u["busy_pct"], 4),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        color=INK,
        fontsize=8,
    )
    save_figure(fig, path)


def plot_breakdown(report: dict, path: str):
    p, total = report["pricing"], report["total_usd"]
    # Lines that cost nothing (no unused GPUs, no billing rounding) are left out, as in the console table.
    lines = [line for line, usd in report["breakdown_usd"].items() if usd]
    values = [report["breakdown_usd"][line] for line in lines]
    fig, (ax,) = new_figure(
        f"stage_{report['stage']}: {usd(total)} on {p['instances']:g} x {p['instance_type']},"
        f" {p['region']}, {p['pricing_model']}",
        size=(7.0, 2.8),
    )
    rows = list(range(len(lines)))[::-1]
    ax.barh(rows, values, height=0.6, color=[COLORS.get(line, MUTED) for line in lines])
    ax.set_yticks(rows, [LINE_NAMES.get(line, line) for line in lines])
    for row, value in zip(rows, values, strict=True):
        share = f" ({100 * value / total:.0f}%)" if total else ""
        ax.annotate(
            f"{usd(value)}{share}",
            (value, row),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=8,
        )
    ax.set_xlim(0, 1.3 * max(values, default=0) or 1)
    ax.set_xlabel("cost (USD)", color=INK, fontsize=9)
    ax.grid(axis="y", visible=False)
    save_figure(fig, path)


def plot_sessions(report: dict, path: str):
    s = report["sessions"]
    per_session = list(s["per_session"].values())
    fig, (cost, gaps) = new_figure(
        f"stage_{report['stage']}: {s['sessions']} sessions; gaps (no request at the LLM: a tool runs or the user"
        f" thinks) are {s['gap_pct']:.0f}% of session time",
        cols=2,
    )
    plot_cdf(cost, [x["cost_usd"] for x in per_session], OTHER_COLOR, "cost of a session (USD)", "sessions (%)")
    gap_pct = [100 * x["gap_s"] / x["open_s"] if x["open_s"] else 0.0 for x in per_session]
    plot_cdf(gaps, gap_pct, OTHER_COLOR, "share of the session's open time in gaps (%)", "sessions (%)")
    gaps.set_xlim(0, 100)
    save_figure(fig, path)


def plot_stage_cost(stage: dict, report: dict, stage_dir: str):
    """Write the stage's cost plots: cost-breakdown.pdf, cost-gpu-busy.pdf and, for agent traces, cost-sessions.pdf."""
    plot_breakdown(report, os.path.join(stage_dir, "cost-breakdown.pdf"))
    plot_gpu_busy(stage, report, os.path.join(stage_dir, "cost-gpu-busy.pdf"))
    if report["sessions"]:
        plot_sessions(report, os.path.join(stage_dir, "cost-sessions.pdf"))
