# SPDX-License-Identifier: Apache-2.0
"""Prices a finished run: stage usage + a price profile, combined by a pluggable policy (wiki/Cost-Model.md)."""

import json
import math
import os
import statistics
from collections import Counter

COST_FILE = "opal_cost.json"

# Weight of one token of each kind relative to an output token: splits the bill into a price per 1M tokens of each
# kind, and gpu_busy between sessions. Defaults from published serving measurements and list prices
# (wiki/Cost-Model.md, "Per-session cost").
TOKEN_WEIGHTS = {"output": 1.0, "recomputed_input": 0.3, "cached_input": 0.01}

# Labels of the bill's lines in the console summary and the plots.
LINE_NAMES = {
    "gpu_busy": "GPU busy (running a batch)",
    "gpu_idle": "GPU idle (worker up, no batch)",
    "gpu_unused": "GPU unused (rented, no worker)",
    "billing_rounding": "billing rounding",
}


def stage_usage(stage: dict, hw, initial_workers: int) -> dict:
    """What a stage provisioned and used. `stage` is StageStatistics.to_dict(), i.e. opal_stats.json."""
    duration = max(0.0, stage["stage_time_end"] - stage["stage_time_start"])
    # The router appends the active worker count once per simulated second (Router._per_second_stats).
    samples = stage["per_unit_workers"] or [initial_workers]
    mean_workers = sum(samples) / len(samples)
    gpus_per_worker = int(hw.get("tp", 1))
    provisioned = mean_workers * gpus_per_worker * duration / 3600
    # GPU-busy seconds added each simulated second, summed over workers. A batch counts when it ends,
    # so one crossing a stage boundary lands in the later stage: cap at what was provisioned.
    busy = min(sum(stage["per_unit_gpu_busy_time"]) * gpus_per_worker / 3600, provisioned)
    tokens = stage["input_output_tokens_sz"]
    return {
        "duration_s": duration,
        "gpu": str(hw["gpu"]),
        "gpus_per_worker": gpus_per_worker,
        "mean_workers": mean_workers,
        "workers_samples": Counter(samples),
        "provisioned_gpu_hours": provisioned,
        "busy_gpu_hours": busy,
        "busy_pct": 100 * busy / provisioned if provisioned else 0.0,
        "finished_requests": stage["finished_requests"],
        # Token counts can be numpy ints (workload generators); int() keeps the report JSON-serializable.
        "input_tokens": int(sum(t[0] for t in tokens)),
        "cached_input_tokens": int(sum(stage["raw_kvc_hit_tokens"])),
        "output_tokens": int(sum(t[1] for t in tokens)),
    }


def gpu_instances(profile: dict, gpu: str, pricing: str = "on_demand", instance_type: str | None = None) -> list[dict]:
    """Every instance type and region of the profile that offers `gpu` with a `pricing` price."""
    options = []
    for name, inst in profile["instances"].items():
        if inst["accelerator"].casefold() != gpu.casefold() or instance_type not in (None, name):
            continue
        for region, prices in inst["prices"].items():
            if pricing not in prices:
                continue
            options.append(
                {
                    "region": region,
                    "instance_type": name,
                    "gpus_per_instance": inst["accelerator_count"],
                    "instance_usd_per_hour": prices[pricing],
                    "usd_per_gpu_hour": prices[pricing] / inst["accelerator_count"],
                }
            )
    if not options:
        available = sorted({inst["accelerator"] for inst in profile["instances"].values()})
        raise ValueError(
            f"No '{pricing}' price for GPU '{gpu}'"
            + (f" on '{instance_type}'" if instance_type else "")
            + f" in price profile '{profile['name']}'. GPUs in this profile: {available}"
        )
    return options


def billing_rounding_hours(usage: dict, workers_per_instance: int, increment_s: float, minimum_s: float) -> float:
    """Instance-hours a cloud bills on top of the exact ones: each instance's run time is raised to `minimum_s`, then
    rounded up to whole `increment_s` steps (0 = exact). Opal only adds workers, so an instance runs from when it is
    first needed to the end of the stage."""
    samples_by_instances = Counter()
    for workers, n in usage["workers_samples"].items():
        samples_by_instances[math.ceil(workers / workers_per_instance)] += n
    total = sum(samples_by_instances.values())
    extra = 0.0
    for k in range(1, max(samples_by_instances, default=0) + 1):
        # Whole sample counts, divided once: a run of exactly 3600 s stays 3600.0, not 3600.0000001.
        run_s = usage["duration_s"] * sum(n for i, n in samples_by_instances.items() if i >= k) / total
        if run_s <= 0:
            continue
        billed_s = max(run_s, minimum_s)
        if increment_s > 0:
            billed_s = increment_s * math.ceil(billed_s / increment_s)
        extra += billed_s - run_s
    return extra / 3600


def instance_rental(usage: dict, profile: dict, cost_config) -> dict:
    """Instance-hours x instance price, as a cloud bills GPU instances.

    Whole instances are billed. The workers' GPUs are split into `gpu_busy` (running a batch) and `gpu_idle` (worker
    up, no batch); GPUs that are rented but hold no worker are `gpu_unused`. `billing_rounding` is what a cloud adds by
    billing in steps (`billing_increment_s`, `billing_minimum_s`); with both at 0 the bill is exact.
    """
    pricing = cost_config["pricing"]
    increment_s = cost_config["billing_increment_s"]
    minimum_s = cost_config["billing_minimum_s"]
    tp = usage["gpus_per_worker"]
    hours = usage["duration_s"] / 3600
    samples = usage["workers_samples"]

    def rank(option):
        # Cost per hour breaks ties in the bill, e.g. a stage that never ran costs 0 on every instance.
        cost_per_hour = option["mean_instances"] * option["instance_usd_per_hour"]
        return option["cost_usd"], cost_per_hour, option["mean_instances"], option["instance_type"]

    best = {}
    for option in gpu_instances(profile, usage["gpu"], pricing, cost_config["instance_type"]):
        # A worker's tp GPUs must sit in one instance: Opal doesn't model traffic between instances.
        workers_per_instance = option["gpus_per_instance"] // tp
        if workers_per_instance == 0:
            continue
        option["mean_instances"] = sum(
            n * math.ceil(workers / workers_per_instance) for workers, n in samples.items()
        ) / sum(samples.values())
        rounding = billing_rounding_hours(usage, workers_per_instance, increment_s, minimum_s)
        option["rounding_instance_hours"] = rounding
        option["cost_usd"] = (option["mean_instances"] * hours + rounding) * option["instance_usd_per_hour"]
        if option["region"] not in best or rank(option) < rank(best[option["region"]]):
            best[option["region"]] = option
    if not best:
        raise ValueError(
            f"No instance in price profile '{profile['name']}' has the {tp} '{usage['gpu']}' GPUs of a worker"
        )
    per_region = sorted(best.values(), key=lambda o: (rank(o), o["region"]))

    region = cost_config["region"] or per_region[0]["region"]
    chosen = next((o for o in per_region if o["region"] == region), None)
    if chosen is None:
        raise ValueError(
            f"GPU '{usage['gpu']}' has no '{pricing}' price in region '{region}' of price profile "
            f"'{profile['name']}'. Regions with a price: {sorted(o['region'] for o in per_region)}"
        )
    rented_gpu_hours = chosen["mean_instances"] * chosen["gpus_per_instance"] * hours
    rounding_gpu_hours = chosen["rounding_instance_hours"] * chosen["gpus_per_instance"]
    gpu_hours = {
        "gpu_busy": usage["busy_gpu_hours"],
        "gpu_idle": usage["provisioned_gpu_hours"] - usage["busy_gpu_hours"],
        "gpu_unused": max(0.0, rented_gpu_hours - usage["provisioned_gpu_hours"]),
        "billing_rounding": rounding_gpu_hours,
    }
    return {
        # Every line is priced at the instance's price per GPU-hour.
        "breakdown_usd": {line: h * chosen["usd_per_gpu_hour"] for line, h in gpu_hours.items()},
        "gpu_hours": gpu_hours,
        "pricing": {
            **chosen,
            "pricing_model": pricing,
            "gpus_used": usage["mean_workers"] * tp,
            "gpus_rented": chosen["mean_instances"] * chosen["gpus_per_instance"],
        },
        "billing": {"increment_s": increment_s, "minimum_s": minimum_s, "exact": increment_s == 0 and minimum_s == 0},
        "by_region": per_region,
    }


POLICIES = {"instance_rental": instance_rental}


def token_prices(total_usd: float, usage: dict, weights: dict) -> dict | None:
    """Price per 1M tokens of each kind, like a provider's price list: the bill split by weighted tokens, so the
    prices times the tokens add back up to the bill. None if the stage has no (weighted) tokens."""
    cached = usage["cached_input_tokens"]
    weighted = (
        weights["output"] * usage["output_tokens"]
        + weights["recomputed_input"] * (usage["input_tokens"] - cached)
        + weights["cached_input"] * cached
    )
    if not weighted:
        return None
    return {kind: w * total_usd / weighted * 1e6 for kind, w in weights.items()}


def at_llm_seconds(spans: list) -> float:
    """Length of the union of (arrival, completion) spans: the time a session has a request at the LLM (queued or
    generating)."""
    total, reach = 0.0, -math.inf
    for arrival, done in sorted(spans):
        if done > reach:
            total += done - max(arrival, reach)
            reach = done
    return total


def session_costs(stage: dict, breakdown_usd: dict, weights: dict) -> dict | None:
    """Split a stage's bill between agent sessions: `gpu_busy` by weighted tokens, the rest of the bill (GPU idle,
    GPU unused, billing rounding) by time open (first arrival to last completion). Also each session's time at the LLM
    (a request queued or generating) and in gaps (none: a tool runs or the user thinks). None if the stage has no
    session ids (only the OTel replay sets them).
    """
    ids = stage.get("raw_session_ids") or []
    if not any(ids):
        return None
    work, spans = Counter(), {}
    rows = zip(
        ids,
        stage["input_output_tokens_sz"],
        stage["raw_kvc_hit_tokens"],
        stage["arrival_completion_times"],
        strict=True,
    )
    for sid, (inp, out), cached, (arrival, done) in rows:
        work[sid] += (
            weights["output"] * out + weights["recomputed_input"] * (inp - cached) + weights["cached_input"] * cached
        )
        spans.setdefault(sid, []).append((arrival, done))
    per_session = {}
    for sid, s in spans.items():
        open_s = max(done for _, done in s) - min(arrival for arrival, _ in s)
        at_llm_s = at_llm_seconds(s)
        per_session[sid] = {"open_s": open_s, "at_llm_s": at_llm_s, "gap_s": open_s - at_llm_s}
    busy = breakdown_usd["gpu_busy"]
    not_busy = sum(breakdown_usd.values()) - busy
    total_work = sum(work.values())
    total_open_s = sum(s["open_s"] for s in per_session.values())
    for sid, s in per_session.items():
        s["cost_usd"] = busy * work[sid] / total_work + not_busy * s["open_s"] / total_open_s
    costs = sorted(s["cost_usd"] for s in per_session.values())
    return {
        "sessions": len(per_session),
        "median_usd": statistics.median(costs),
        "p90_usd": costs[math.ceil(0.9 * len(costs)) - 1],
        "max_usd": costs[-1],
        # Means, not medians: they add up (open = at the LLM + gap), and gap_pct = mean_gap_s / mean_open_s.
        "mean_open_s": total_open_s / len(per_session),
        "mean_at_llm_s": sum(s["at_llm_s"] for s in per_session.values()) / len(per_session),
        "mean_gap_s": sum(s["gap_s"] for s in per_session.values()) / len(per_session),
        "gap_pct": 100 * sum(s["gap_s"] for s in per_session.values()) / total_open_s,
        "per_session": per_session,
    }


def parse_cost_config(cost_config) -> dict:
    """The `cost` config section, checked once and with every default filled in: what the policies read."""
    if not cost_config.get("profile"):
        raise ValueError("The 'cost' config section needs a 'profile' (e.g. cost-profiles/cloud/aws-<date>.json)")
    policy = str(cost_config.get("policy", "instance_rental")).casefold()
    if policy not in POLICIES:
        raise ValueError(f"Unknown cost policy '{policy}'. Available: {sorted(POLICIES)}")
    pricing = str(cost_config.get("pricing", "on_demand")).casefold()
    if pricing not in ("on_demand", "spot"):
        raise ValueError(f"Unknown pricing model '{pricing}'. Available: ['on_demand', 'spot']")
    increment_s = float(cost_config.get("billing_increment_s", 0))
    minimum_s = float(cost_config.get("billing_minimum_s", 0))
    if increment_s < 0 or minimum_s < 0:
        raise ValueError("'billing_increment_s' and 'billing_minimum_s' are seconds >= 0 (0 = bill exactly)")
    weights = {**TOKEN_WEIGHTS, **(cost_config.get("token_weights") or {})}
    if set(weights) != set(TOKEN_WEIGHTS) or min(weights.values()) < 0 or max(weights.values()) == 0:
        raise ValueError(f"'token_weights' takes non-negative weights for {sorted(TOKEN_WEIGHTS)}, not all zero")
    return {
        "profile": cost_config["profile"],
        "policy": policy,
        "region": cost_config.get("region"),
        "pricing": pricing,
        "instance_type": cost_config.get("instance_type"),
        "billing_increment_s": increment_s,
        "billing_minimum_s": minimum_s,
        "token_weights": weights,
    }


def price_run(stages: list[dict], config) -> list[dict]:
    """Price every stage. `stages`: StageStatistics.to_dict() per stage; `config`: the simulation config."""
    cost_config = parse_cost_config(config["cost"])
    policy, weights = cost_config["policy"], cost_config["token_weights"]
    with open(cost_config["profile"]) as f:
        profile = json.load(f)
    if profile.get("currency", "USD") != "USD":
        raise ValueError(f"Price profile '{profile['name']}' is in {profile['currency']}: the cost layer reports USD")

    reports = []
    for i, stage in enumerate(stages):
        usage = stage_usage(stage, config["worker"]["hw"], config["simulation"]["num_workers"])
        priced = POLICIES[policy](usage, profile, cost_config)
        total = sum(priced["breakdown_usd"].values())
        tokens = usage["input_tokens"] + usage["output_tokens"]
        report = {
            "stage": i,
            "policy": policy,
            "total_usd": total,
            "per_request_usd": total / usage["finished_requests"] if usage["finished_requests"] else None,
            "per_1M_tokens_usd": total / tokens * 1e6 if tokens else None,
            "per_1M_tokens_by_kind_usd": token_prices(total, usage, weights),
            "token_weights": weights,
            **priced,
            "sessions": session_costs(stage, priced["breakdown_usd"], weights),
            "usage": usage,
            "profile": {
                "name": profile["name"],
                "snapshot_date": profile.get("snapshot_date"),
                "path": cost_config["profile"],
                "source": profile.get("source"),
            },
        }
        report["by_region"] = report.pop("by_region")  # the long list goes last, after what a reader looks for
        reports.append(report)
    return reports


def format_report(r: dict) -> list[str]:
    """The console summary of one stage's report, every number with its unit."""
    p, u, b = r["pricing"], r["usage"], r["billing"]
    hours = u["duration_s"] / 3600
    per_request = f"{r['per_request_usd']:.6f} USD/request" if r["per_request_usd"] is not None else "no requests"
    prices, w = r["per_1M_tokens_by_kind_usd"], r["token_weights"]
    if prices is None:
        per_1m = "no tokens"
    else:
        per_1m = (
            f"{prices['output']:.3f} USD output, {prices['recomputed_input']:.3f} USD recomputed input,"
            f" {prices['cached_input']:.3f} USD cached input"
            f" (token weights {w['output']:g} / {w['recomputed_input']:g} / {w['cached_input']:g})"
        )
    cached_pct = 100 * u["cached_input_tokens"] / u["input_tokens"] if u["input_tokens"] else 0.0
    lines = [
        f"===== cost: stage_{r['stage']} =====",
        f"setup     {p['mean_instances']:g} x {p['instance_type']} ({p['gpus_per_instance']} x {u['gpu']} each),"
        f" {p['region']}, {p['pricing_model']}: {p['instance_usd_per_hour']:.2f} USD/h per instance",
        f"run       {u['duration_s']:,.1f} s = {hours:.3f} h, {u['finished_requests']:,} requests,"
        f" {u['input_tokens']:,} input ({cached_pct:.0f}% cached) + {u['output_tokens']:,} output tokens",
        f"workers   {u['mean_workers']:g} x tp {u['gpus_per_worker']} = {p['gpus_used']:g} of"
        f" {p['gpus_rented']:g} rented GPUs; {p['gpus_rented']:g} GPUs x {hours:.3f} h ="
        f" {p['gpus_rented'] * hours:.3f} GPU-h rented",
        f"cost      {r['total_usd']:.4f} USD ({per_request})",
        f"per 1M    {per_1m}",
    ]
    if not b["exact"]:
        lines.append(f"billing   in {b['increment_s']:g} s steps, {b['minimum_s']:g} s minimum per instance")
    total = r["total_usd"]
    lines.append(f"          {'line':<32}{'GPU-h':>10}{'share':>9}{'USD':>12}")
    for line, usd in r["breakdown_usd"].items():
        name, gpu_h = LINE_NAMES.get(line, line), r["gpu_hours"][line]
        if usd == 0 and gpu_h == 0:
            continue  # e.g. no unused GPUs, or no billing rounding: the table stays short
        share = f"{100 * usd / total:.1f}%" if total else "-"
        lines.append(f"          {name:<32}{gpu_h:>10.3f}{share:>9}{usd:>12.4f}")
    share = "100.0%" if total else "-"
    lines.append(f"          {'total':<32}{sum(r['gpu_hours'].values()):>10.3f}{share:>9}{total:>12.4f}")
    s = r["sessions"]
    if s:
        lines.append(
            f"sessions  {s['sessions']} sessions, cost per session: median {s['median_usd']:.4f} USD,"
            f" p90 {s['p90_usd']:.4f} USD, max {s['max_usd']:.4f} USD"
        )
        lines.append(
            f"          time per session (mean): {s['mean_open_s']:,.0f} s open ="
            f" {s['mean_at_llm_s']:,.0f} s at the LLM (queued or generating) + {s['mean_gap_s']:,.0f} s in gaps"
            f" (tool or user), {s['gap_pct']:.0f}% in gaps"
        )
    return lines


def report_run_cost(
    stages: list[dict], config, output_dir: str | None = None, log_file=None, plot_graphs: bool = False
):
    """Price the run and print it (also to `log_file`). If `output_dir` is given, write stage_N/opal_cost.json and,
    with `plot_graphs` (the -g flag), the stage's cost plots next to it."""
    reports = price_run(stages, config)
    lines = []
    for stage, r in zip(stages, reports, strict=True):
        lines += format_report(r)
        if output_dir is not None:
            stage_dir = os.path.join(output_dir, f"stage_{r['stage']}")
            with open(os.path.join(stage_dir, COST_FILE), "w") as f:
                json.dump(r, f, indent=2)
            if plot_graphs:
                # Imported here: matplotlib is only needed when plots are written.
                from opal.cost.plots import plot_stage_cost

                plot_stage_cost(stage, r, stage_dir)
    lines.append("===== cost: run =====")
    lines.append(f"total     {sum(r['total_usd'] for r in reports):.4f} USD over {len(reports)} stage(s)")
    for line in lines:
        print(line)
        if log_file is not None:
            print(line, file=log_file)
    return reports
