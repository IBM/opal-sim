# SPDX-License-Identifier: Apache-2.0
"""Prices a finished run: stage usage + a price profile, combined by a pluggable policy (wiki/Cost-Model.md)."""

import json
import math
import os
from collections import Counter

COST_FILE = "opal_cost.json"


def stage_usage(stage: dict, hw, initial_workers: int) -> dict:
    """What a stage provisioned and used. `stage` is StageStatistics.to_dict(), i.e. opal_stats.json."""
    duration = max(0.0, stage["stage_time_end"] - stage["stage_time_start"])
    # The router appends the active worker count once per simulated second (Router._per_second_stats).
    samples = stage["per_unit_workers"] or [initial_workers]
    mean_workers = sum(samples) / len(samples)
    gpus_per_worker = int(hw.get("tp", 1))
    tokens = stage["input_output_tokens_sz"]
    return {
        "duration_s": duration,
        "gpu": str(hw["gpu"]),
        "gpus_per_worker": gpus_per_worker,
        "mean_workers": mean_workers,
        "workers_time_share": {w: n / len(samples) for w, n in Counter(samples).items()},
        "provisioned_gpu_hours": mean_workers * gpus_per_worker * duration / 3600,
        "finished_requests": stage["finished_requests"],
        # Token counts can be numpy ints (workload generators); int() keeps the report JSON-serializable.
        "input_tokens": int(sum(t[0] for t in tokens)),
        "output_tokens": int(sum(t[1] for t in tokens)),
    }


def gpu_instances(profile: dict, gpu: str, pricing: str = "on_demand", instance_type: str | None = None) -> list[dict]:
    """Every instance type and region of the profile that offers `gpu` with a `pricing` price."""
    if pricing not in ("on_demand", "spot"):
        raise ValueError(f"Unknown pricing model '{pricing}'. Available: ['on_demand', 'spot']")
    options = [
        {
            "region": region,
            "instance_type": name,
            "gpus_per_instance": inst["accelerator_count"],
            "instance_price_per_hour": prices[pricing],
            "price_per_gpu_hour": prices[pricing] / inst["accelerator_count"],
        }
        for name, inst in profile["instances"].items()
        if inst["accelerator"].casefold() == gpu.casefold() and instance_type in (None, name)
        for region, prices in inst["prices"].items()
        if pricing in prices
    ]
    if not options:
        available = sorted({inst["accelerator"] for inst in profile["instances"].values()})
        raise ValueError(
            f"No '{pricing}' price for GPU '{gpu}'"
            + (f" on '{instance_type}'" if instance_type else "")
            + f" in price profile '{profile['name']}'. GPUs in this profile: {available}"
        )
    return options


def gpu_hour_rental(usage: dict, profile: dict, cost_config) -> dict:
    """Instance-hours x instance price: the GPU line of CachedAttention's bill (USENIX ATC'24).

    Whole instances are billed; GPUs that are rented but hold no worker are reported as `gpu_unused`.
    """
    pricing = str(cost_config.get("pricing", "on_demand")).casefold()
    tp = usage["gpus_per_worker"]
    hours = usage["duration_s"] / 3600

    def rank(option):
        # Cost per hour, so a zero-length stage still picks the right instance; then fewer instances, then the name.
        return option["instances"] * option["instance_price_per_hour"], option["instances"], option["instance_type"]

    best = {}
    for option in gpu_instances(profile, usage["gpu"], pricing, cost_config.get("instance_type")):
        # A worker's tp GPUs must sit in one instance: Opal doesn't model traffic between instances.
        workers_per_instance = option["gpus_per_instance"] // tp
        if workers_per_instance == 0:
            continue
        option["instances"] = sum(
            share * math.ceil(workers / workers_per_instance) for workers, share in usage["workers_time_share"].items()
        )
        option["cost"] = option["instances"] * hours * option["instance_price_per_hour"]
        if option["region"] not in best or rank(option) < rank(best[option["region"]]):
            best[option["region"]] = option
    if not best:
        raise ValueError(
            f"No instance in price profile '{profile['name']}' has the {tp} '{usage['gpu']}' GPUs of a worker"
        )
    per_region = sorted(best.values(), key=lambda o: (rank(o)[0], o["region"]))

    region = cost_config.get("region") or per_region[0]["region"]
    chosen = next((o for o in per_region if o["region"] == region), None)
    if chosen is None:
        raise ValueError(
            f"GPU '{usage['gpu']}' has no '{pricing}' price in region '{region}' of price profile "
            f"'{profile['name']}'. Regions with a price: {sorted(o['region'] for o in per_region)}"
        )
    used = usage["provisioned_gpu_hours"] * chosen["price_per_gpu_hour"]
    return {
        "pricing": {
            **chosen,
            "pricing_model": pricing,
            "gpus_used": usage["mean_workers"] * tp,
            "gpus_rented": chosen["instances"] * chosen["gpus_per_instance"],
        },
        "breakdown": {"gpu_used": used, "gpu_unused": max(0.0, chosen["cost"] - used)},
        "by_region": per_region,
    }


POLICIES = {"gpu_hour_rental": gpu_hour_rental}


def price_run(stages: list[dict], config, cost_config) -> list[dict]:
    """Price every stage. `stages`: StageStatistics.to_dict() per stage; `config`: the simulation config."""
    if not cost_config.get("profile"):
        raise ValueError("The 'cost' config section needs a 'profile' (e.g. cost-profiles/cloud/aws-<date>.json)")
    policy = str(cost_config.get("policy", "gpu_hour_rental")).casefold()
    if policy not in POLICIES:
        raise ValueError(f"Unknown cost policy '{policy}'. Available: {sorted(POLICIES)}")
    with open(cost_config["profile"]) as f:
        profile = json.load(f)

    reports = []
    for i, stage in enumerate(stages):
        usage = stage_usage(stage, config["worker"]["hw"], config["simulation"]["num_workers"])
        priced = POLICIES[policy](usage, profile, cost_config)
        total = sum(priced["breakdown"].values())
        tokens = usage["input_tokens"] + usage["output_tokens"]
        reports.append(
            {
                "stage": i,
                "policy": policy,
                "currency": profile.get("currency", "USD"),
                "total": total,
                "per_request": total / usage["finished_requests"] if usage["finished_requests"] else None,
                "per_1M_tokens": total / tokens * 1e6 if tokens else None,
                **priced,
                "usage": usage,
                "profile": {
                    "name": profile["name"],
                    "snapshot_date": profile.get("snapshot_date"),
                    "path": cost_config["profile"],
                    "source": profile.get("source"),
                },
            }
        )
    return reports


def report_run_cost(stages: list[dict], config, cost_config, output_dir: str | None = None, log_file=None):
    """Price the run, print it (also to `log_file`) and, if `output_dir` is given, write stage_N/opal_cost.json."""
    reports = price_run(stages, config, cost_config)
    lines = ["===== cost ====="]
    for r in reports:
        p = r["pricing"]
        lines.append(
            f"stage_{r['stage']}: {r['total']:.4f} {r['currency']}"
            f" | {p['instances']:g} x {p['instance_type']}, {p['region']}, {p['pricing_model']}"
            f" | {p['gpus_used']:g} of {p['gpus_rented']:g} GPUs used"
        )
        if output_dir is not None:
            with open(os.path.join(output_dir, f"stage_{r['stage']}", COST_FILE), "w") as f:
                json.dump(r, f, indent=2)
    lines.append(f"total: {sum(r['total'] for r in reports):.4f} {reports[0]['currency']}")
    for line in lines:
        print(line)
        if log_file is not None:
            print(line, file=log_file)
    return reports
