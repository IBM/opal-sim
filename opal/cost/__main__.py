# SPDX-License-Identifier: Apache-2.0
"""Price a finished run folder without re-simulating; options override the config's "cost" section.

Usage:
    python -m opal.cost simulation-runs/sim-<timestamp> [--region us-west-2] [--pricing spot] [--no-write]
"""

import argparse
import json
from pathlib import Path

from opal.cost.cost_model import report_run_cost

OPTIONS = ("profile", "policy", "region", "pricing", "instance_type")


def main(argv: list[str] | None = None) -> list[dict]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="run folder holding sim_config.json and stage_N/opal_stats.json")
    parser.add_argument("--profile", help="price profile JSON, e.g. cost-profiles/cloud/aws-<date>.json")
    parser.add_argument("--policy", help="cost policy, e.g. gpu_hour_rental")
    parser.add_argument("--region", help="region to price in (default: the cheapest one)")
    parser.add_argument("--pricing", help="on_demand or spot")
    parser.add_argument("--instance-type", dest="instance_type", help="restrict to one instance type")
    parser.add_argument("--no-write", action="store_true", help="only print, don't write opal_cost.json")
    args = parser.parse_args(argv)

    config = json.loads((args.run_dir / "sim_config.json").read_text())
    stats_files = sorted(args.run_dir.glob("stage_*/opal_stats.json"), key=lambda p: int(p.parent.name.split("_")[1]))
    if not stats_files:
        parser.error(f"no stage_N/opal_stats.json under {args.run_dir}")
    stages = [json.loads(p.read_text()) for p in stats_files]
    cost_config = dict(config.get("cost", {}))
    for option in OPTIONS:
        if getattr(args, option) is not None:
            cost_config[option] = getattr(args, option)
    return report_run_cost(stages, config, cost_config, output_dir=None if args.no_write else str(args.run_dir))


if __name__ == "__main__":
    main()
