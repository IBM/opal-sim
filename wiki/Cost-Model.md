# Cost Model

The cost layer (`opal/cost/`) prices a finished simulation run: what the infrastructure Opal simulated would have cost to rent, per stage, with a breakdown. It is **opt-in**: without a `cost` section in the config, Opal behaves exactly as before. Tracking issue: [#58](https://github.com/IBM/opal-sim/issues/58) (the cost part of [#54](https://github.com/IBM/opal-sim/issues/54)).

## Architecture

```
 Opal run                            Price profile (cost-profiles/)
 stage statistics + sim config       dated $ per instance and region
        |                                         |
        v                                         |
   1. Usage            what was provisioned       |
   stage_usage()       and used, no prices        |
        |                                         v
        +------------->  3. Policy  <------  2. Profile
                         gpu_hour_rental()   gpu_instances()
                         how usage and prices become money
                               |
                               v
                         4. Report  ->  stage_N/opal_cost.json + console summary
                         price_run(), report_run_cost()
```

All four pieces are plain functions in `opal/cost/cost_model.py`; `python -m opal.cost` (`opal/cost/__main__.py`) is the offline entry point. Each piece changes for a different reason, so each can be swapped alone:

- **Usage** knows Opal and nothing about prices. It reads the stage statistics (`StageStatistics.to_dict()`, the same content as `opal_stats.json`) and the `worker.hw` config.
- **Profile** knows prices and nothing about Opal. A new provider, region or date is a new data file, not new code.
- **Policy** is one way of computing cost, taken from a paper. Policies are pluggable: functions registered by name in `POLICIES`.
- **Report** is what gets written and printed.

Because usage and prices are separate, one simulated run can be priced in every region of a profile without re-simulating (the `by_region` list in the report).

The cost layer runs from `OpalSimulator.run()` right after `write_simulation_data()`. The same code also runs offline on a finished run folder (see [Offline pricing](#offline-pricing)).

## Configuration

Add a `cost` section to the simulation config, for example:

```json
"cost": {
  "profile": "cost-profiles/cloud/aws-<date>.json",
  "policy": "gpu_hour_rental",
  "region": "us-east-1",
  "pricing": "on_demand"
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | string | required | Price profile JSON. A relative path is resolved from the working directory, like other paths in the config. |
| `policy` | string | `gpu_hour_rental` | Cost policy (see [Policies](#policies)). Case-insensitive. |
| `region` | string | cheapest region | Region to price in. If omitted, the cheapest region with a price for the simulated GPU is used. |
| `pricing` | string | `on_demand` | `on_demand` or `spot`. Case-insensitive. |
| `instance_type` | string | cheapest instance | Restrict to one instance type (e.g. `p5.48xlarge`). |

The simulated GPU is `worker.hw.gpu` (e.g. `"H100"`), matched case-insensitively against the profile's accelerator names; GPUs per worker is `worker.hw.tp`.

## Output

With `simulation.save_simulation_data` on, each stage gets `stage_N/opal_cost.json` next to `opal_stats.json`. Main fields:

| Field | Meaning |
|-------|---------|
| `total`, `currency` | Cost of the stage. |
| `per_request`, `per_1M_tokens` | `total` divided by finished requests, and by input + output tokens (× 10⁶). `null` if zero. |
| `breakdown` | Cost per resource line; the lines add up to `total`. `gpu_hour_rental`: `gpu_busy` (the workers' GPUs while a batch runs), `gpu_idle` (the workers' GPUs with no batch running) and `gpu_unused` (rented with the instance but holding no worker). |
| `pricing` | Region, pricing model, instance type, GPUs per instance, instance price per hour, price per GPU-hour, mean instances rented, and `gpus_used` / `gpus_rented`. |
| `usage` | Stage duration, GPU, GPUs per worker, mean workers, provisioned and busy GPU-hours, busy share, requests and tokens. |
| `by_region` | The same usage priced in every region of the profile, cheapest first. |
| `profile` | Name, snapshot date, path and source of the price profile. |

One line per stage and the run total are printed after the stage summaries, and appended to `simulation.log` like them:

```
===== cost =====
stage_0: 0.0959 USD | 1 x p5.4xlarge, us-east-1, on_demand | 1 of 1 GPUs used, 85% busy
stage_1: 0.0971 USD | 1 x p5.4xlarge, us-east-1, on_demand | 1 of 1 GPUs used, 16% busy
total: 0.1930 USD
```

Per stage: what you pay, the instances rented (`1 x p5.4xlarge`; a fraction when the router scaled during the stage), region and pricing model, how many of the rented GPUs hold workers, and the share of the workers' GPU time spent running batches.

## Offline pricing

Price a finished run folder again, e.g. with another region, pricing model or profile, without re-simulating:

```shell
uv run python -m opal.cost simulation-runs/sim-<timestamp>
uv run python -m opal.cost simulation-runs/sim-<timestamp> --region us-west-2 --pricing spot
uv run python -m opal.cost simulation-runs/sim-<timestamp> --profile cost-profiles/cloud/aws-<date>.json --no-write
```

It reads `sim_config.json` and every `stage_N/opal_stats.json`; command-line options override the config's `cost` section. Without `--no-write` it overwrites `stage_N/opal_cost.json`.

## Policies

### `gpu_hour_rental`

The GPU line of the cost model in CachedAttention (Gao et al., USENIX ATC 2024): rented instance-hours × instance price. Whole instances are billed, as a cloud does.

```
cost   = instances × hours × instance price per hour              -> total
used   = GPU-hours of the workers × price per GPU-hour            (price per GPU-hour = instance price ÷ GPUs per instance)
busy   = busy GPU-hours of the workers × price per GPU-hour       -> breakdown.gpu_busy
idle   = used − busy                                              -> breakdown.gpu_idle
unused = cost − used                                              -> breakdown.gpu_unused
```

`instances` = workers ÷ workers per instance, rounded up, averaged over the stage if the number of workers changes (`pricing.instances`); workers per instance = GPUs per instance ÷ `worker.hw.tp`, rounded down. "X of Y GPUs used" compares the workers' GPUs with the GPUs of the rented instances.

- **Workers** come from the router, which records the number of active workers once per simulated second (`Router._per_second_stats`). A stage shorter than one second has no samples and uses `simulation.num_workers`.
- **Instances**: a worker's `worker.hw.tp` GPUs must fit in one instance (Opal doesn't model traffic between instances), so an instance holds ⌊GPUs per instance ÷ `tp`⌋ workers, and instance types with fewer GPUs than `tp` are skipped. For each sample, instances needed = ⌈workers ÷ workers per instance⌉; their mean over the stage × stage duration = instance-hours. This stays exact when the router adds workers.
- **Choice**: in each region, the instance type with the lowest cost per hour (instances × instance price; the same ranking as the total, and still defined for a stage that never ran); on a tie, the one needing fewer instances. With `region` unset, the cheapest region.
- **Busy, idle and unused**: the GPU-hours of the workers (mean workers × `tp` × duration) are split into `gpu_busy`, while a batch runs, and `gpu_idle`, the rest (e.g. every session waiting on a tool). `gpu_unused` is the rest of the bill, i.e. GPUs rented with the instance that hold no worker. All three are priced at instance price ÷ GPUs per instance. The console line shows `X of Y GPUs used` and the busy share.
- **Busy time** comes from the router, which records once per simulated second the GPU-busy seconds that all workers added in that second (`per_unit_gpu_busy_time`, see [[Statistics collection and processing]]). Their sum over the stage × `tp` = busy GPU-hours. A batch counts when it ends, so one that crosses a stage boundary lands in the later stage; busy is capped at the workers' GPU-hours.

Assumptions to keep in mind when reading the numbers:

- The instance is matched by `worker.hw.gpu` only. It is not checked that the instance has enough memory or local disk for the configured KV tiers (`kvc.CPUMemory`, `kvc.LocalNVMe`); the timing model uses `worker.hw.tflops`/`mem_bw_TBps`, not the GPU name.
- The instance price includes its CPUs, memory and local disk; there is no separate storage or network line yet.
- "Busy" means a batch is running, whatever its size: a batch of one request counts as much as a full one.
- Busy time after the run's last per-second sample (under a second) is not counted, so very short runs slightly overstate idle.

### Adding a policy

1. Write a function `my_policy(usage, profile, cost_config)` in `opal/cost/cost_model.py`. `usage` is the dict from `stage_usage()`, `profile` the loaded price profile, `cost_config` the `cost` section. Return `{"pricing": {...}, "breakdown": {line: cost}}`; any other keys are copied into the report as-is.
2. Register it in `POLICIES` and select it with `cost.policy`.
3. Cite the paper it reproduces in the docstring, add a test with hand-checkable numbers in `tests/test_cost_model.py`, and describe it in this page.

If a policy needs a quantity that `stage_usage()` doesn't return yet, add it there from the stage statistics.

## Price profiles

Profiles are dated JSON snapshots under `cost-profiles/`, so a run's cost is reproducible and no network access is needed at simulation time. A cloud profile looks like this:

```json
{
  "name": "aws",
  "kind": "cloud",
  "currency": "USD",
  "price_unit": "hour",
  "snapshot_date": "2026-10-09",
  "instances": {
    "p5.48xlarge": {
      "accelerator": "H100", "accelerator_count": 8, "vcpus": 192.0, "memory_gib": 2048.0,
      "prices": {"us-east-1": {"on_demand": 55.04, "spot": 26.9431}}
    }
  }
}
```

Prices are per hour for the whole instance. Add a new dated file rather than editing an old one, so earlier results stay reproducible.

## Planned

Following [#58](https://github.com/IBM/opal-sim/issues/58): storage cost (DRAM/SSD GB-hours per tier), per-session cost, a provisioning diagnosis (over/under-provisioned, bottleneck), on-prem TCO profiles, and pricing tool execution on CPU servers separately from the LLM on GPU servers.
