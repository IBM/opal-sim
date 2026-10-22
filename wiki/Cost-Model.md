# Cost Model

The cost layer (`opal/cost/`) prices a finished simulation run: what the infrastructure Opal simulated would have cost to rent, per stage, with a breakdown and plots. It is **opt-in**: it runs only when the config has a `cost` section.

The cost is a **lower bound**: the least a cloud would charge for exactly what Opal simulated. Instances start and stop instantly, each is billed for exactly its run time (unless [billing granularity](#billing-granularity) is set). For now it bills the GPU instances only.

## Architecture

```
 Opal run                          Price profile
 (stats + config)                  (cost-profiles/)
       |                                 |
       v                                 v
   1. Usage  --->  3. Policy  <---  2. Profile
                       |
                       v
                   4. Report  --->  opal_cost.json, console summary, plots
```

Plain functions in `opal/cost/cost_model.py` (plots in `opal/cost/plots.py`); `python -m opal.cost` (`opal/cost/__main__.py`) is the offline entry point.

- **Usage** (`stage_usage()`) reads the stage statistics (`StageStatistics.to_dict()`, the same content as `opal_stats.json`) and the `worker.hw` config; no prices.
- **Profile** (`gpu_instances()`) is a dated data file of instance prices; a new provider, region or date is a new file.
- **Policy** is one way of computing cost: a function registered by name in `POLICIES`, e.g. `instance_rental()`.
- **Report** (`price_run()`, `report_run_cost()`) is what gets written, printed and plotted.

The layer runs from `OpalSimulator.run()` right after `write_simulation_data()`.

## Configuration

Add a `cost` section to the simulation config, for example:

```json
"cost": {
  "profile": "cost-profiles/cloud/aws-<date>.json",
  "policy": "instance_rental",
  "region": "us-east-1",
  "pricing": "on_demand"
}
```

Every key, with its default, is in [[Configuration Simulation]]; `token_weights` is explained under [Per-session cost](#per-session-cost).

The simulated GPU is `worker.hw.gpu` (e.g. `"H100"`), matched case-insensitively against the profile's accelerator names; GPUs per worker is `worker.hw.tp`.

## Output

With `simulation.save_simulation_data` on, each stage gets `stage_N/opal_cost.json` next to `opal_stats.json`, and with `-g` the [plots](#plots). Main fields:

| Field | Meaning |
|-------|---------|
| `total_usd` | Cost of the stage. |
| `per_request_usd`, `per_1M_tokens_usd` | Cost per finished request and per 1M tokens (input + output). |
| `per_1M_tokens_by_kind_usd`, `token_weights` | Price per 1M output, recomputed input and cached input tokens, and the weights behind it (see [Price per 1M tokens](#price-per-1m-tokens)). |
| `breakdown_usd`, `gpu_hours` | Cost and GPU-hours of each bill line: `gpu_busy`, `gpu_idle`, `gpu_unused`, `billing_rounding`. They add up to the total. |
| `pricing`, `billing` | The instance chosen (type, region, price, how many) and the billing step. |
| `usage` | What the stage used: duration, GPU, workers, busy GPU-hours, requests and tokens. |
| `sessions` | Cost and time per session, agent traces only (see [Per-session cost](#per-session-cost)). |
| `profile`, `by_region` | The price profile used, and the stage priced in each of its regions, cheapest first. |

A summary per stage and the run total are printed after the stage summaries, and appended to `simulation.log` like them. For `configs/defaults.json` with the example profile from [Price profiles](#price-profiles); the two `sessions` lines are printed for agent traces only:

```
===== cost: stage_0 =====
setup     1 x p5.48xlarge (8 x H100 each), us-east-1, on_demand: 55.04 USD/h per instance
run       50.2 s = 0.014 h, 100 requests, 871,662 input (96% cached) + 7,872 output tokens
workers   1 x tp 1 = 1 of 8 rented GPUs; 8 GPUs x 0.014 h = 0.112 GPU-h rented
cost      0.7672 USD (0.007672 USD/request)
per 1M    29.579 USD output, 8.874 USD recomputed input, 0.296 USD cached input (token weights 1 / 0.3 / 0.01)
          line                                 GPU-h    share         USD
          GPU busy (running a batch)           0.012    10.7%      0.0819
          GPU idle (worker up, no batch)       0.002     1.8%      0.0140
          GPU unused (rented, no worker)       0.098    87.5%      0.6713
          total                                0.112   100.0%      0.7672
sessions  <n> sessions, cost per session: median <x> USD, p90 <x> USD, max <x> USD
          time per session (mean): <x> s open = <x> s at the LLM (queued or generating) + <x> s in gaps (tool or user), <x>% in gaps
===== cost: stage_1 =====
...
===== cost: run =====
total     1.5442 USD over 2 stage(s)
```

### Price per 1M tokens

The bill as a provider's price list: the bill is shared by weighted tokens, with the same `token_weights` as the [per-session split](#per-session-cost), so the prices times the stage's tokens add back up to `total_usd`:

```
weighted_tokens = token_weights.output * output_tokens
                + token_weights.recomputed_input * (input_tokens - cached_input_tokens)
                + token_weights.cached_input * cached_input_tokens

per_1M_tokens_by_kind_usd.<kind> = token_weights.<kind> * total_usd / weighted_tokens * 1e6
```

`cached_input_tokens` is the input served from the KV cache (`raw_kvc_hit_tokens`); the rest is recomputed.

## Plots

PDF files next to `opal_cost.json`, like Opal's own plots, only with the `-g` flag (`opal/main.py -g`, or `python -m opal.cost <run_dir> -g`):

| File | What it shows |
|------|---------------|
| `cost-breakdown.pdf` | The bill lines that cost something, in USD and as a share of the total. |
| `cost-gpu-busy.pdf` | The workers' GPU busy share over the stage, and its CDF per simulated second. |
| `cost-sessions.pdf` | Agent traces only: the CDF of the cost per session and of the share of session time in gaps. |

## Offline pricing

Price a finished run folder again, e.g. with another region, pricing model, profile or billing, without re-simulating:

```shell
uv run python -m opal.cost simulation-runs/sim-<timestamp>
uv run python -m opal.cost simulation-runs/sim-<timestamp> --region us-west-2 --pricing spot
uv run python -m opal.cost simulation-runs/sim-<timestamp> --billing-increment-s 3600 --billing-minimum-s 3600
uv run python -m opal.cost simulation-runs/sim-<timestamp> --profile cost-profiles/cloud/aws-<date>.json --no-write
```

It reads `sim_config.json` and every `stage_N/opal_stats.json`; command-line options override the config's `cost` section. Without `--no-write` it overwrites `stage_N/opal_cost.json`; with `-g` it also writes the plots.

## Policies

### `instance_rental`

Rented instance-hours times the instance price. Whole instances are billed, as a cloud does: a worker's `tp` GPUs (`worker.hw.tp`) must fit in one instance, and rented GPUs that hold no worker are billed as unused.

```
workers_per_instance = floor(gpus_per_instance / tp)
instances            = mean over the stage of ceil(workers / workers_per_instance)
usd_per_gpu_hour     = instance_usd_per_hour / gpus_per_instance

gpu_busy         = busy_gpu_hours                           # a batch is running
gpu_idle         = provisioned_gpu_hours - busy_gpu_hours   # a worker is up, no batch
gpu_unused       = max(0, instances * gpus_per_instance * hours - provisioned_gpu_hours)
billing_rounding = rounding_instance_hours * gpus_per_instance   # 0 unless billing_* is set

each line costs its GPU-hours * usd_per_gpu_hour, and they add up to
total_usd = (instances * hours + rounding_instance_hours) * instance_usd_per_hour
```

- **Workers and busy time** come from the router, sampled once per simulated second (see [[Router]]); `provisioned_gpu_hours = mean_workers * tp * hours`.
- **Instance type**: the one that hosts the workers for the lowest cost per hour, in `region`, or in the cheapest region if `region` is unset.

### Billing granularity

A cloud bills each instance in steps, with a minimum. With `billing_increment_s` and `billing_minimum_s` set, every instance's run time is raised to the minimum and rounded up to whole steps; the difference is the `billing_rounding` line. Published rules (October 2026):

| Cloud | Rule | `billing_increment_s` | `billing_minimum_s` |
|-------|------|-----------------------|---------------------|
| AWS EC2, Linux / Windows / RHEL / Ubuntu Pro ([pricing](https://aws.amazon.com/ec2/pricing/on-demand/)) | per second, 60 s minimum | `1` | `60` |
| AWS EC2, SUSE Linux Enterprise Server | full hours | `3600` | `3600` |
| Google Compute Engine, vCPUs / GPUs / memory ([pricing](https://cloud.google.com/compute/vm-instance-pricing)) | 1 minute minimum, then per second | `1` | `60` |
| Azure Virtual Machines ([pricing FAQ](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)) | full minutes, extra seconds not billed (rounds down) | `0` | `0` |

### Adding a policy

1. Write a function `my_policy(usage, profile, cost_config)` in `opal/cost/cost_model.py`. `usage` is the dict from `stage_usage()`, `profile` the loaded price profile, `cost_config` the `cost` section. Return at least `{"breakdown_usd": {line: cost}}`; any other keys are copied into the report as-is. The console summary and the plots also read `pricing`, `billing` and `gpu_hours`, as returned by `instance_rental`.
2. Register it in `POLICIES` and select it with `cost.policy`.

If a policy needs a quantity that `stage_usage()` doesn't return yet, add it there from the stage statistics.

## Per-session cost

For an agent trace (the OTel trace replay, where every request carries a `session_id`), each stage's bill is also split between sessions: `gpu_busy` by weighted tokens, the rest by the time each session is open. The per-session costs add up to the stage total.

```
session_cost    = gpu_busy * weighted_tokens / total_weighted_tokens
                + (gpu_idle + gpu_unused + billing_rounding) * open_s / total_open_s

weighted_tokens = token_weights.output * output_tokens
                + token_weights.recomputed_input * (input_tokens - cached_tokens)
                + token_weights.cached_input * cached_tokens

open_s = last_completion - first_arrival   # the session's first request to its last
gap_s  = open_s - at_llm_s                 # at_llm_s: time with a request queued or generating
```

A gap is time with no request at the LLM: a tool runs or the user thinks.

**`token_weights`** (default `{"output": 1.0, "recomputed_input": 0.3, "cached_input": 0.01}`): only the ratios matter. An uncached input token costs about 0.43–0.46 of an output token in large-scale DeepSeek serving ([LMSYS, 2025](https://lmsys.org/blog/2025-05-05-large-scale-ep/), [DeepSeek, 2025](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)) and 0.2–0.33 in list prices ([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing), October 2026); a cached one 0.005–0.02 in list prices. To charge output tokens only, set the other two weights to 0.

## Assumptions

- The instance is matched by `worker.hw.gpu` only. It is not checked that it has enough memory or local disk for the configured KV tiers; the timing model uses `worker.hw.tflops`/`mem_bw_TBps`, not the GPU name.
- The instance price includes its CPUs, memory and local disk; storage outside the instance is not billed.
- Each stage is billed as its own deployment, and a session that spans two stages is split per stage.

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

Prices are per hour for the whole instance, in US dollars (the report's `_usd` keys; a profile in another `currency` is rejected).

No profile ships with the repository. To try the cost layer, save the example above as a file (e.g. `cost-profiles/cloud/aws-example.json`) and point `cost.profile` to it: `configs/defaults.json` then runs on one `p5.48xlarge`, 1 of 8 GPUs used.

## See also

- [[Configuration Simulation]] — the `cost` keys next to every other config section.
- [[Statistics collection and processing]] — the stage statistics the cost layer reads: `per_unit_workers`, `per_unit_gpu_busy_time`, `raw_session_ids`, `arrival_completion_times`.
- [[Router]] — `_per_second_stats`, which samples the worker count and the GPU-busy seconds once per simulated second.
- [[Testing]] — what `tests/test_cost_model.py` covers.
