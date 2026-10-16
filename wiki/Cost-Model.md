# Cost Model

The cost layer (`opal/cost/`) prices a finished simulation run: what the infrastructure Opal simulated would have cost to rent, per stage, with a breakdown and plots. It is **opt-in**: without a `cost` section in the config, Opal behaves exactly as before. Tracking issue: [#58](https://github.com/IBM/opal-sim/issues/58) (the cost part of [#54](https://github.com/IBM/opal-sim/issues/54)).

**The cost is a lower bound.** It is the least a cloud would charge to run exactly what Opal simulated, in an ideal world:

- instances start and stop instantly: no boot time is billed, and a worker the router adds is billed from the second it serves;
- each instance is billed for exactly its run time, with no minimum (unless [billing granularity](#billing-granularity) is set);
- workers are packed densely into instances, and the cheapest instance type and region are picked;
- only the GPU instances are billed: storage, network and CPU for tool calls have no line yet ([Planned](#planned)).

A real bill is this number plus what the ideal world leaves out. Setting `billing_increment_s` / `billing_minimum_s` adds the first of those, the rounding of a cloud's billing steps.

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
                         4. Report  ->  stage_N/opal_cost.json + console summary + plots
                         price_run(), report_run_cost(), format_report(), plots.py
```

The four pieces are plain functions in `opal/cost/cost_model.py`; the plots are in `opal/cost/plots.py`. `python -m opal.cost` (`opal/cost/__main__.py`) is the offline entry point. Each piece changes for a different reason, so each can be swapped alone:

- **Usage** knows Opal and nothing about prices. It reads the stage statistics (`StageStatistics.to_dict()`, the same content as `opal_stats.json`) and the `worker.hw` config.
- **Profile** knows prices and nothing about Opal. A new provider, region or date is a new data file, not new code.
- **Policy** is one way of computing cost, taken from a paper. Policies are pluggable: functions registered by name in `POLICIES`.
- **Report** is what gets written, printed and plotted.

Because usage and prices are separate, one simulated run can be priced in every region of a profile without re-simulating (the `by_region` list in the report).

The cost layer runs from `OpalSimulator.run()` right after `write_simulation_data()`. The `cost` section is checked earlier, when `OpalSimulator` is built, by pricing an empty stage: a wrong profile, policy, pricing model, region or billing setting fails before the run instead of after it. The same code also runs offline on a finished run folder (see [Offline pricing](#offline-pricing)).

## Terms

Two different things wait in an agent trace, and the report keeps them apart:

| Term | Meaning | In the report |
|------|---------|---------------|
| GPU busy | A worker's GPUs are running a batch. | `breakdown_usd.gpu_busy`, `gpu_hours.gpu_busy` |
| GPU idle | A worker is up, but its GPUs have no batch to run. | `breakdown_usd.gpu_idle`, `gpu_hours.gpu_idle` |
| GPU unused | GPUs rented with an instance that hold no worker. | `breakdown_usd.gpu_unused`, `gpu_hours.gpu_unused` |
| Session open | From a session's first request arriving to its last one completing. | `sessions.per_session.<id>.open_s` |
| Session at the LLM | At least one of the session's requests is at the LLM: queued or generating. | `at_llm_s` |
| Session gap | The session is open but has no request at the LLM: a tool runs or the user thinks. | `gap_s`, `sessions.gap_pct` |

GPU idle is about the hardware, session gaps about one session. A session in a gap doesn't make the GPUs idle while other sessions keep them busy; the GPUs are idle only when no batch runs at all.

## Configuration

Add a `cost` section to the simulation config, for example:

```json
"cost": {
  "profile": "cost-profiles/cloud/aws-<date>.json",
  "policy": "gpu_hour_rental",
  "region": "us-east-1",
  "pricing": "on_demand",
  "billing_increment_s": 1,
  "billing_minimum_s": 60
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `profile` | string | required | Price profile JSON, in USD. A relative path is resolved from the working directory, like other paths in the config. |
| `policy` | string | `gpu_hour_rental` | Cost policy (see [Policies](#policies)). Case-insensitive. |
| `region` | string | cheapest region | Region to price in. If omitted, the cheapest region with a price for the simulated GPU is used. |
| `pricing` | string | `on_demand` | `on_demand` or `spot`. Case-insensitive. |
| `instance_type` | string | cheapest instance | Restrict to one instance type (e.g. `p5.48xlarge`). |
| `billing_increment_s` | number | `0` | Billing step in seconds: `1` per second, `60` per minute, `3600` per hour; `0` bills exactly (see [Billing granularity](#billing-granularity)). |
| `billing_minimum_s` | number | `0` | Minimum billed time per instance, in seconds. |
| `token_weights` | object | `{"output": 1, "recomputed_input": 0.3, "cached_input": 0.01}` | Weight of one token of each kind relative to an output token: splits the bill into a [price per 1M tokens](#price-per-1m-tokens) of each kind, and `gpu_busy` between sessions ([Per-session cost](#per-session-cost)). Kinds left out keep their default. |

The simulated GPU is `worker.hw.gpu` (e.g. `"H100"`), matched case-insensitively against the profile's accelerator names; GPUs per worker is `worker.hw.tp`.

## Output

With `simulation.save_simulation_data` on, each stage gets `stage_N/opal_cost.json` and the [plots](#plots) next to `opal_stats.json`. Every key carries its unit: `_usd` (US dollars), `_s` (seconds), `_hours` / `gpu_hours` (GPU-hours), `_pct` (percent, 0-100). Main fields, in file order:

| Field | Meaning |
|-------|---------|
| `total_usd` | Cost of the stage. |
| `per_request_usd`, `per_1M_tokens_usd` | `total_usd / finished_requests` and `total_usd / (input_tokens + output_tokens) * 1e6`; `null` if the divisor is 0. |
| `per_1M_tokens_by_kind_usd`, `token_weights` | Price per 1M `output`, `recomputed_input` and `cached_input` tokens, like a provider's price list (see [Price per 1M tokens](#price-per-1m-tokens)), and the weights used; `null` without tokens. |
| `breakdown_usd` | Cost per line; the lines add up to `total_usd`. `gpu_hour_rental`: `gpu_busy`, `gpu_idle`, `gpu_unused` (see [Terms](#terms)) and `billing_rounding`. |
| `gpu_hours` | The same lines in GPU-hours; each line costs `gpu_hours * usd_per_gpu_hour`. |
| `pricing` | Region, pricing model, instance type, GPUs per instance, `instance_usd_per_hour`, `usd_per_gpu_hour`, mean `instances` rented, `rounding_instance_hours`, `cost_usd`, and `gpus_used` / `gpus_rented`. |
| `billing` | `increment_s`, `minimum_s`, and `exact` (both 0: the lower bound). |
| `sessions` | Agent traces only, else `null` (see [Per-session cost](#per-session-cost)). |
| `usage` | Stage duration (`duration_s`), GPU, GPUs per worker, mean workers, provisioned and busy GPU-hours, `busy_pct`, requests, and input / cached input / output tokens. |
| `profile` | Name, snapshot date, path and source of the price profile. |
| `by_region` | The same usage priced in every region of the profile, cheapest first (`cost_usd`). |

A summary per stage and the run total are printed after the stage summaries, and appended to `simulation.log` like them. For `configs/defaults.json` with the example profile from [Price profiles](#price-profiles):

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
===== cost: stage_1 =====
...
===== cost: run =====
total     1.5442 USD over 2 stage(s)
```

`setup` is the instances rented (a fraction when the router scaled during the stage), their GPUs, region, pricing model and prices; `run` the stage's wall-clock duration and tokens (`cached` = input served from the KV cache); `per 1M` the [price per 1M tokens](#price-per-1m-tokens) of each kind; `workers` how many of the rented GPUs hold workers, and the GPU-hours rented (`gpus_rented * hours`). The table splits those GPU-hours into the bill lines: GPU-hours are GPUs times hours (8 GPUs busy for 0.4 h are 3.2 GPU-h), and each line gets the same share of the instance bill (`gpu_hours * usd_per_gpu_hour`, i.e. `total_usd * line_gpu_hours / total_gpu_hours`). Lines that cost nothing get no row (e.g. `GPU unused` when every rented GPU holds a worker, `billing rounding` with exact billing). With `billing_increment_s` or `billing_minimum_s` set, a `billing   in <x> s steps, <x> s minimum per instance` line follows `cost`, and the `billing rounding` row makes the total exceed the GPU-hours rented. For an agent trace, two more lines give the cost per session and how the mean session spends its time (means add up; medians wouldn't):

```
sessions  <n> sessions, cost per session: median <x> USD, p90 <x> USD, max <x> USD
          time per session (mean): <x> s open = <x> s at the LLM (queued or generating) + <x> s in gaps (tool or user), <x>% in gaps
```

### Price per 1M tokens

The bill as a provider's price list: what one million tokens of each kind cost, such that the prices times the stage's tokens add back up to `total_usd`. The bill is shared by weighted tokens, with the same `token_weights` as the [per-session split](#per-session-cost):

```
weighted_tokens = token_weights.output * output_tokens
                + token_weights.recomputed_input * (input_tokens - cached_input_tokens)
                + token_weights.cached_input * cached_input_tokens

per_1M_tokens_by_kind_usd.<kind> = token_weights.<kind> * total_usd / weighted_tokens * 1e6
```

`cached_input_tokens` is the input served from the KV cache (`raw_kvc_hit_tokens`), the rest of the input is recomputed. Unlike `per_1M_tokens_usd` (the bill over all tokens), these prices tell input and output apart, which matters when most of the input is cached.

## Plots

Written as PNG next to `opal_cost.json`, whenever the report is written (a run with `save_simulation_data` on, or `python -m opal.cost` without `--no-write`). They don't depend on Opal's `-g` flag.

| File | What it shows |
|------|---------------|
| `cost-breakdown.png` | The bill lines of `breakdown_usd` that cost something, in USD and as a share of the total. |
| `cost-gpu-busy.png` | Left: the workers' GPU busy share over the stage (on long stages, the mean over windows of up to 3600 s, so at most ~300 points). Right: the CDF of the per-second busy share, with the stage's overall busy share and the share of seconds with no batch. |
| `cost-sessions.png` | Agent traces only: the CDF of the cost per session, and the CDF of each session's share of open time spent in gaps. |

```
busy_pct[t] = min(100, 100 * per_unit_gpu_busy_time[t] / per_unit_workers[t])   # one value per simulated second
```

A batch counts in the second it ends (`per_unit_gpu_busy_time`), so a batch longer than one second shows as a full second plus earlier seconds with less: per-second values are capped at 100. Colours are fixed per line (busy, idle, unused, rounding), and anything that is not a bill line (sessions) has its own colour, so a session gap is never drawn like GPU idle.

## Offline pricing

Price a finished run folder again, e.g. with another region, pricing model, profile or billing, without re-simulating:

```shell
uv run python -m opal.cost simulation-runs/sim-<timestamp>
uv run python -m opal.cost simulation-runs/sim-<timestamp> --region us-west-2 --pricing spot
uv run python -m opal.cost simulation-runs/sim-<timestamp> --billing-increment-s 3600 --billing-minimum-s 3600
uv run python -m opal.cost simulation-runs/sim-<timestamp> --profile cost-profiles/cloud/aws-<date>.json --no-write
```

It reads `sim_config.json` and every `stage_N/opal_stats.json`; command-line options override the config's `cost` section. Without `--no-write` it overwrites `stage_N/opal_cost.json` and the plots.

## Policies

### `gpu_hour_rental`

The GPU line of the cost model in CachedAttention (Gao et al., USENIX ATC 2024): rented instance-hours times the instance price. Whole instances are billed, as a cloud does.

```
workers_per_instance = floor(gpus_per_instance / tp)
instances            = mean over the stage of ceil(workers / workers_per_instance)   # pricing.instances
usd_per_gpu_hour     = instance_usd_per_hour / gpus_per_instance

gpu_busy         = busy_gpu_hours                                    # gpu_hours.*; each line costs
gpu_idle         = provisioned_gpu_hours - busy_gpu_hours            #   gpu_hours * usd_per_gpu_hour
gpu_unused       = max(0, instances * gpus_per_instance * hours - provisioned_gpu_hours)
billing_rounding = rounding_instance_hours * gpus_per_instance       # 0 unless billing_* is set

total_usd = (instances * hours + rounding_instance_hours) * instance_usd_per_hour
```

`tp` is `worker.hw.tp`; `provisioned_gpu_hours` and `busy_gpu_hours` are in `usage` (see below). "X of Y rented GPUs" compares the workers' GPUs (`mean_workers * tp`) with the GPUs of the rented instances (`instances * gpus_per_instance`).

- **Workers** come from the router, which records the number of active workers once per simulated second (`Router._per_second_stats`). A stage shorter than one second has no samples and uses `simulation.num_workers`.
- **Instances**: a worker's `worker.hw.tp` GPUs must fit in one instance (Opal doesn't model traffic between instances), so an instance holds `floor(gpus_per_instance / tp)` workers, and instance types with fewer GPUs than `tp` are skipped. Each per-second sample needs `ceil(workers / workers_per_instance)` instances; `instances` is their mean over the stage and `instances * hours` the instance-hours. This stays exact when the router adds workers.
- **Choice**: in each region, the instance type with the lowest cost per hour (`instances * instance_usd_per_hour`, still defined for a stage that never ran); on a tie, the one needing fewer instances. With `region` unset, the cheapest region (`cost_usd`, rounding included).
- **Busy, idle and unused**: the GPU-hours of the workers (`provisioned_gpu_hours = mean_workers * tp * duration / 3600`) are split into `gpu_busy`, while a batch runs, and `gpu_idle`, the rest. `gpu_unused` is the GPUs rented with the instance that hold no worker. All lines are priced at `usd_per_gpu_hour`.
- **Busy time** comes from the router, which records once per simulated second the GPU-busy seconds that all workers added in that second (`per_unit_gpu_busy_time`, see [[Statistics collection and processing]]). `busy_gpu_hours = sum(per_unit_gpu_busy_time) * tp / 3600`. A batch counts when it ends, so one that crosses a stage boundary lands in the later stage; `busy_gpu_hours` is capped at `provisioned_gpu_hours`.

Assumptions to keep in mind when reading the numbers:

- The instance is matched by `worker.hw.gpu` only. It is not checked that the instance has enough memory or local disk for the configured KV tiers (`kvc.CPUMemory`, `kvc.LocalNVMe`); the timing model uses `worker.hw.tflops`/`mem_bw_TBps`, not the GPU name.
- The instance price includes its CPUs, memory and local disk; there is no separate storage or network line yet.
- "Busy" means a batch is running, whatever its size: a batch of one request counts as much as a full one.
- Busy time after the run's last per-second sample (under a second) is not counted, so very short runs slightly overstate idle.

### Billing granularity

A cloud bills each instance in steps, with a minimum. With `billing_increment_s` and `billing_minimum_s` set, every instance's run time is rounded up, and the difference is the `billing_rounding` line:

```
run_s[k]   = duration * (share of the stage needing at least k instances)   # instance k, k = 1, 2, ...
billed_s[k] = ceil(max(run_s[k], billing_minimum_s) / billing_increment_s) * billing_increment_s
rounding_instance_hours = sum(billed_s[k] - run_s[k]) / 3600
```

With `billing_increment_s = 0`, `billed_s[k] = max(run_s[k], billing_minimum_s)`. The router only adds workers, so instance `k` runs in one piece, from when it is first needed to the end of the stage. Published rules (October 2026):

| Cloud | Rule | `billing_increment_s` | `billing_minimum_s` |
|-------|------|-----------------------|---------------------|
| AWS EC2, Linux / Windows / RHEL / Ubuntu Pro ([pricing](https://aws.amazon.com/ec2/pricing/on-demand/)) | per second, 60 s minimum | `1` | `60` |
| AWS EC2, SUSE Linux Enterprise Server | full hours | `3600` | `3600` |
| Google Compute Engine, vCPUs / GPUs / memory ([pricing](https://cloud.google.com/compute/vm-instance-pricing)) | 1 minute minimum, then per second | `1` | `60` |
| Azure Virtual Machines ([pricing FAQ](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/)) | full minutes; the extra seconds are not billed | `0` | `0` |

Azure rounds down, so its bill is at most the exact one; the cost layer only rounds up, and exact billing is the closest setting.

- **Each stage is billed as its own deployment**, like the instance choice: a multi-stage run with per-hour billing is rounded once per stage.
- Rounding matters when instances run briefly: short stages, or instances the router adds late. A stage of hours on per-second billing rounds by at most a second per instance.

### Adding a policy

1. Write a function `my_policy(usage, profile, cost_config)` in `opal/cost/cost_model.py`. `usage` is the dict from `stage_usage()`, `profile` the loaded price profile, `cost_config` the `cost` section. Return at least `{"breakdown_usd": {line: cost}}`; any other keys are copied into the report as-is. The console summary and the plots also read `pricing`, `billing` and `gpu_hours`, as returned by `gpu_hour_rental`.
2. Register it in `POLICIES` and select it with `cost.policy`.
3. Cite the paper it reproduces in the docstring, add a test with hand-checkable numbers in `tests/test_cost_model.py`, and describe it in this page.

If a policy needs a quantity that `stage_usage()` doesn't return yet, add it there from the stage statistics.

## Per-session cost

For an agent trace (the OTel trace replay, where every request carries a `session_id`), each stage's bill is also split between sessions, and each session's time is split into time at the LLM and gaps (see [Terms](#terms)):

```
session_cost    = gpu_busy * weighted_tokens / total_weighted_tokens
                + (gpu_idle + gpu_unused + billing_rounding) * open_s / total_open_s

weighted_tokens = token_weights.output * output_tokens
                + token_weights.recomputed_input * (input_tokens - cached_tokens)
                + token_weights.cached_input * cached_tokens

open_s      = last_completion - first_arrival                    # of the session's requests, at the router
at_llm_s    = length of the union of its (arrival, completion) spans
gap_s       = open_s - at_llm_s
gap_pct     = 100 * sum(gap_s) / sum(open_s)                    # over the stage's sessions
```

Token counts are summed over the session's requests, and `total_*` over all sessions of the stage. `cached_tokens` is the input served from the KV cache (`raw_kvc_hit_tokens`); the rest of the input is recomputed. The per-session costs add up to the stage total. It needs `raw_session_ids` and `arrival_completion_times` from the stage statistics ([[Statistics collection and processing]]).

`sessions` in the report holds the number of `sessions`, `median_usd` / `p90_usd` / `max_usd` cost per session, `mean_open_s` / `mean_at_llm_s` / `mean_gap_s`, `gap_pct` (`= 100 * mean_gap_s / mean_open_s`), and `per_session` (session id -> `cost_usd`, `open_s`, `at_llm_s`, `gap_s`). The `token_weights` used are at the top of the report.

**Choosing `token_weights`.** Only the ratios matter, since the split shares a known bill. Take them from published numbers of real serving systems rather than from the simulator, so they can be checked:

- **recomputed input:** large-scale DeepSeek serving measures an uncached input token at about 0.43–0.46 of an output token: SGLang on 96 H100 serves 52.3k input and 22.3k output tokens/s per node ([LMSYS, 2025](https://lmsys.org/blog/2025-05-05-large-scale-ep/)), and DeepSeek's own fleet 73.7k and 14.8k with 56% of the input cached ([DeepSeek, 2025](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)). List prices put it at 0.2–0.33 ([Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing), October 2026);
- **cached input:** no measured figure is published; list prices charge 0.005–0.02 of an output token.

The default is the middle of those ranges:

```json
"token_weights": {"output": 1.0, "recomputed_input": 0.3, "cached_input": 0.01}
```

The weights are not derived from Opal's own timing model, so the split follows published serving costs and can differ from the step times Opal simulates. To charge output tokens only, set the other two weights to 0.

Assumptions:

- The `cached_input` weight covers serving cached tokens only; keeping the KV between turns is a storage cost with no line yet.
- Everything but `gpu_busy` is split by time open. A trace replay has a fixed list of sessions, so it ends with a tail where the last sessions run alone and the GPUs mostly wait; bigger setups wait longer in it. When comparing setups, keep in mind (or leave out) that tail.
- The split uses the `gpu_hour_rental` breakdown lines.
- A session that spans two stages is split per stage.

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

Prices are per hour for the whole instance, in US dollars (the report's `_usd` keys; a profile in another `currency` is rejected). Add a new dated file rather than editing an old one, so earlier results stay reproducible.

No profile ships with the repository. To try the cost layer, save the example above as a file (e.g. `cost-profiles/cloud/aws-example.json`) and point `cost.profile` to it: `configs/defaults.json` then runs on one `p5.48xlarge`, 1 of 8 GPUs used.

## Planned

Following [#58](https://github.com/IBM/opal-sim/issues/58):

- **Storage options**: the instances priced as now, plus what keeping the KV tier would add on a storage service, e.g. a block volume of the tier's size, or a managed cache priced by capacity and data transferred.
- A provisioning diagnosis (over/under-provisioned, bottleneck), on-prem TCO profiles, and pricing tool execution on CPU servers separately from the LLM on GPU servers.
- Later: a scheduler that uses the cost model to pick the cheapest schedule.

## See also

- [[Configuration Simulation]] — the `cost` keys next to every other config section.
- [[Statistics collection and processing]] — the stage statistics the cost layer reads: `per_unit_workers`, `per_unit_gpu_busy_time`, `raw_session_ids`, `arrival_completion_times`.
- [[Router]] — `_per_second_stats`, which samples the worker count and the GPU-busy seconds once per simulated second.
- [[OTel Trace Replay]] — the workload that sets `session_id`, which the per-session split needs.
- [[Testing]] — what `tests/test_cost_model.py` covers.
