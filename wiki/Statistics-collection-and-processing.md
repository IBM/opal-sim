# Statistics collection and processing

Opal collects statistics at four layers that feed into each other, from the finest to the coarsest granularity:

1. **Per-request timestamps** — every `LLMRequest` carries an `LLMRequestStats` object (`opal/core/request.py`) that records when the request hit each stage of its lifecycle (creation, router arrival, worker arrival, KVC lookup, GPU start/end, completion).
2. **Per-stage aggregation** — a `StageStatistics` object (`opal/stats/stage_statistics.py`) per workload stage collects one entry per finished request (latency, TTFT, queueing time, KVC hit tokens, tier breakdown, etc.) plus running counters (finished/queued/failed requests).
3. **Per-second sampling** — the `Router`'s `_per_second_stats` process (`opal/router/router.py`) samples system-wide throughput, active worker count, and average GPU utilization once per simulated second, appending to the same `StageStatistics` object.
4. **Final output** — at the end of each stage (and at the end of the whole run), `StageStatistics` is rendered to a human-readable summary (printed to stdout and appended to `simulation.log`), optionally plotted to PDF, and always serialized to `opal_stats.json`.

This page walks through each layer using the actual field and method names in the code. For the list of output files this produces, see the README's ["What are simulation outputs"](https://github.com/IBM/opal-sim#what-are-simulation-outputs) section — it is not repeated here.

## Layer 1: per-request stats (`LLMRequestStats`)

Defined in `opal/core/request.py`. Every `LLMRequest` gets one `LLMRequestStats` instance (`request.stats`) at creation time. It stores a numbered sequence of raw timestamps:

```
_1_creation_time            # request created at the workload generator
_2a_router_arrival_time     # arrival at the router
_2b_worker_time             # arrival at the vLLM worker
_3_start_processing_time    # scheduler looks at the request (KVC lookup starts)
_4_request_ready_time       # request is READY (KVC fetched, or no prefix match)
_5_gpu_start_time           # added to a GPU batch
_6_prefill_done_time        # prefill finished (note: mark_prefill_done() actually
                             # writes this into `_5_prefill_done_time`, a
                             # differently-named attribute set outside __init__ —
                             # see below)
_7_decode_done_time         # decode finished / request fully done
_8_done_at_router           # completion observed back at the router
```

plus:
- `__scheduler_timestamps` — the raw list of every scheduler-step timestamp this request was touched at (appended via `add_scheduler_timestamp()`). The first `__num_prefill_sched_steps` entries are prefill steps; the rest are decode steps.
- `__kvc_hit_tokens` — number of prefix tokens served from the KV cache (`set_prefix_hit_tokens()` / `get_prefix_hit_tokens()`).
- `__kvc_tier_tokens` — a `dict[str, int]` mapping cache tier name (e.g. `"CPUMemory"`, `"LocalNVMe"`, `"DistributedFS"`, or `"Cache Miss"`) to token count for this request (`set_kvc_tier_tokens()` / `get_kvc_tier_tokens()`).

Derived getters compute the metrics that matter downstream:

| Method | Meaning |
|---|---|
| `get_queue_time()` | `_5_gpu_start_time - _1_creation_time` — time spent anywhere before reaching the GPU. |
| `get_ttft()` | `_5_prefill_done_time - _1_creation_time` — time to first token. Set by `mark_prefill_done()`, which snapshots the last prefill entry out of `__scheduler_timestamps`. |
| `get_kvc_fetch_time()` | `_4_request_ready_time - _3_start_processing_time` — KV cache fetch/lookup latency. |
| `get_decode_times_including_ttft()` | `[TTFT, itl_1, itl_2, ...]` — TTFT as element 0, followed by the inter-token latency (delta between consecutive scheduler steps) for every decode step. |
| `get_total_worker_time()` | `_7_decode_done_time - _2b_worker_time` — end-to-end time inside the worker. |
| `get_gpu_time()` | last scheduler timestamp minus first — total time the request occupied the GPU scheduler (includes scheduling overhead). |

The worker that actually populates most of these fields during the scheduling loop is documented in [[vLLM Worker]] (see its "Statistics Collected" section for the mapping from scheduler phases to these timestamps — that page's field names such as `worker_arrival_time`, `_3_start_processing_time`, `_4_request_ready_time`, `_5_gpu_start_time`, `_7_decode_done_time`, `scheduler_timestamps`, and `prefix_hit_tokens` refer to this same `LLMRequestStats` object).

`IORequest` (also in `request.py`) has its own lightweight `_Stats` inner class (`arrival_time`, `submission_time`, `completion_time`, `start_kvc_time`, `io_time`) used for KV-cache I/O timing, separate from `LLMRequestStats`.

## Layer 2: per-stage aggregation (`StageStatistics`)

Defined in `opal/stats/stage_statistics.py`. The `WorkloadOrchestrator` (`opal/workloads/workload_orchestrator.py`) creates one `StageStatistics` instance per workload stage in `self.stage_stats`, and exposes the currently-running stage's instance via `get_active_stage_stats()`. Stage boundaries are marked with `stage_time_start` / `stage_time_end`, set when the orchestrator's `run()` loop starts/finishes each stage.

### How requests get recorded

The `Router` (`opal/router/router.py`) drives most of the bookkeeping:
- `_accept_requests()` increments `get_active_stage_stats().queued_requests` as soon as a request arrives at the router — this is the "total requests seen" counter.
- `_collect_completion()` calls `get_active_stage_stats().add_finished_request(request)` once a request comes back on the results queue.

`add_finished_request(request)` reads `request.stats` (the `LLMRequestStats` from Layer 1) and appends one entry to each of these raw-value lists:

| Field | Source |
|---|---|
| `raw_latency_values` | via `add_total_time_in_system({"E2E": stats.get_total_worker_time()})` — also bumps `total_latency`, `finished_requests`, and a bucketed `latencies` histogram (power-of-2 bins from 1s up to ~10,000,000s). |
| `raw_queuing_values` | `stats.get_queue_time()` |
| `raw_ttft_values` | `stats.get_ttft()` |
| `raw_kvc_io_time` | `stats.get_kvc_fetch_time()` |
| `raw_kvc_hit_tokens` | `stats.get_prefix_hit_tokens()` |
| `raw_gpu_time` | `stats.get_gpu_time()` |
| `raw_decode_values` | `stats.get_decode_times_including_ttft()` (list of lists, one per request) |
| `input_output_tokens_sz` | `(request.input_length, request.output_length)` tuples |
| `kvc_tier_tokens` | accumulated (`defaultdict(int)`) from `stats.get_kvc_tier_tokens()`, tier name -> running token total across all requests in the stage |

`failed_requests` exists as a counter field but is not incremented anywhere in the router/worker code as of this writing; `print_summary_results()` falls back to `max(failed_requests, queued_requests - finished_requests)` so a non-zero "Failed requests" figure can still show up in the summary even though nothing sets the field directly.

### Per-second series (populated from Layer 3)

Three parallel lists are appended to once per simulated second (see next section): `per_unit_req_done`, `per_unit_workers`, `per_unit_gpu_utilization`, via `add_per_unit_workdone()`, `add_per_unit_workers()`, and `add_per_unit_gpu_utilization()` respectively.

### Computing summary metrics

- `calculate_user_metrics_in_ms()` turns `raw_ttft_values` and `raw_decode_values` into `(mean, median, p99)` triplets for TTFT, TPOT, and ITL, converted to milliseconds. Internally `_calculate_itl_tpot()` splits each request's `[TTFT, itl_1, itl_2, ...]` array into a per-request TPOT (mean of that request's inter-token gaps) and a flat pool of every individual ITL (token-weighted) across all requests.
- `get_average_latency()` = `total_latency / finished_requests`.
- `get_histogram()` returns the bucketed `latencies` array as a percentage of `finished_requests`.
- `get_histogram_breakdown(num_bins)` / `_calculate_itl_tpot` support a richer per-source latency breakdown, used by `plot.plot_stacked_histogram` (this particular breakdown path expects `self.latencies` to be a list of per-request `dict` breakdowns rather than the numpy histogram array that `__init__` actually creates it as — it is not exercised by the default `add_finished_request` code path).
- `print_summary_results(log_file=None)` renders the vLLM-style benchmark block you see at the end of a run (`Successful requests`, `Failed requests`, `Benchmark duration`, throughput, TTFT/TPOT/ITL triplets, and — when any KV-cache tier tokens were recorded — a `KVC Tier Token Breakdown` section with per-tier token counts and percentages). If `log_file` is given, the same text is written there too.

### Persistence: `to_dict()` / `from_dict()` / JSON round-trip

`to_dict()` serializes every field above (numpy arrays converted via `.tolist()`) into a plain dict; `from_dict()` reconstructs a `StageStatistics` from that dict (calling `__init__` first so `bins`/`latencies` get their default shape, then overwriting). `raw_kvc_hit_tokens` is read with `.get(..., [])` so old JSON files without that field still load. `write_to_json(path)` writes the dict out using a custom compact formatter (`dump_json_compact_lists` — one key per line, but lists/arrays kept on a single line for readability) rather than plain `json.dump`.

## Layer 3: per-second sampling (Router)

The `Router` runs a dedicated SimPy process, `_per_second_stats()` (see [[Router]] for the full list of router processes), that wakes up once every simulated second for as long as the simulation is running and pushes three samples into the *currently active* stage's `StageStatistics`:

- `stats.add_per_unit_workdone(self.cumulative_finished - last)` — requests completed since the previous tick (this feeds `thrp-request-sec.pdf`).
- `stats.add_per_unit_workers(len(self._active_workers))` — how many workers are active right now (feeds `thrp-workers-sec.pdf`; flat if auto-scaling is disabled).
- `stats.add_per_unit_gpu_utilization(utilization)`, where `utilization = mean(w.gpu_busy_time for w in active_workers) * 100 / now` — average, since simulation start, of the fraction of time each active worker's GPU has been busy (feeds `gpu-utilization-per-sec.pdf`).

`sample_workdone_per_K(k)` (on `StageStatistics`, delegating to `opal.utils.util.sample_series_K`) can downsample any of these per-second series by averaging every `k` consecutive samples — used by the plotting code's `per_K` parameter to reduce noise on long runs.

## Layer 4: output artifacts

At the end of a run, `OpalSimulator._process_per_stage()` (`opal/core/opal.py`) iterates `workload_orchestrator.stage_stats` and, per stage:

1. Prints a `===== stage_N =====` header and calls `StageStatistics.print_summary_results(log_file=...)`, writing the vLLM-style benchmark block both to stdout and to `simulation.log` (this is the mechanism behind "Write stage summary statistics to simulation.log").
2. If `-g`/`--graphs` was passed, calls `simend_plot(stats, config, stage_dir)` (`opal/stats/plot.py`), which produces the CDF, throughput, worker-count, GPU-utilization, and histogram PDFs for that stage.
3. Separately, `OpalSimulatorEnvironment.write_simulation_data()` (`opal/core/environment.py`) — invoked from `OpalSimulator.run()` when `simulation.save_simulation_data` is `true` (see [[Configuration Simulation]]) — writes each stage's full `StageStatistics.to_dict()` to `stage_N/opal_stats.json`.

The exact set of per-stage files (`cdf-latencies.pdf`, `gpu-utilization-per-sec.pdf`, `histo-latencies.pdf`, `thrp-request-sec.pdf`, `thrp-workers-sec.pdf`, `opal_stats.json`) and top-level files (`sim_config.json`, `simulation.log`) is documented in the README's "What are simulation outputs" and "Logging" sections — refer there for the canonical list; this page only explains how each file's content is produced.

## Consuming `opal_stats.json` programmatically

Because `StageStatistics.to_dict()` / `from_dict()` are symmetric, the simplest way to reload a stage's stats outside of a live simulation is:

```python
import json
from opal.stats.stage_statistics import StageStatistics

with open("simulation-runs/sim-.../stage_0/opal_stats.json") as f:
    data = json.load(f)
stats = StageStatistics.from_dict(data)

print(stats.get_average_latency())
print(stats.calculate_user_metrics_in_ms())   # (ttft_mean, ttft_median, ttft_p99, tpot_mean, ...)
```

`tests/test_stage_statistics.py` exercises this same machinery directly (without going through a full simulation run) and is a good second reference for exact field behavior — e.g. it confirms that `raw_kvc_hit_tokens` defaults to `[]` when reloading an older JSON file that predates that field, and that `print_summary_results()` works whether or not a `log_file` is supplied.

## See also

- [[vLLM Worker]] — where most `LLMRequestStats` timestamps are actually set during scheduling.
- [[Router]] — full list of router-owned SimPy processes, including `_per_second_stats`.
- [[Configuration Simulation]] — the `simulation.save_simulation_data` key that gates whether `opal_stats.json` gets written at all.
