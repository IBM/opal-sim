# Router

The `Router` class (`opal/router.py`) is the central request dispatcher in OpalSim. It accepts incoming LLM requests, selects a target worker based on a configurable routing policy, and collects completions.

## Architecture

```
Workload Generator
       |
       v
  [input_queue]
       |
       v
    Router  ──policy──>  Worker 0
       |                 Worker 1
       |                 Worker N
       v
  [results_queue]
       |
       v
  Stage Statistics
```

The router runs several concurrent SimPy processes:
- **`_accept_requests`** — pulls requests from the input queue, applies the routing policy, and dispatches to a worker.
- **`_collect_completion`** — gathers finished requests from the shared results queue and records statistics.
- **`_per_second_stats`** — samples per-second throughput, worker count, and GPU utilization for plots.
- **`process_events`** — drains infrastructure events as they arrive (KVC updates, worker `SystemEvent` telemetry). KVC events update prefix-aware routing; `SystemEvent`s produce a `MetricsSnapshot`.
- **`_per_second_scaling`** (optional) — auto-scales workers when queue depth exceeds a threshold.

## Routing Policies

Set via `router.router_params.policy` in the config:

| Policy | Behavior |
|--------|----------|
| `RoundRobin` | Cycles through workers sequentially. |
| `LeastLoaded` | Picks the worker with the fewest outstanding requests. |
| `Random` | Uniform random selection across active workers. |
| `MaxPrefix` | Scores workers by KV-cache prefix overlap (via KVBM) and routes to the best match. Falls back to random when no prefix data exists. |
| `Balanced` | (Not yet implemented.) |

### MaxPrefix Details

The `MaxPrefix` policy uses the KV Block Manager (KVBM) to score each worker based on how much of the incoming request's prefix is already cached. When multiple workers tie for the highest score, one is selected randomly. If no prefix data is available (cold start), it falls back to random routing.

## Auto-Scaling

When `enable_scaling` is `true`, the router checks worker queue depths every simulated second. If any worker's outstanding request count exceeds `max_queue_threshold`, a new worker is added after `scale_latency` seconds (simulating provisioning time), up to `max_workers`.

## Configuration

```json
"router": {
  "router_params": {
    "policy": "MaxPrefix",
    "enable_scaling": false,
    "max_queue_threshold": 4,
    "scale_latency": 40,
    "max_workers": 50,
    "latency_percentile": "p95",
    "latency_window": 50,
    "max_event_batch_size": 64
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `policy` | Routing algorithm (see table above). |
| `enable_scaling` | Enable dynamic worker auto-scaling. |
| `max_queue_threshold` | Queue depth that triggers a scale-up. |
| `scale_latency` | Simulated seconds to provision a new worker. |
| `max_workers` | Maximum number of workers allowed. |
| `latency_percentile` | Percentile for snapshot TTFT/ITL SLOs (`p90`, `p95`, `p99`). |
| `latency_window` | Number of most recent completed requests over which those percentiles are computed. |
| `max_event_batch_size` | Max events processed per batch in `process_events`. |

## Event Processing

Workers emit `KVCEvent` and `SystemEvent` messages to the router's event queue. `process_events` blocks until an event arrives, then drains up to `max_event_batch_size` events and forwards them to the KVBM.

`SystemEvent` telemetry is the last report from each worker (pushed every `worker.periodic_infra_update_time` seconds). When a drain includes at least one `SystemEvent`, the router records a `MetricsSnapshot`: reported queue depth and KV-cache utilization, plus TTFT/ITL at `latency_percentile` over the last `latency_window` completions. Snapshots are not taken on a wall-clock tick; KVC-only batches do not create snapshots.
