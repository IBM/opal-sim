# Router

The `Router` class (`opal/router.py`) is the central request dispatcher in OpalSim. It accepts incoming LLM requests, selects a target worker based on a configurable routing policy, and collects completions. For the full config reference (all `router.router_params` defaults alongside every other simulation section), see [[Configuration Simulation]].

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
- **`_per_second_stats`** — samples per-second throughput and GPU utilization.
- **`process_events`** — batches and processes infrastructure events (KVC updates, system events) used by prefix-aware policies.
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
    "periodic_infra_update_collection_time": 30,
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
| `periodic_infra_update_collection_time` | Interval (sim seconds) between event processing cycles when the event queue is empty. |
| `max_event_batch_size` | Max events processed per batch in `process_events`. |

## Event Processing

Workers emit `KVCEvent` and `SystemEvent` messages to the router's event queue. These are batched (up to `max_event_batch_size`) and forwarded to the KVBM, which maintains the prefix-cache state used by the `MaxPrefix` policy. When the event queue is empty, the router sleeps for `periodic_infra_update_collection_time` before checking again.
