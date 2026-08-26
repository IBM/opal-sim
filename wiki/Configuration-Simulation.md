# Configuring Opal 

Opal is configured via a JSON file. All parameters have defaults defined in `configs/defaults.json`. You can override any subset of parameters by passing your own config file — only the keys you specify are overridden.

---

## simulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation_time` | float | `-1.0` | Run the simulation for the given virtual seconds. If `-1`, run until the workload finishes (all requests generated or all trace events replayed). |
| `max_wall_time_sec` | float | `-1.0` | Cap the run after this many real (wall-clock) seconds. If `-1`, no wall-clock cap. When the cap is hit the run stops gracefully (partial stats are still written) and a warning is logged. Can be overridden from the CLI with `--max-wall-time`. |
| `seed` | int | `42` | Python random seed for reproducibility. |
| `num_workers` | int | `1` | Initial number of LLM workers at simulation start. |
| `save_simulation_data` | bool | `true` | Save per-request statistics and simulation results to the output directory. |
| `show_progress` | bool | `false` | Show a tqdm progress bar during the simulation. |

---

## model

Nested under `model.model_params`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | `"granite-3.3-8b-instruct"` | Model name. Used to resolve the config from a local directory or HuggingFace. |
| `config_dir` | string | `"./model-configs/"` | Local directory containing the model's `config.json`. If present, the model is loaded from `<config_dir>/<name>/config.json`. |
| `hf_url` | string | — | HuggingFace model identifier (e.g., `"ibm-granite/granite-3.3-8b-instruct"`). If set, the config is fetched from HuggingFace. Mutually exclusive with `config_dir`. |

---

## router

Nested under `router.router_params`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy` | string | `"MaxPrefix"` | Routing policy. Supported: `RoundRobin`, `LeastLoaded`, `Random`, `MaxPrefix`, `Balanced`. |
| `enable_scaling` | bool | `false` | Enable dynamic worker scale-up when queues exceed threshold. |
| `max_queue_threshold` | int | `4` | When any worker's queue reaches this size, trigger a scale-up event. |
| `scale_latency` | float | `40` | Virtual seconds it takes to start a new worker after a scale-up is triggered. |
| `max_workers` | int | `50` | Maximum number of workers to scale up to. |
| `latency_percentile` | string | `"p95"` | Percentile used for snapshot TTFT/ITL SLOs. Supported: `p90`, `p95`, `p99`. |
| `latency_window` | int | `50` | Number of most recent completed requests over which TTFT/ITL percentiles are computed. Snapshots store `-1` until this many requests have finished. |
| `max_event_batch_size` | int | `64` | Maximum number of infrastructure events processed per drain in `process_events`. |

---

## workload

Workloads are defined as a list of stages under `workload.stages`. Each stage has a `type` and a `workload_params` dict. Stages run sequentially.

### Workload types

| Type | Description |
|------|-------------|
| `UniformReqRate` | Generates requests at a uniform rate. |
| `ExponentialReqRate` | Poisson arrival process with configurable jitter. |
| `trace` | Replays requests from a JSONL trace file. |

### Common workload_params (UniformReqRate, ExponentialReqRate)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_rate` | float | `2.0` | Mean requests generated per virtual second. |
| `total_requests` | int | `100` | Total number of requests to generate. `-1` means run for `simulation_time` or until trace exhausted. |
| `prompt_size_min` | int | `32` | Minimum prompt size (tokens). Uniformly sampled from `[min, max]`. |
| `prompt_size_max` | int | `16384` | Maximum prompt size (tokens). |
| `output_tokens_min` | int | `32` | Minimum output/decode tokens per request. |
| `output_tokens_max` | int | `128` | Maximum output/decode tokens per request. |
| `default_prefix_length` | int | `1024` | Length of shared prefix sampled from previous requests (for KV cache hit simulation). |
| `jitter` | float | `0.0` | (ExponentialReqRate only) Controls deviation from mean inter-arrival time. `0` = nearly uniform, `1.0` = maximum variance. |
| `max_outstanding_requests` | int | — | Optional. If set, limits how many requests can be in-flight before the workload pauses generation. |

### Trace workload_params

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_requests` | int | `10` | Number of trace entries to replay. `-1` = replay all. |
| `chunk_size` | int | `1` | Chunk size used for prefix hashing in trace replay. |
| `multiplier_to_sec` | float | `0.001` | Multiplier to convert trace timestamps to virtual seconds (e.g., `0.001` for milliseconds). |
| `trace_file` | string | — | Path to the JSONL trace file. |

---

## worker

### worker.worker_params

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `worker_local_queue_capacity` | int | `1` | Each worker's local queue capacity where the router places incoming requests. |
| `periodic_infra_update_time` | float | `5` | Interval (virtual seconds) at which the worker pushes status (`SystemEvent`) to the router. This is the sole bound on how stale worker metrics in a `MetricsSnapshot` can be. |
| `kvcevent_coalesce_time` | float | `30` | Time window for coalescing KV cache events before processing. |

### worker.hw

Hardware specification for each worker.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu` | string | `"H100"` | GPU name (informational). |
| `memory_gb` | float | `80` | GPU memory in GB. Used to compute KV cache capacity and scheduling limits. |
| `tflops` | float | `989.5` | GPU peak TFLOPS. Used by the roofline inference model. |
| `mem_bw_TBps` | float | `3.3` | GPU memory bandwidth in TB/s. Used by the roofline inference model. |
| `tp` | int | `1` | Tensor parallelism degree. |

### worker.vllm_params

Parameters modeling the vLLM-style continuous batching scheduler.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_seqs` | int | `256` | Maximum number of sequences in a running batch. |
| `max_num_batched_tokens` | int | `8192` | Maximum tokens processed in a single scheduler step. |
| `max_kvc_ready_requests` | int | `8` | Maximum number of requests with KV cache ready waiting to enter the GPU batch. |
| `chunked_prefill` | bool | `true` | Enable chunked prefill (split long prefills across multiple steps). |
| `block_size` | int | `16` | KV cache block size in tokens (paged attention). |

### worker.inference_params

Controls the analytical model used to predict prefill (TTFT) and decode latency.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"roofline"` | Inference latency model. Supported: `roofline` (uses hardware specs to compute latency analytically) or `synthetic` (fixed mean latency). |
| `mean_latency_secs` | float | `1.0` | Mean latency per step when using the `synthetic` model. |
| `a` | string | `"4"` | Regression constant (legacy, used in older moonshot model). |
| `b` | string | `"24"` | Regression constant (legacy, used in older moonshot model). |

---

## kvc

KV cache management configuration for tiered storage.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kvc_tiers` | list[string] | `["CPUMemory"]` | Which storage tiers are enabled. Supported: `CPUMemory`, `LocalNVMe`, `DistributedFS`. |
| `chunk_size` | int | `256` | KV cache chunk size in tokens. Determines granularity of cache storage and lookup. |
| `save_unfull_chunk` | bool | `true` | Whether to save partial (not full) chunks to the cache. |

### Tier configurations

Each tier is configured as a separate object keyed by its name:

#### CPUMemory (host DRAM)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bandwidth_GBps` | float | `50` | Read/write bandwidth in GB/s. |
| `latency_nsec` | float | `200` | Access latency in nanoseconds. |
| `concurrency` | int | `1000000` | Maximum concurrent I/O operations. |
| `capacity_GB` | float | `8` | Total capacity in GB. |

#### LocalNVMe

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bandwidth_GBps` | float | `10` | Read/write bandwidth in GB/s. |
| `latency_nsec` | float | `10000` | Access latency in nanoseconds. |
| `concurrency` | int | `1000` | Maximum concurrent I/O operations. |
| `capacity_GB` | float | `1024` | Total capacity in GB. |

#### DistributedFS

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bandwidth_GBps` | float | `100` | Aggregate bandwidth in GB/s (shared across all workers). |
| `latency_nsec` | float | `100000` | Network + access latency in nanoseconds. |
| `concurrency` | int | `1000` | Maximum concurrent I/O operations. |
| `capacity_GB` | float | `1048576` | Total capacity in GB (effectively unlimited). |
