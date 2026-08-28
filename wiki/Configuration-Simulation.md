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

This section is only about *loading* the model's `config.json`. How that config gets turned into a per-request latency estimate — including the architecture-aware roofline model family (MoE, MLA, sparse attention, Mamba hybrids) — is controlled by `worker.inference_params` and `worker.hw` below; see [[LLM Inference Roofline Migration Plan]] and [[LLM Model Roofline Analysis]] for the underlying model.

---

## router

See [[Router]] for the routing-policy explanations (including why `MaxPrefix` is the default and how KVBM prefix-aware routing works), the auto-scaling mechanism, and event-batching internals. Nested under `router.router_params`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `policy` | string | `"MaxPrefix"` | Routing policy. Supported: `RoundRobin`, `LeastLoaded`, `Random`, `MaxPrefix`, `Balanced`. |
| `enable_scaling` | bool | `false` | Enable dynamic worker scale-up when queues exceed threshold. |
| `max_queue_threshold` | int | `4` | When any worker's queue reaches this size, trigger a scale-up event. |
| `scale_latency` | float | `40` | Virtual seconds it takes to start a new worker after a scale-up is triggered. |
| `max_workers` | int | `50` | Maximum number of workers to scale up to. |
| `periodic_infra_update_collection_time` | float | `30` | Interval (virtual seconds) at which the router collects infrastructure status from workers. |
| `max_event_batch_size` | int | `64` | Maximum number of requests the router dispatches per scheduling cycle. |

---

## workload

Workloads are defined as a list of stages under `workload.stages`. Each stage has a `type` and a `workload_params` dict. Stages run sequentially. See [[Running Workloads]] for the orchestrator/stage architecture that drives this list.

### Workload types

| Type | Description |
|------|-------------|
| `UniformReqRate` | Generates requests at a uniform rate. |
| `ExponentialReqRate` | Poisson arrival process with configurable jitter. |
| `trace` | Replays requests from a flat JSONL trace file (`{"timestamp", "input_length", "output_length", "hash_ids"}` rows). See [[Running Workloads]]. |
| `otel` | Replays real captured agentic traffic from OpenTelemetry `gen_ai` traces (session/turn-structured, not flat rows). See [[OTel Trace Replay]]. |
| `RAGWorkload` | Synthetic retrieval-augmented-generation workload for exercising prefix-cache reuse across a document corpus. See [[RAG Workload]]. |
| `SC25Workload` | See `opal/workloads/sc25_blog.py`. |

Every workload type also accepts a generic `time_duration_sec` in its `workload_params` (default `-1`, no per-stage timeout) — a stage-local wall-clock-of-virtual-time cap, independent of `total_requests`.

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
| `max_outstanding_requests` | int | `32` | Max concurrent in-flight requests before the workload pauses generation. |

### Trace workload_params

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_requests` | int | `10` | Number of trace entries to replay. `-1` = replay all. |
| `chunk_size` | int | `512` | Chunk size used for prefix hashing in trace replay (code fallback; `configs/defaults.json` ships `1`). |
| `multiplier_to_sec` | float | `1` | Multiplier to convert trace timestamps to virtual seconds (code fallback; `configs/defaults.json` ships `0.001`, i.e. millisecond timestamps). |
| `trace_file` | string | — | Path to the JSONL trace file. |

For the `otel` workload type's own parameters (`tokenizer`, `pretokenized`, `inter_turn_multiplier`, `max_concurrent_sessions`, etc. — a different set from the flat `trace` type above, despite the similarly-named `multiplier_to_sec`), see [[OTel Trace Replay]]. For `RAGWorkload`'s parameters (`num_documents`, `document_size`, `docs_per_request`, etc.), see [[RAG Workload]].

---

## worker

See [[vLLM Worker]] for the scheduler this section configures (interrupt-driven batching, preemption, KV-cache rate-limiting) and [[LLM Inference Roofline Migration Plan]] for how `worker.hw`/`worker.inference_params` feed into request-timing.

### worker.worker_params

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `worker_local_queue_capacity` | int | `1` | Documented as each worker's local queue capacity, but not currently read anywhere in `opal/` — dead config key as of this writing. |
| `periodic_infra_update_time` | float | `30` | Interval (virtual seconds) at which the worker reports its status to the router. |
| `kvcevent_coalesce_time` | float | `30` | Time window for coalescing KV cache events before processing. |

### worker.hw

Hardware specification for each worker.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpu` | string | `"H100"` | GPU name (informational). |
| `memory_gb` | float | `80` | GPU memory in GB. Used to compute KV cache capacity and scheduling limits in the legacy engine; not read by the `llm_inference` engine (see below). |
| `tflops` | float | `989.5` | GPU peak TFLOPS. Read by both engines; under the `llm_inference` engine this is treated as the fp16 figure specifically (see `tflops_fp16` below). |
| `mem_bw_TBps` | float | `3.3` | GPU memory bandwidth in TB/s. Used by both engines' roofline math. |
| `tp` | int | `1` | Tensor parallelism degree. Scales effective compute and aggregate bandwidth in both engines. |
| `tflops_fp16` | float | = `tflops` | *(`llm_inference` engine only.)* Explicit fp16 peak FLOPS, overriding the bare `tflops` alias. |
| `tflops_fp32` | float | `tflops_fp16 / 2` | *(`llm_inference` engine only.)* |
| `tflops_fp8` | float | `tflops_fp16 * 2` | *(`llm_inference` engine only.)* |
| `tflops_fp4` | float | `tflops_fp16 * 2` | *(`llm_inference` engine only.)* Not compounded further from fp8 — real fp4:fp8 ratios vary too much by GPU generation to guess. |
| `num_gpus` | int | = `tp` | *(`llm_inference` engine only.)* |

### worker.vllm_params

Parameters modeling the vLLM-style continuous batching scheduler — see [[vLLM Worker]] for how each is used inside `_build_batch`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_num_seqs` | int | `256` | Maximum number of sequences in a running batch. |
| `max_num_batched_tokens` | int | `8192` | Maximum tokens processed in a single scheduler step. (The scheduler dataclass's own fallback if this key is omitted entirely is `2048` — but `configs/defaults.json` always sets `8192`.) |
| `max_kvc_ready_requests` | int | `8` | Maximum number of requests with KV cache ready waiting to enter the GPU batch. (Dataclass fallback if omitted: `4`; `configs/defaults.json` sets `8`.) |
| `chunked_prefill` | bool | `true` | Enable chunked prefill (split long prefills across multiple steps). |
| `block_size` | int | `16` | KV cache block size in tokens (paged attention). |
| `lookahead_reqs` | int | `256` | Max waiting requests scanned per batch rebuild once the batch started non-empty. |

### worker.inference_params

Controls the analytical model used to predict prefill (TTFT) and decode latency. Two engine families exist — see [[LLM Inference Roofline Migration Plan]] for the full design and [[vLLM Worker]]'s Timing Calculation section for how a batch's time is actually computed:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `"roofline"` | Selects the inference-timing engine. `"roofline"` or `"synthetic"` (or anything else) select the legacy `GPUModel` — `"synthetic"` uses `mean_latency_secs`, anything else uses the `a`/`b` regression. **`"llm_inference"`** selects the newer per-architecture roofline model family (MoE/MLA/sparse-attention/Mamba-hybrid aware) built via `ModelFactory` from the model's actual `config.json` — see [[LLM Model Roofline Analysis]]. |
| `mean_latency_secs` | float | `1.0` | Mean latency per step when using the legacy `synthetic` model. |
| `a` | string | `"4"` | Regression constant for the legacy engine's older moonshot-model math. Also usable as an override input (`a_override`) to the `llm_inference` engine, in which case it additionally shifts that engine's default compute/memory efficiency to `1.0` (see below) on the assumption it's carried over from a config already calibrated for the legacy engine. |
| `b` | string | `"24"` | Regression constant, same dual role as `a` above (`b_override` for the `llm_inference` engine). |
| `compute_efficiency` | float | `1.0` if `a`/`b` set, else `0.4` | *(`llm_inference` engine only.)* Fraction of `worker.hw`'s peak FLOPS assumed achievable. |
| `memory_efficiency` | float | `1.0` if `a`/`b` set, else `0.6` | *(`llm_inference` engine only.)* Fraction of `worker.hw`'s peak bandwidth assumed achievable. |

---

## kvc

KV cache management configuration for tiered storage. See [[vLLM Worker]]'s KV Cache Integration section for how lookups/fetches interact with the scheduler, and [[RAG Workload]] for a worked example that exercises tiered lookup across `CPUMemory`/`DistributedFS`. For checking which of the keys below (or under `worker`/`router`/`workload`) your own config actually exercises, see [[Config Usage Tracking]].

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kvc_tiers` | list[string] | `["CPUMemory"]` | Which storage tiers are enabled. Supported: `CPUMemory`, `LocalNVMe`, `DistributedFS`. |
| `chunk_size` | int | `256` | KV cache chunk size in tokens. Determines granularity of cache storage and lookup. |
| `save_unfull_chunk` | bool | `true` | Documented as whether to save partial (not full) chunks to the cache, but currently hardcoded to `True` unconditionally in `OpalTokenDatabase.__init__` — this config key has no effect as of this writing. |

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
