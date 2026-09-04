## Running Workloads

Opal uses a stage-based workload orchestration system. A simulation can define one or more workload stages that execute sequentially, each with its own type, parameters, and termination conditions. For the full config reference (defaults as shipped in `configs/defaults.json`, plus every other simulation section), see [[Configuration Simulation]].

### Architecture

The workload system has three layers:

1. **WorkloadOrchestrator** (`opal/workload_orchestrator.py`) — Manages the lifecycle of all stages. It dynamically loads workload classes from the `opal/workloads/` directory by matching the `type` field in the configuration to a class name (case-insensitive). Stages run sequentially; the orchestrator waits for each stage to complete before starting the next.

2. **AbstractWorkload** (`opal/workloads/abstract_workload.py`) — Base class that all workload types must extend. It handles:
   - Scheduling a per-stage timeout (via `time_duration_sec`)
   - Running request generation and response processing as concurrent SimPy processes
   - Tracking generated/received request counts

3. **Concrete workload classes** — Implement `generate_requests()` to produce `LLMRequest` objects and feed them to the router.

### Available Workload Types

| Type (config string) | Class | Description |
|---|---|---|
| `UniformReqRate` | `UniformReqRate` | Generates requests at a fixed rate (requests/sec). Prompt and output sizes are drawn uniformly at random from configurable min/max ranges. |
| `ExponentialReqRate` | `ExponentialReqRate` | Same as `UniformReqRate` but inter-arrival times follow an exponential distribution with configurable jitter. |
| `Trace` | `Trace` | Replays a JSONL trace file. Each line contains `timestamp`, `input_length`, `output_length`, and `hash_ids`. Requests are dispatched at the recorded timestamps. |
| `otel` | `Otel` | Replays real captured agentic traffic from OpenTelemetry `gen_ai` traces (session/turn-structured, not flat rows like `Trace`). See [[OTel Trace Replay]] for its own parameter set. |
| `RAGWorkload` | `RAGWorkload` | Synthetic retrieval-augmented-generation workload for exercising prefix-cache reuse across a document corpus. See [[RAG Workload]]. |
| `SC25Workload` | `SC25Workload` | A specialized benchmark workload that runs a cold pass (unique prompts) followed by a warm pass (repeated prompts for 100% cache hit), used for KV-cache performance characterization. |

### Configuration

Workloads are configured under the `"workload"` key in the simulation JSON. The `"stages"` array defines the ordered list of stages:

```json
"workload": {
  "stages": [
    {
      "type": "UniformReqRate",
      "workload_params": {
        "request_rate": 2.0,
        "total_requests": 100,
        "prompt_size_min": 32,
        "prompt_size_max": 16384,
        "default_prefix_length": 1024,
        "jitter": 0.0,
        "output_tokens_min": 32,
        "output_tokens_max": 128
      }
    },
    {
      "type": "trace",
      "workload_params": {
        "total_requests": 10,
        "chunk_size": 1,
        "multiplier_to_sec": 0.001,
        "trace_file": "traces/hello.jsonl"
      }
    }
  ]
}
```

### Workload Parameters

#### UniformReqRate / ExponentialReqRate

| Parameter | Description |
|---|---|
| `request_rate` | Target requests per second |
| `total_requests` | Stop after this many requests (set to `-1` or `0` for unlimited) |
| `prompt_size_min` / `prompt_size_max` | Range for randomly generated prompt token count |
| `default_prefix_length` | Number of prefix tokens (used for KV-cache matching) |
| `output_tokens_min` / `output_tokens_max` | Range for generated output token count |
| `time_duration_sec` | (Optional) Stage timeout in seconds |
| `max_outstanding_requests` | (Optional) Max concurrent in-flight requests (default: 32) |
| `jitter` | (ExponentialReqRate only) Controls variance in inter-arrival times |

#### Trace

| Parameter | Description |
|---|---|
| `trace_file` | Path to the JSONL trace file |
| `total_requests` | Replay only this many entries (`-1` for all) |
| `chunk_size` | Token chunk granularity for hash expansion (default: 512) |
| `multiplier_to_sec` | Multiply raw timestamps by this factor to convert to seconds (default: 1) |

### Termination Conditions

A workload stage stops generating requests when **any** of the following becomes true:

1. **Global timeout** — The simulation-wide `simulation_time` has elapsed.
2. **Local timeout** — The stage's `time_duration_sec` has elapsed.
3. **Request count** — The configured `total_requests` have been generated.

After request generation stops, the stage waits for all outstanding responses to return before the orchestrator advances to the next stage.

### Adding a New Workload Type

1. Create a new Python file in `opal/workloads/`.
2. Define a class that extends `AbstractWorkload`.
3. Implement the `generate_requests()` generator method — it should `yield` requests into `self.req_router.input_queue` with appropriate inter-arrival delays.
4. Set `self.is_finished = True` when generation completes.
5. Use the class name (case-insensitive) as the `"type"` value in the configuration.

No registration step is needed — the orchestrator discovers classes automatically by scanning the `opal/workloads/` directory.
