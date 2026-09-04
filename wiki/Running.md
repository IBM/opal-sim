# Running

Every Opal simulation is started the same way: put the repo root on `PYTHONPATH` and invoke `opal/main.py` with a config file. See [[Install Requirements]] for environment setup.

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python ./opal/main.py -c <path-to-config.json> -g -o simulation-runs
```

- `-c` — path to the simulation config JSON (defaults to `./configs/defaults.json` if omitted).
- `-g` — also render the final graphs (omit for a faster, plot-free run).
- `-o` — output directory for run artifacts (defaults to `simulation-runs/`).
- `--max-wall-time` — cap the run by real elapsed wall-clock seconds (separate from the simulated-time/request-count stopping conditions inside the config).

What varies between runs is the `"workload"` section of the config — specifically the `"stages"` array, where each stage picks a `"type"` and its `"workload_params"`. This page shows a minimal, runnable example for each workload type. For the full architecture and parameter reference (including termination-condition rules), see [[Running Workloads]]. For the complete config schema beyond `workload`, see [[Configuration Simulation]].

## UniformReqRate

Fixed-rate synthetic traffic with prompt/output sizes drawn uniformly at random. This is the first stage of the shipped `configs/defaults.json`:

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
    }
  ]
}
```

```bash
python ./opal/main.py -c configs/defaults.json
```

## ExponentialReqRate

Same request/prompt-size generation as `UniformReqRate`, but inter-arrival times are drawn from an exponential (Poisson-arrival) distribution, with `jitter` controlling the variance. No shipped config uses this type; take the `UniformReqRate` stage above and swap the `type`:

```json
"workload": {
  "stages": [
    {
      "type": "ExponentialReqRate",
      "workload_params": {
        "request_rate": 2.0,
        "total_requests": 100,
        "prompt_size_min": 32,
        "prompt_size_max": 16384,
        "default_prefix_length": 1024,
        "jitter": 0.3,
        "output_tokens_min": 32,
        "output_tokens_max": 128
      }
    }
  ]
}
```

Save this as e.g. `configs/exponential-example.json` and run:

```bash
python ./opal/main.py -c configs/exponential-example.json
```

## Trace

Replays a flat JSONL trace file (`timestamp`, `input_length`, `output_length`, `hash_ids` per line). This is the second stage of `configs/defaults.json`, replaying `traces/hello.jsonl`:

```json
"workload": {
  "stages": [
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

```bash
python ./opal/main.py -c configs/defaults.json
```

(Other shipped configs with a `trace` stage: `configs/lei.json`, `configs/hf.json`, `configs/pr26-test.json`, `configs/demo-mc-10-100-workers.json`.)

## otel

Replays real captured agentic OpenTelemetry `gen_ai` traces (session/turn-structured — not flat rows like `Trace`). Minimal example, adapted from the shipped `configs/defaults_otel.json`:

```json
"workload": {
  "stages": [
    {
      "type": "otel",
      "workload_params": {
        "trace_file": "traces/synthetic_otel_traces.jsonl",
        "tokenizer": "meta-llama/Llama-3.1-8B-Instruct",
        "pretokenized": false,
        "total_requests": 10,
        "multiplier_to_sec": 1,
        "inter_turn_multiplier": 1,
        "max_concurrent_sessions": 2
      }
    }
  ]
}
```

```bash
python ./opal/main.py -c configs/defaults_otel.json
```

Full parameter reference, single- vs. multi-session behavior, and pre-tokenizing traces: see [[OTel Trace Replay]].

## RAGWorkload

Synthetic retrieval-augmented-generation workload that exercises prefix-cache reuse across a document corpus. Minimal example, adapted from the shipped `configs/RAG_KVCtiers-HBM-DRAM-DFS.json`:

```json
"workload": {
  "stages": [
    {
      "type": "RAGWorkload",
      "workload_params": {
        "num_documents": 10,
        "document_size": 16384,
        "system_prompt_size": 1024,
        "docs_per_request": 4,
        "output_tokens": 128,
        "request_rate": 0.25,
        "jitter": 0.0,
        "total_requests": 1000,
        "time_duration_sec": -1,
        "max_concurrent_requests": 1
      }
    }
  ]
}
```

```bash
python ./opal/main.py -c configs/RAG_KVCtiers-HBM-DRAM-DFS.json
```

(Other tiering variants: `configs/RAG_KVCtiers-HBM.json`, `configs/RAG_KVCtiers-HBM-DRAM.json`.) Full parameter reference and cache-warmup math: see [[RAG Workload]].

## SC25Workload

A fixed benchmark pattern used for KV-cache performance characterization: it runs a **cold pass** over ten hardcoded prompt sizes (4092 up to 130944 tokens, each with freshly-generated `hash_ids`, guaranteeing cache misses), then a **warm pass** that replays the exact same prompts with the same `hash_ids` (guaranteeing 100% KV-cache hits). Unlike the other workload types, the prompt sizes are hardcoded in `opal/workloads/sc25_blog.py` and are not configurable — `workload_params` can simply be empty:

```json
"workload": {
  "stages": [
    {
      "type": "SC25Workload",
      "workload_params": {}
    }
  ]
}
```

This workload self-terminates after the cold+warm passes complete, so it (like `trace`) is exempt from needing `total_requests`/`time_duration_sec`/`simulation_time` set. Save this as e.g. `configs/sc25-example.json` and run:

```bash
python ./opal/main.py -c configs/sc25-example.json
```

## Multi-stage workloads

The `"stages"` array can chain **different** workload types to run sequentially in one simulation — e.g. a `UniformReqRate` warm-up stage followed by a `trace` replay stage, as in `configs/defaults.json` above. The orchestrator waits for each stage to fully drain (all in-flight responses returned) before starting the next. See [[Running Workloads]] for the termination-condition rules that decide when a stage stops generating new requests.

## Output

Each run writes its artifacts (per-stage stats, graphs, logs, and the resolved config) into a timestamped folder under `simulation-runs/` (or wherever `-o` points). See the README's "What are simulation outputs" section for the full file listing.
