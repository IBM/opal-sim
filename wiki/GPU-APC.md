## GPU APC (Automatic Prefix Caching)

GPU APC models vLLM-style **Automatic Prefix Caching**: full KV-cache blocks that are already resident in GPU HBM are reused across requests (and across turns of the same session) instead of being recomputed. When a new request shares a prompt prefix with something already cached in the GPU, those blocks are matched, ref-counted, and shared — the request skips prefill for the matched tokens.

APC sits *above* the distributed KV-cache (KVC) tiers. The lookup order for a prefix is fastest-first: **GPU APC → CPUMemory → LocalNVMe → DistributedFS**. When APC evicts an idle block, it can write it through to the first KVC tier (CPU DRAM) so the prefix is still reusable from a slower tier later.

APC is a per-worker feature implemented in [`opal/worker/vllm_worker.py`](../opal/worker/vllm_worker.py); the eviction policy lives in [`opal/kvcache/eviction_policy.py`](../opal/kvcache/eviction_policy.py).

### Ready-to-run example

The repo ships a working config with APC enabled:

- **Config:** [`configs/defaults_apc.json`](../configs/defaults_apc.json) — a worker with `enable_gpu_apc: true`, LRU eviction, and a CPU DRAM tier for evicted-block write-through.

Run it directly:

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python ./opal/main.py -c ./configs/defaults_apc.json
```

To see APC activity in the log (evictions, hits, proactive migration):

```bash
OPAL_LOG_LEVEL=INFO PYTHONPATH=`pwd`:$PYTHONPATH python ./opal/main.py -c ./configs/defaults_apc.json
```

> `defaults_apc.json` ships with two workload stages — a synthetic `UniformReqRate` stage and an `otel` trace stage. APC works with either; multi-turn / agentic workloads (the `otel` stage, see [[OTel Trace Replay]]) benefit most because later turns re-hit the prefix cached from earlier turns.

### The parts of `defaults_apc.json` that drive APC

APC is configured entirely inside `worker.vllm_params`, with two other sections mattering for capacity and write-through: `worker.hw` (how much HBM exists) and `kvc` (where evicted blocks go).

#### 1. `worker.vllm_params` — the APC knobs

```jsonc
"vllm_params": {
  "enable_gpu_apc": true,        // master switch — nothing below matters if false
  "apc_eviction_policy": "lru",  // which policy decides what to evict
  "apc_lru_ttl": 0,              // seconds a block is immune to eviction after last access
  "block_size": 16,              // KV block granularity (tokens per block == hashing unit)
  "gpu_memory_utilization": 0.9, // (standard vLLM) fraction of HBM usable for KV cache
  "max_num_seqs": 256,
  "max_num_batched_tokens": 8192,
  "max_kvc_ready_requests": 8,
  "chunked_prefill": true
}
```

| Key | Default | What it does |
|---|---|---|
| `enable_gpu_apc` | `false` | **Master switch.** Must be `true` to turn APC on. When off, no APC state is allocated and the eviction monitor never starts. |
| `apc_eviction_policy` | `"lru"` | Which eviction policy to instantiate. Currently **only `"lru"`** is implemented (`make_apc_policy` raises on anything else). |
| `apc_lru_ttl` | `0` | Seconds a block stays **immune to eviction** after its last access. `0` = no TTL (evict purely by recency). A positive value protects hot prefixes from being evicted under a short burst — evict() skips any idle block still inside its window. |
| `block_size` | `16` | Tokens per KV block. This is also the **APC hashing granularity** — one cached block == `block_size` tokens == one physical GPU block. Smaller blocks = finer prefix sharing but more hashing overhead. |
| `gpu_memory_utilization` | `0.9` | Standard vLLM knob: fraction of GPU memory available for KV cache. Together with `worker.hw.memory_gb` it sets `total_gpu_blocks`, which caps how much can live in APC. |

Two additional, **optional** knobs control *proactive* eviction (draining idle APC blocks to CPU DRAM before HBM fills up). They aren't in the shipped file but are read if present:

| Key | Default | What it does |
|---|---|---|
| `hbm_eviction_threshold` | `0.9` | When idle-APC utilization of non-active GPU capacity exceeds this fraction, the background monitor proactively evicts (migrates) blocks to the CPU tier. |
| `apc_eviction_check_interval_sec` | `1.0` | Base interval for the background eviction monitor. It self-tightens toward `0.1 s` as utilization climbs above the threshold. |

#### 2. `worker.hw` — how much GPU there is to cache in

```jsonc
"hw": {
  "gpu": "H100",
  "memory_gb": 80,   // HBM per GPU — larger = more APC capacity
  "tflops": 989.5,
  "mem_bw_TBps": 3.3,
  "tp": 1            // tensor-parallel degree; KV cache is spread across TP ranks
}
```

`total_gpu_blocks = (memory_gb * tp * 1024^3 - model_size_bytes) / (block_size * kv_bytes_per_token)`. More HBM (or higher `tp`) means more blocks, means a bigger prefix cache and fewer evictions.

#### 3. `kvc` — where evicted blocks land (write-through target)

```jsonc
"kvc": {
  "kvc_tiers": ["CPUMemory"],
  "chunk_size": 16,          // KVC-tier hashing granularity
  "CPUMemory": {
    "bandwidth_GBps": 50,
    "latency_nsec": 200,
    "concurrency": 1000000,
    "capacity_GB": 8
  }
}
```

When APC evicts an idle block whose prefix lands **exactly on a `chunk_size` boundary**, it fires a background `store()` to the first KVC tier (`CPUMemory` here). That prefix can then still be re-fetched from CPU DRAM instead of being fully recomputed. If you list no KVC tiers, evicted blocks are simply dropped (recompute-on-miss).

> Keep `block_size` (APC) and `kvc.chunk_size` aligned (both `16` here). APC only writes through on chunk-aligned prefixes because the KVC tier re-hashes at its own `chunk_size`; a mismatch means many evictions are never persisted.

### How APC decides what to keep (LRU + ref counting)

- **Ref counting.** Every block a request is actively using is *pinned* (`ref_count > 0`) and cannot be evicted. When a request finishes with a block it *decrefs*; the block stays discoverable and shareable but becomes evict-eligible once its count hits 0. Shared prefixes across concurrent requests just bump the same block's ref count.
- **LRU ordering.** Only idle (`ref_count == 0`) blocks sit in the eviction queue, front = least-recently-used. A block released back to idle re-enters at the most-recently-used end. Eviction cost is O(victims), not O(resident) — pinned blocks are never scanned.
- **TTL.** With `apc_lru_ttl > 0`, an idle block still inside its window is skipped (left in place) during eviction.

### Tuning APC

| Symptom / goal | Turn this knob |
|---|---|
| APC never fills / few hits | Workload has little prefix overlap, or `total_gpu_blocks` is huge — nothing to tune. Try a multi-turn/agentic (`otel`) workload. |
| Hot prefixes evicted too eagerly under bursts | Raise `apc_lru_ttl` to protect recently-used blocks for N seconds. |
| Want more room in APC | Raise `worker.hw.memory_gb`, raise `tp`, or lower `gpu_memory_utilization` headroom pressure. |
| Evicted prefixes aren't reusable later | Add / enlarge a `CPUMemory` tier in `kvc` and keep `chunk_size == block_size`. |
| Proactive migration too aggressive/lazy | Tune `hbm_eviction_threshold` (lower = migrate earlier) and `apc_eviction_check_interval_sec`. |
| Finer prefix sharing | Lower `block_size` (e.g. `8`) — more sharing, more hashing overhead; align `kvc.chunk_size` with it. |

### Verifying APC is working

At `OPAL_LOG_LEVEL=INFO` you'll see, per worker:

- `GPU APC enabled: policy=LRUPolicy, block_size=16, ...` at startup.
- `[LRU.evict] requested=… evicted=… scanned=… ttl_skipped=… eviction_queue=… pinned=… resident=…` on each eviction.
- `[APC] [Evict] Memory util=… > eviction threshold=… Proactively migrated N block(s) to CPU DRAM …` when the background monitor fires.

Per-tier prefix-cache hit rate (with `apc` as the fastest tier) is reported in the run's statistics — see [[Statistics collection and processing]]. Each simulation run is saved under `simulation-runs/sim-…/` (see the [README](../README.md) "Output" section).

### See also

- [[OTel Trace Replay]] — agentic/multi-turn workloads that exercise APC hardest.
- [[Configuration Simulation]] — the full config-file structure.
- [KVCache manager](KVCache-manager) — the tiered KV cache APC writes through to.
