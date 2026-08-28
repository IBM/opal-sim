# KV Cache Manager

The KV cache manager is split across two files with distinct scopes:

- **`opal/kvcache/kvc_manager.py`** — per-worker, local cache engine (`OpalKVCacheEngine`) that chunks tokens into cache keys, and a tiered storage layer (`OpalStorageManager` / `OpalStorageBackend`) that models GPU→CPU/NVMe/distributed-FS placement, lookup, and capacity accounting.
- **`opal/kvcache/kvbm.py`** — the KV Block Manager (`KVBM`), a cluster-global aggregator that consumes `KVCEvent`s emitted by every worker to build a prefix-overlap index used by the router's `MaxPrefix` policy. See [[Router]] for how the router consumes `KVBM.scorer()`.

This page covers the manager/storage internals. For the worker-side state machine that calls into this engine (`WAITING → FETCH_KVC → READY`, rate-limiting via `max_kvc_ready_requests`), see [[vLLM Worker]]. For the full `kvc.*` config key reference, see [[Configuration Simulation]].

## Overview

`OpalKVCacheEngine` is described in its own docstring as resembling the LMCache/LMCacheEngine API. It exposes three operations, mirroring the vLLM/LMCache integration split:

- **`store()`** — worker's write path: turn generated tokens into chunk keys and push them into the tiered storage backends.
- **`retrieve()`** — worker's read path: given tokens, find which chunks are already cached and pull them back (attributed per storage tier).
- **`lookup()`** — a cheaper read-only check (no data movement) used by the scheduler to decide how many leading tokens already have a cache hit, without paying transfer cost.

Each worker owns its own `OpalKVCacheEngine` instance (constructed with a `worker_id`); `KVBM` additionally constructs one privately (with `worker_id=-1`, "the global instance") purely to reuse its `token_database` for chunking — it never stores or retrieves anything through it.

## Tokens → Chunks → Keys

Both `store()`, `retrieve()`, and `lookup()` funnel through `OpalTokenDatabase.process_tokens()`, which:

1. Splits the token list into fixed-size chunks (`kvc.chunk_size`, default 256 — see [[Configuration Simulation]]).
2. Computes a rolling **prefix hash** per chunk (`_prefix_hash`): each chunk's hash is `sha256((previous_chunk_hash, this_chunk_tokens))`, so the hash of chunk *i* depends on all preceding chunks — this is what makes matching prefix-sensitive rather than content-addressed per chunk in isolation.
3. Optionally (`make_key=True`) wraps each hash into an `OpalCacheEngineKey` — a dataclass carrying `fmt`, `model_name`, `world_size`, `worker_id`, `chunk_hash`, and `dtype_str`, hashable and stringifiable (`to_string()`/`from_string()`) for use as a dict key inside storage backends.

The hashing itself goes through `sha256_fast()`, a hand-optimized SHA-256 wrapper with a process-wide memoization cache (`_hash_cache`, keyed by Python's built-in `hash()` of the input tuple) to avoid repeated hashing of identical prefixes — `get_hash_cache_stats()` / `clear_hash_cache()` are exposed for instrumentation/tests. A slower, pickle-based `sha256()` is kept around and selectable via `get_hash_fn_by_name("sha256_legacy")`, but nothing in the codebase selects it — `OpalTokenDatabase.__init__` always hardcodes `self.hash_algorithm = "sha256"`, so the legacy path is effectively dead unless a caller is added.

`OpalEngineMetadata` (model name, world size, worker id, tensor format/dtype/shape, MLA flag) is attached to every key so that caches from different models/dtypes/tensor-parallel layouts never collide, even if the token content and chunk boundaries happen to match.

## Storage tiers and lookup/hit logic

`OpalStorageManager` owns an ordered dict of `OpalStorageBackend` instances, one per entry in `kvc.kvc_tiers` (default `["CPUMemory"]`; also supports `LocalNVMe`, `DistributedFS` — see [[Configuration Simulation]] for their bandwidth/latency/capacity defaults). Each backend pairs an `AbstractDevice` performance model (from `opal.infra.io_model`) with a plain Python dict (`self._index`) that serves as the actual key→`OpalMemoryObj` store — there is no real memory allocator here, just a hash table plus a simulated I/O cost for every access.

**Lookup / hit determination is a strict prefix walk, tier by tier, in the order tiers are listed in `kvc_tiers`:**

- `OpalStorageManager.batched_contains(keys)` iterates the configured tiers in order. For each tier it calls `backend.batched_contains(keys)`, which (in the generic base implementation) walks the given keys **in order** and stops at the first miss (`contains()` returns `False`) — so a hit count of *N* means "the first *N* keys are present," not "N keys are present somewhere in this tier."
- Only the *remaining* (unmatched) keys are then tried against the next tier. Hits are accumulated across tiers into a single `total_hit_chunks`, and `block_mapping` records which tier served which contiguous key range.
- This means the overall cache hit for a request is the length of its **contiguous matched prefix across all tiers combined**, honoring the same prefix-sensitivity baked into the chunk hashing above. A gap (a missing chunk followed by present chunks) is never counted as a hit for anything after the gap.
- `OpalKVCacheEngine.lookup()` layers one more prefix concept on top: `num_computed_tokens` (already-processed tokens, e.g. from a previous partial schedule) is used to skip chunks that are already known-computed (`if end <= aligned_computed_tokens: continue`) before even asking the storage manager. The return value is the token offset up to which the prefix is available (computed-or-cached), which is exactly the quantity [[vLLM Worker]] uses to decide `FETCH_KVC` vs `READY`.

**Retrieve** (`OpalKVCacheEngine.retrieve()`) is lookup's counterpart that actually moves data: it builds the same chunk list, resolves `get_block_mapping()` (same tier-ordered prefix logic as above) to find where each hit chunk lives, then calls `batched_get()` per tier to pull memory objects out. It returns a boolean mask over the input tokens plus a `tier_tokens` dict giving the exact token count served from each tier — useful for tier-attribution stats. Notably the actual GPU-copy step is stubbed out: the code that would call a GPU connector to transfer `reordered_chunks` onto the GPU is present only as a commented-out block, so no data-transfer time for retrieval into the worker's live compute is charged. (`max_fetch`, its truncation parameter, mirrors the rate-limiting note in [[vLLM Worker]] about lookup outrunning retrieve.)

**Store** (`OpalKVCacheEngine.store()`) chunks the tokens, wraps each chunk as an `OpalMemoryObj(size, tokens, "gpu")` (the `location_tier` field is set to `"gpu"` unconditionally at construction and never updated afterward — effectively dead/unused metadata once the object is placed into a specific tier's backend), and calls `OpalStorageManager.batched_put()`.

## Storing / capacity accounting — no eviction

`batched_put()` walks tiers in configured order and, per tier, per key:

- Skips a key if it's already present in that tier (`backend.contains(k)`).
- Otherwise stores it via `submit_put_task()` **only if it fits** (`memory_obj.size <= dev.capacity_remaining_bytes`), decrementing `capacity_remaining_bytes` and inserting into the backend's `_index` dict.
- The moment a key does *not* fit in the current tier, the loop `break`s out of that tier entirely (leaving any later keys in the batch unstored at this tier) and the **unstored suffix of the batch is retried against the next configured tier**.

There is no eviction policy anywhere in this code path. Once a tier's `capacity_remaining_bytes` is exhausted, it simply stops accepting new chunks for the rest of the simulation (or until it happens to have room from a request that is smaller) — nothing is ever removed from `self._index` to make room. `OpalStorageBackend.remove()`, `pin()`, and `unpin()` are all declared but `raise NotImplementedError`. In other words, `kvc_manager.py` currently models tiered *capacity* but not tiered *replacement*; the practical effect in a long-running simulation is that whichever tier fills up first permanently stops caching new chunks and any excess simply falls through to (and eventually exhausts) the next tier, with cache misses reappearing once every tier is full. If eviction is intended, `apc`/`gpu_eviction.py`-style logic (see the GPU APC design elsewhere in this repo) does not currently live here.

Each successful new insertion produces a `KVCEvent(worker_id, hash(key), src_tier=-1, dst_tier=tier_index, KVCEventType.INSERT)`. These are collected per `batched_put()` call and handed to the owning worker via `worker.append_kvc_events(kvc_events)` (looked up through `self.opal_env.registry.get_worker(self.worker_id)`), which the worker then batches and periodically forwards to the router (`kvcevent_coalesce_time`, see [[Configuration Simulation]] and [[vLLM Worker]]) for KVBM ingestion.

Note that `KVCEventType` also defines `DELETE`, `MOVE`, and `COPY`, but nothing in `kvc_manager.py` ever emits them (only `INSERT` is produced), and `KVBM`'s event handler (`OpalWorkerState.process_kvc_event`, below) actively `raise`s if it ever receives anything other than `INSERT`. This is consistent with there being no eviction/move logic to report — the enum values exist for a future feature that isn't wired up yet.

## KVBM: cluster-wide prefix state for routing

`KVBM` (in `kvbm.py`) is explicitly documented as giving "a cluster-global view of the kv cache content," fed by the `KVCEvent`s described above and consumed by the router (see [[Router]]'s "MaxPrefix Details" section). It maintains two structures:

- **`_worker_state: dict[worker_id, OpalWorkerState]`** — one `OpalWorkerState` per worker, each holding a `_prefix_set` (a plain Python `set` of chunk hashes that worker has ever inserted; `process_kvc_event` just does `self._prefix_set.add(hashes)` on `INSERT`, and hashes are never removed — consistent with there being no eviction events to remove them on).
- **`_chunk_to_workers: dict[chunk_hash, set[worker_id]]`** — a reverse index built alongside `_worker_state`, mapping each chunk hash to the set of workers known to hold it. This is a deliberate performance optimization (per the inline comment) so `scorer()` doesn't have to scan every worker for every request.

`process_kvc_events(events)` and `process_system_events(events)` are the ingestion entry points the router calls after batching worker-reported events (see [[Router]]'s "Event Processing" section); both lazily create an `OpalWorkerState` the first time a given `worker_id` is seen.

### Scoring (`KVBM.scorer(req)`)

Given an `LLMRequest`, `scorer()`:

1. Chunks `req.hash_ids` using the same `OpalTokenDatabase` chunking logic as the storage engine (via a locally-constructed `OpalKVCacheEngine(..., worker_id=-1)` used only for its `token_database` — see Overview above).
2. Uses `_chunk_to_workers` to build a `candidate_workers` set — only workers known to hold *at least one* of the request's chunk hashes are considered. If no worker holds any chunk, it returns `{}` immediately (the router then falls back to random selection, per [[Router]]).
3. For each candidate worker, calls `OpalWorkerState.match_prompt(chunk_hashes)`, which walks the request's chunk hashes **in order** and stops at the first hash not in that worker's `_prefix_set` — so, consistent with the storage-tier lookup logic above, this is a contiguous-prefix match, not a set-overlap count.
4. The score per worker is `matched_chunks / len(chunk_hashes)` — the fraction of the request's prefix (from the start) that worker already has cached. The router picks the highest-scoring worker (ties broken randomly).

This is a purely event-driven, workers-report-what-they-store model: KVBM never queries a worker directly and never sees evictions (because none are ever emitted), so its view of a worker's cache can only grow, never shrink, for the lifetime of the simulation — it is an increasingly optimistic approximation of real per-worker cache state as a run progresses, since real KV caches would eventually evict.

## Known Issues

1. **No eviction implemented.** As detailed above, storage backends only ever grow their `_index` until `capacity_remaining_bytes` runs out, at which point that tier permanently stops accepting new chunks for the rest of the run. `remove()`/`pin()`/`unpin()` are unimplemented stubs on `OpalStorageBackend`.
2. **KVBM's cached prefix state is monotonically growing and never invalidated.** Because no `DELETE`/`MOVE` `KVCEvent`s are ever produced (issue 1), `OpalWorkerState._prefix_set` only accumulates hashes; `KVBM`'s router-facing view of "what's cached where" therefore increasingly diverges from reality (particularly once a tier fills up and genuinely starts missing) the longer a simulation runs.
3. **`location_tier` field on `OpalMemoryObj` is dead metadata.** It's hardcoded to `"gpu"` at construction in `store()` and never read or updated to reflect the tier it actually lands in; the real tier assignment lives only in which backend's `_index` the object ends up in.
4. **Legacy hash path is unreachable.** `get_hash_fn_by_name("sha256_legacy")` exists to select the pickle-based `sha256()`, but `OpalTokenDatabase` always hardcodes `hash_algorithm = "sha256"` (the fast/cached path), so the legacy function has no live caller.
5. **`allocate_and_copy_objects()` is an unused stub.** It's a module-level function with a hardcoded/overwritten `time` variable (`time = 0.1` immediately followed by `time = 0.2`, so the first assignment is dead) and `print()` debug statements; it is not called anywhere in `kvc_manager.py` or (as far as this page's scope covers) the worker code.
6. **GPU-copy step in `retrieve()` is commented out.** The block that would call a GPU connector's `batched_to_gpu()` to actually land retrieved memory objects on the GPU is dead code (a triple-quoted comment), so retrieval time/cost past the storage-tier fetch is not modeled.
7. **`OpalKVCacheEngine.__del__` logs at `WARNING` level** for what is purely informational shutdown-time counter reporting (`counter_lookup`, `counter_lookup_tokens`) — not an error condition, just a logging-level mismatch worth knowing about if you're grepping simulation logs for real warnings.

## Cross-references

- [[Router]] — how `MaxPrefix` consumes `KVBM.scorer()`, and how the router batches/forwards `KVCEvent`s.
- [[vLLM Worker]] — the per-request `WAITING → FETCH_KVC → READY` state machine built on top of `OpalKVCacheEngine.lookup()` / async retrieve, and KVC-fetch rate-limiting.
- [[Configuration Simulation]] — full `kvc.*` key reference (tiers, `chunk_size`, the dead `save_unfull_chunk` flag) plus `worker.worker_params.kvcevent_coalesce_time`.
- [[RAG Workload]] — a workload designed to exercise tiered prefix-cache reuse across a document corpus.
