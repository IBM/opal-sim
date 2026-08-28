# vllm_worker.py - vLLM Scheduler Integration

For the full config reference — every `worker.*` default as actually shipped in `configs/defaults.json` (which override some of this page's code-level fallback defaults, noted below) — see [[Configuration Simulation]].

## Overview

`vllm_worker.py` implements an LLM worker that integrates vLLM v1's continuous batching scheduler with SimPy discrete event simulation. The module uses an **interrupt-driven architecture** (no polling) with a **persistent batch model** that accurately simulates real vLLM behavior including preemption, KV cache fetch rate-limiting, and tensor parallelism.

## Key Features

### 1. **Interrupt-Driven Scheduling (No Polling)**
- Two cooperating coroutines: intake loop and scheduler loop
- Both sleep indefinitely when idle, waking only on interrupts
- Eliminates wasteful polling with small timeouts
- Atomic scheduler steps prevent mid-execution corruption

### 2. **vLLM-Style Continuous Batching**
- FIFO scheduling for both prefill and decode phases
- Dynamic batch formation based on resource constraints
- Support for mixed prefill and decode in the same batch
- Persistent batch model: batch rebuilt only when requests complete

### 3. **GPU Memory Block Management**
- Configurable block size (default 16 tokens)
- Tensor parallelism support (KV cache distributed across TP ranks)
- Tracks allocated blocks per request throughout lifecycle
- Proper block allocation/deallocation at request start/completion
- Enforces memory constraints during batch construction

### 4. **KV Cache Integration with Rate-Limiting**
- Prefix matching via `OpalKVCacheEngine.lookup()` (results cached per request)
- Asynchronous KV cache fetching simulation
- Rate-limiting via `max_kvc_ready_requests` to prevent GPU memory hogging
- Proper state transitions for cache operations (WAITING → FETCH_KVC → READY)

### 5. **Chunked Prefill Support**
- Large prompts processed in configurable chunks
- Prevents memory overflow and improves latency
- Chunk size derived from `max_num_batched_tokens`

### 6. **Request Preemption**
- Least-work-done eviction strategy when GPU memory is exhausted
- Maximum 3 preemptions per request before skipping
- Preempted requests reset to WAITING and re-enter queue at front
- Requests near completion (within 5% of decode) are protected

### 7. **Realistic Timing Model**
- Per-batch timing: `max(longest_prefill_chunk, batched_decode_latency)`
- GPU model integration for accurate latency calculations
- Accounts for prefix matching (already-processed tokens) in prefill time
- Batched decode latency calculation

### 8. **Full Request Lifecycle Tracking**
- Complete state machine using single `RequestPhase` enum
- Detailed statistics collection with per-step timestamps
- KVC prefix hit tracking

## Architecture

### Interrupt-Driven Design

```
queue_work() [router calls this]
    │
    ├─ Put request in _worker_local_queue
    └─ Interrupt _check_new_requests (if idle)
              │
              ├─ Get request from queue
              ├─ Add to waiting_requests
              └─ Interrupt _vllm_scheduling_loop (if idle)
                        │
                        └─ _scheduler_step() [atomic, never interrupted mid-work]
```

Two flags control interrupt delivery:
- `_check_new_requests_idle`: True only during intake's idle sleep
- `_scheduler_busy`: False only during scheduler's idle sleep

### Request State Flow

```
New Request → queue_work()
    ↓
WAITING (in waiting_requests)
    ↓
KVC Lookup (cached per request)
    ↓
Has Prefix Match?
    ├─ Yes → Allocate blocks → FETCH_KVC (async retrieve)
    │                              ↓
    │                           READY
    └─ No  → READY
                ↓
        Added to Batch (moved to running_requests)
                ↓
        PREFILL_CHUNKED (process chunks)
                ↓
        Prefill Complete?
                ↓
            DECODE (1 token per step)
                ↓
        All Tokens Generated?
                ↓
            COMPLETED → KVC store → Release blocks → Output
```

Preemption can occur at any running phase:
```
PREFILL_CHUNKED or DECODE → WAITING (progress reset, front of queue)
```

### Core Components

#### 1. **RequestPhase (Enum)**
Single source of truth for request states:
- `WAITING`: In waiting queue, not yet processed for KVC lookup
- `FETCH_KVC`: Async KVC data transfer in progress (blocks already committed)
- `READY`: KVC complete or no prefix match, ready to be scheduled into a batch
- `PREFILL_CHUNKED`: Processing prompt tokens in chunks
- `DECODE`: Generating output tokens one per scheduler step
- `COMPLETED`: Request finished, pending KVC store and memory release

#### 2. **VLLMSchedulerConfig**
Configuration parameters for scheduler:
- `max_num_seqs`: Maximum sequences in a batch (default 256)
- `max_num_batched_tokens`: Maximum tokens per batch (dataclass fallback 2048 if omitted entirely; `configs/defaults.json` ships 8192)
- `chunked_prefill`: Enable chunked prefill (default True)
- `max_model_len`: Maximum sequence length (from model config)
- `gpu_memory_kvcache_bytes`: Available GPU memory for KV cache
- `max_kvc_ready_requests`: Max KVC-fetching requests in waiting queue (dataclass fallback 4 if omitted entirely; `configs/defaults.json` ships 8)
- `lookahead_reqs`: Max waiting requests to scan when batch is non-empty (default 256)

#### 3. **VLLMSchedulerRequest**
Extends `SchedulerRequest` to wrap `LLMRequest`:
- Maintains reference to original request for stats tracking
- Includes `hash_ids` for KVC lookup
- Tracks allocated GPU memory blocks
- Tracks preemption count (max 3)
- Caches KVC lookup results to avoid repeated lookups

#### 4. **BatchMetadata**
Tracks the current persistent batch:
- `prefill_requests` / `decode_requests`: Requests in each phase
- `request_tokens`: Maps request_id to tokens scheduled in this step
- `total_tokens` / `prefill_tokens` / `decode_tokens`: Token accounting

#### 5. **LLMWorkerVLLMScheduler**
Main worker class with key methods:

- `queue_work()`: Entry point from router, interrupt-driven wake-up
- `_vllm_scheduling_loop()`: Main loop, sleeps when idle, processes atomically when active
- `_scheduler_step()`: Execute one iteration (build/run batch, advance states, handle completions)
- `_check_new_requests()`: Intake loop, sleeps until queue_work() interrupts
- `_build_batch()`: 4-phase batch construction with preemption
- `_calculate_batch_time()`: Compute batch execution time
- `_update_request_states_and_stats()`: Advance states and move prefill→decode within batch
- `_move_completed_requests()`: Move completed from running queue and batch
- `_handle_completed_requests()`: Store KVC, release blocks, output results
- `_async_kvc_retrieve()`: Async coroutine for KVC data transfer
- `_preempt_request()`: Evict least-work-done request from batch
- `_can_issue_kvc_fetch()`: Rate-limit KVC fetching to prevent memory hogging

## Configuration

### Required Configuration Parameters

Add to your `sim_config/*.json`:

```json
{
  "worker": {
    "worker_params": {
      "periodic_infra_update_time": 1.0,
      "kvcevent_coalesce_time": 0.1
    },
    "vllm_params": {
      "max_num_seqs": 8,
      "max_num_batched_tokens": 2048,
      "chunked_prefill": true,
      "block_size": 16,
      "max_kvc_ready_requests": 4,
      "lookahead_reqs": 256
    },
    "hw": {
      "gpu": "H100",
      "memory_gb": 80,
      "tflops": 989,
      "mem_bw_TBps": 3.35,
      "tp": 1
    }
  }
}
```

### Configuration Parameters Explained

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `max_num_seqs` | Max sequences in a batch | 256 | Batch size limit |
| `max_num_batched_tokens` | Max tokens per batch | 8192 (`configs/defaults.json`; dataclass fallback 2048 if key omitted) | Token throughput & chunk size |
| `chunked_prefill` | Enable chunked prefill | true | Large prompt handling |
| `block_size` | Tokens per GPU memory block | 16 | Memory granularity |
| `max_kvc_ready_requests` | Max KVC-fetching requests in waiting queue | 8 (`configs/defaults.json`; dataclass fallback 4 if key omitted) | KVC fetch rate-limiting |
| `lookahead_reqs` | Max waiting requests scanned per rebuild | 256 | Batch rebuild scan depth |
| `memory_gb` | GPU memory in GB | - | Total per-GPU memory |
| `tp` | Tensor parallelism degree | 1 | Effective memory = tp * memory_gb |

### Tensor Parallelism Memory Model

With TP enabled, total effective memory for KV cache is `tp * memory_gb`. The model is loaded once (shared across TP ranks), but KV cache is distributed across all ranks:

```
total_gpu_memory = memory_gb * tp * 1024^3
free_for_kvcache = total_gpu_memory - model_size_bytes
total_gpu_blocks = free_for_kvcache / (block_size * key_value_bytes)
```

## Scheduling Algorithm

### Batch Creation Logic (`_build_batch`)

The scheduler builds or rebuilds a batch in 4 phases:

1. **Phase 1: Resume Existing Decode Requests (Highest Priority)**
   - Each decode request generates 1 token per step
   - Calculate blocks needed for the new token
   - If constraints violated → mark for preemption

2. **Phase 2: Resume Existing Prefill Requests**
   - Calculate chunk size (min of ideal chunk, remaining token budget)
   - Calculate blocks needed for the chunk
   - If constraints violated → mark for preemption

3. **Phase 3: Preempt Requests That Cannot Fit**
   - Free GPU blocks from preempted requests
   - Reset progress (prompt_processed=0, decode_tokens_generated=0)
   - Move to front of waiting queue with incremented preemption count

4. **Phase 4: Add New READY Requests from Waiting Queue (with Lookahead Limit)**
   - Perform KVC lookup for WAITING requests (result cached)
   - Rate-limit KVC fetches via `max_kvc_ready_requests`
   - Initiate async KVC retrieve for prefix matches
   - Add READY requests to batch (allocate blocks, move to running)
   - Stop on `max_num_seqs`, token budget exhaustion, or memory exhaustion
   - Apply `lookahead_reqs` limit when batch started non-empty

### Persistent Batch Model

The batch is NOT rebuilt from scratch every step. Instead:
- The batch persists across steps until a request completes
- On completion: remove finished request, then rebuild to fill vacated slot
- This avoids repeated waiting-queue scans per step

### Resource Constraints

Batch creation respects three constraints:

1. **Max Sequences**: `max_num_seqs` limit
2. **Max Tokens**: `max_num_batched_tokens` limit (token budget = compute proxy)
3. **GPU Memory Blocks**: Available free blocks

### GPU Memory Block Management

```python
# Blocks grow incrementally as request progresses
current_tokens = prompt_processed + decode_tokens_generated
new_tokens = current_tokens + tokens_to_add
current_blocks = ceil(current_tokens / block_size)
new_blocks = ceil(new_tokens / block_size)
blocks_needed = new_blocks - current_blocks

# Allocate blocks
if blocks_needed <= free_gpu_blocks:
    request.allocated_blocks += blocks_needed
    free_gpu_blocks -= blocks_needed
```

### KVC Fetch Rate-Limiting

To prevent KVC-fetching requests from hogging GPU memory and causing excessive preemptions of running requests:

```python
# Only allow N requests to be in FETCH_KVC or READY-with-blocks state
kvc_ready_count = count(req for req in waiting_requests
                        if req.phase == FETCH_KVC
                        or (req.phase == READY and req.allocated_blocks > 0))

if kvc_ready_count >= max_kvc_ready_requests:
    skip this request  # Leave in WAITING state
```

### Preemption Strategy

When a running request cannot fit in the batch (memory pressure):

1. Select candidate with **least work done** (lowest work_score)
2. Skip requests near completion (within 5% of decode tokens)
3. Skip requests already preempted 3 times
4. Free all allocated blocks
5. Reset progress to 0, move to front of waiting queue
6. Request will go through KVC lookup again on next scheduling

### Timing Calculation

For each batch, `_calculate_batch_time` delegates entirely to whichever
inference engine `worker.inference_params.model` selects (built once per
simulation, shared by every worker — see [[LLM Inference Roofline Migration Plan]]):

```python
batch_time = gpu_model.estimate(batch)["time_s"]
```

The legacy engine's `estimate()` reproduces the same prefill/decode-latency
split this section used to compute inline: `max(per-prefill-request prefill
latency, batched decode latency)`. The newer roofline model family instead
combines prefill and decode cost into one compute-vs-memory roofline pass —
see the migration doc for the difference.

## Statistics Collected

### Per-Request Statistics

Tracked in `request.llm_request.stats`:

- `worker_arrival_time`: Arrival at worker
- `_3_start_processing_time`: Start of KVC lookup processing
- `_4_request_ready_time`: Request enters READY state
- `_5_gpu_start_time`: Added to batch, GPU processing starts
- `_7_decode_done_time`: Request fully complete
- `scheduler_timestamps`: List of per-step timestamps
- `prefix_hit_tokens`: Number of prefix-matched tokens

### Worker-Level Metrics

- `gpu_busy_time`: Total GPU active time
- Queue lengths: `waiting_requests`, `running_requests`, `completed_requests`
- GPU memory: `total_gpu_blocks`, `free_gpu_blocks`, `kvc_fetch_blocks_in_flight`

## Comparison with worker_single_stage.py

| Feature | worker_single_stage.py | vllm_worker.py |
|---------|------------------------|----------------|
| Batching | No batching (1 req at a time) | Continuous batching |
| Scheduling | Simple FIFO | FIFO with resource constraints |
| Architecture | Polling | Interrupt-driven (no polling) |
| Prefill | All-at-once | Chunked support |
| KVC Lookup | Optional | Integrated with rate-limiting |
| Timing | Per-request sequential | Per-batch parallel |
| State Tracking | Basic | Full vLLM state machine |
| Memory Management | Simple | GPU block management with TP |
| Preemption | None | Least-work-done strategy |
| Batch Model | N/A | Persistent (rebuild on completion) |
| Realism | Low | High (matches vLLM v1) |

## Known Issues

1. `_periodic_infra_updates` is defined but never started in `_run()` — the router never receives load/utilization updates from this worker.

2. Dead code in `_async_kvc_retrieve`: the assert guarantees `actual_kvc_blocks == estimated_kvc_blocks`, making the subsequent adjustment if-block unreachable.

3. Starvation risk: preempted requests reset to `prompt_processed=0` and re-enter at front of waiting queue, but under sustained memory pressure they can be repeatedly preempted (up to cap of 3) with no further recourse or priority escalation.

4. Leaked `request_tokens` entries: Phase 3 preemption removes requests from batch lists but never does `del batch.request_tokens[req.request_id]`.

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Key Log Messages

- `"New request X added to waiting queue (phase: WAITING)"`
- `"Request X: initiating async KVC retrieve for N tokens, allocated B blocks"`
- `"Fetched KVC tokens N for Request X"`
- `"Request X: no prefix match, now READY"`
- `"Added READY request X: tokens=T, blocks_needed=B, free_blocks=F/TOTAL"`
- `"PREEMPTED request X: freed B blocks (work_score=W, preemption_count=C)"`
- `"Request X: prefill complete, transitioning to DECODE"`
- `"Request X: released B blocks after saving KVC store"`
- `"Request X retired on worker.W: prefill=P tokens, decode=D tokens"`
- `"KVC fetch blocked: N KVC-ready requests in waiting queue >= max"`
- `"Scheduler interrupted during work - unexpected while busy"`

### Invariant Checks

Enable detailed memory and batch invariant checking:
```python
# In check_invariants(), set:
enable_variant_checks = True
```

This verifies:
- `init_free_blocks == free_gpu_blocks + sum(all allocated blocks)`
- Batch token budget matches sum of per-request tokens
- No COMPLETED requests in waiting queue

## Module Structure

The `vllm_worker.py` module contains:

1. **Scheduler Data Structures** (lines 89-208)
   - `RequestPhase`: Request state enum (6 states)
   - `VLLMSchedulerConfig`: Scheduler configuration with validation
   - `SchedulerRequest`: Base request class with progress tracking
   - `VLLMSchedulerRequest`: Extended request with LLM integration and preemption tracking
   - `BatchMetadata`: Batch information with per-request token tracking

2. **Scheduler Helper Functions** (lines 215-228)
   - `schedule_prefill_chunk()`: Calculate tokens to process per chunk

3. **Worker Implementation** (lines 235-1738)
   - `LLMWorkerVLLMScheduler`: Main worker class
   - Interrupt-driven architecture with two cooperating coroutines
   - 4-phase batch building with preemption
   - Async KVC retrieval with rate-limiting
   - Persistent batch model

## References

- [vLLM v1 Architecture](https://docs.vllm.ai/en/latest/)
- [Continuous Batching Paper](https://arxiv.org/abs/2309.06180)
- [LMCache Integration](https://github.com/LMCache/LMCache)
- `opal/vllm_worker.py`: Complete implementation

---

**Author**: Opal Simulator Team  
**Date**: 2026-05-11  
**Version**: 3.0.0
