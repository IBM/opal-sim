# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import re
import time

"""
TODO(atr) - Known issues to address:

1. _periodic_infra_updates is defined but never started in _run().
   The router never receives load/utilization updates from this worker.

2. Dead code in _async_kvc_retrieve: the assert on line ~1508 guarantees
   actual_kvc_blocks == estimated_kvc_blocks, making the subsequent
   if-block unreachable.

3. Starvation risk: preempted requests reset to prompt_processed=0 and get
   inserted at the front of waiting_requests, but under sustained memory
   pressure they can be repeatedly preempted (up to cap of 3) with no
   further recourse or priority escalation.

4. Leaked request_tokens entries: Phase 3 preemption removes requests from
   batch lists but never does `del batch.request_tokens[req.request_id]`.

5. Unused variable: total_requests in _periodic_infra_updates (line ~608)
   is computed but never used.
"""

"""
LLM Worker with vLLM v1 Scheduler Integration

This module implements a worker that integrates vLLM v1's continuous batching scheduler
with SimPy discrete event simulation. It combines scheduling logic with worker execution.

Key Features:
- Interrupt-driven scheduling (no polling; sleeps indefinitely when idle)
- vLLM-style continuous batching with FIFO scheduling
- GPU memory block management (configurable block size, default 16 tokens)
- Tensor parallelism support (KV cache distributed across TP ranks)
- KV cache prefix matching and async retrieval with rate-limiting
- Chunked prefill support for large prompts
- Per-batch timing based on GPU model (prefill + batched decode)
- Preemption of running requests when GPU memory is exhausted
- Persistent batch model (batch rebuilt only when requests complete)

State Transitions:
    WAITING → FETCH_KVC (if prefix match) → READY → PREFILL_CHUNKED → DECODE → COMPLETED
    WAITING → READY (if no prefix match) → PREFILL_CHUNKED → DECODE → COMPLETED
    Any running phase → WAITING (on preemption, progress reset)

Request States (RequestPhase):
    - WAITING: In waiting queue, not yet processed for KVC lookup
    - FETCH_KVC: Async KVC data transfer in progress
    - READY: KVC complete or no prefix match, ready to be scheduled into a batch
    - PREFILL_CHUNKED: Processing prompt tokens in chunks
    - DECODE: Generating output tokens one per scheduler step
    - COMPLETED: Request finished, pending KVC store and memory release
"""

"""
Design decisions (2026-03-23, resolved):
 * KVC fetch rate-limiting: max_kvc_ready_requests caps how many waiting requests
   can have committed GPU blocks for KVC fetching, preventing live-lock under long queues.
 * Preemption: least-work-done strategy with max 3 preemptions per request.
   Preempted requests reset to WAITING and re-enter the queue at the front.
 * Persistent batch: the scheduler builds a batch once and runs it until a request
   completes, then rebuilds. No repeated waiting-queue scans per step.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Generator, List, Optional, cast
import logging
import itertools

import simpy

from opal.events import KVCEvent, SystemEvent
from opal.gpu_model import GPUModel
from opal.kvc_manager import OpalKVCacheEngine, OpalTokenDatabase
from opal.llm_model import OpalModelConfig
from opal.request import LLMRequest
from opal.util import parse_bool, safe_process
from opal.apc_policy import BaseAPCPolicy, make_apc_policy
from opal.gpu_apc import GpuApc

# ================================================================================
# SCHEDULER DATA STRUCTURES
# ================================================================================


class RequestPhase(Enum):
    """Request processing phases in vLLM scheduler."""

    WAITING = "waiting"
    FETCH_KVC = "fetch_kvc"
    READY = "ready"
    PREFILL_CHUNKED = "chunked"
    DECODE = "decode"
    COMPLETED = "completed"


@dataclass
class VLLMSchedulerConfig:
    """Configuration parameters for vLLM v1 scheduler."""

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    max_model_len: int = 8192
    chunked_prefill: bool = True
    gpu_memory_kvcache_bytes: int = 16 * 1024**3
    max_kvc_ready_requests: int = 4  # Max number of KVC-ready requests waiting in the waiting queue
    lookahead_reqs: int = 256  # Max number of waiting requests to scan when batch is non-empty
    enable_gpu_apc: bool = False  # GPU Automatic Prefix Caching 

    def __post_init__(self):
        if False == self.chunked_prefill:
            self.max_num_batched_tokens = self.max_model_len * self.max_num_seqs
            self.log = logging.getLogger("VLLMSchedulerConfig")
            self.log.warning(
                f"chunked_prefill is False, max_num_batched_tokens {self.max_num_batched_tokens} is set to max_model_len {self.max_model_len} * max_num_seqs {self.max_num_seqs}"
            )
        self.validate()

    @property
    def prefill_chunk_size(self) -> int:
        """Prefill chunk size derived from max_num_batched_tokens."""
        if True == self.chunked_prefill:
            val = self.max_num_batched_tokens
        else:
            val = self.max_model_len
        return val

    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.max_num_seqs > 0, "max_num_seqs must be positive"
        assert self.max_num_batched_tokens > 0, "max_num_batched_tokens must be positive"
        assert self.max_model_len > 0, "max_model_len must be positive"
        assert self.gpu_memory_kvcache_bytes > 0, "gpu_memory_kvcache_bytes must be positive"
        assert self.max_kvc_ready_requests > 0, "max_kvc_ready_requests must be positive"


@dataclass
class SchedulerRequest:
    """Represents a single request being tracked by the scheduler."""

    request_id: int
    prompt_tokens: int
    output_tokens: int
    phase: RequestPhase = RequestPhase.WAITING
    prompt_processed: int = 0
    decode_tokens_generated: int = 0
    kvc_fetch_in_progress: bool = False
    apc_owned_hashes: set = field(default_factory=set) # Blocks this request references; decref'd at retirement.
    apc_resolved_tokens: int = 0 # Tokens already processed by APC resolution. 
    apc_chain_hash: Optional[int] = None # Prefix-hash so far — lets hashing resume from the suffix.
    apc_hash_ids_cache: Optional[list] = None # Cached prompt+decode token ids, extended in place.
    apc_private_blocks: int = 0 # This request's blocks not in the APC table; freed directly.

    def is_prefill_complete(self) -> bool:
        """Check if prefill phase is complete."""
        return self.prompt_processed >= self.prompt_tokens

    def is_decode_complete(self) -> bool:
        """Check if decode phase is complete."""
        return self.decode_tokens_generated >= self.output_tokens

    def is_completed(self) -> bool:
        """Check if request is fully completed."""
        return self.phase == RequestPhase.COMPLETED

    def remaining_prefill_tokens(self) -> int:
        """Get number of prefill tokens remaining."""
        return max(0, self.prompt_tokens - self.prompt_processed)

    def remaining_decode_tokens(self) -> int:
        """Get number of decode tokens remaining."""
        return max(0, self.output_tokens - self.decode_tokens_generated)


class VLLMSchedulerRequest(SchedulerRequest):
    """Extended SchedulerRequest that wraps an LLMRequest for vLLM scheduling."""

    def __init__(self, llm_request: LLMRequest):
        """Initialize from an LLMRequest."""
        super().__init__(
            request_id=llm_request.id,
            prompt_tokens=llm_request.input_length,
            output_tokens=llm_request.output_length,
        )
        self.llm_request = llm_request
        self.hash_ids = llm_request.hash_ids
        self.allocated_blocks = 0
        self.preemption_count = 0  # Track number of times this request has been preempted


@dataclass
class BatchMetadata:
    """Metadata about a scheduled batch."""

    prefill_requests: List[SchedulerRequest] = field(default_factory=list)
    decode_requests: List[SchedulerRequest] = field(default_factory=list)
    total_tokens: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    kv_memory_used: int = 0
    can_fit: bool = True
    request_tokens: Dict[int, int] = field(default_factory=dict)  # Maps request_id to tokens in this batch

    def num_sequences(self) -> int:
        """Total number of sequences in batch."""
        return len(self.prefill_requests) + len(self.decode_requests)

    def is_empty(self) -> bool:
        """Check if batch is empty."""
        return self.num_sequences() == 0


# ================================================================================
# SCHEDULER HELPER FUNCTIONS
# ================================================================================


def schedule_prefill_chunk(request: SchedulerRequest, config: VLLMSchedulerConfig) -> int:
    """
    Determine how many tokens to process for a prefill request.

    Args:
        request: Request in prefill phase
        config: Scheduler configuration

    Returns:
        Number of tokens to process in this step
    """
    remaining = request.remaining_prefill_tokens()
    return min(remaining, config.prefill_chunk_size)


# ================================================================================
# VLLM WORKER IMPLEMENTATION
# ================================================================================


class LLMWorkerVLLMScheduler:
    """
    LLM Worker with vLLM v1 continuous batching scheduler integration.

    Architecture: interrupt-driven, no polling. Two coroutines cooperate:
    - _check_new_requests: intake loop, sleeps until queue_work() interrupts it
    - _vllm_scheduling_loop: scheduler loop, sleeps until intake interrupts it

    Request lifecycle:
    1. queue_work() → intake wakes → request enters WAITING
    2. _build_batch() performs KVC lookup:
       - prefix match → allocate blocks, async retrieve (FETCH_KVC) → READY
       - no match → READY immediately
    3. READY requests are added to persistent batch → PREFILL_CHUNKED
    4. Batch executes each step until prefill complete → DECODE
    5. Decode generates 1 token/step until output complete → COMPLETED
    6. Completed requests store KVC, release blocks, exit to router

    Preemption: when GPU memory is exhausted, the request with least work done
    is evicted from the batch, its blocks freed, and it re-enters WAITING.
    """

    _id = itertools.count(start=0)

    def __init__(self, opal_env, opal_config, output_req_queue, infra_update_queue):
        self.opalEnv = opal_env
        self.opalConfig = opal_config
        self.simpy_env = self.opalEnv.simpy_env
        self.id = next(LLMWorkerVLLMScheduler._id)
        self.log = logging.getLogger(str(self))

        # Configuration
        self.periodic_infra_update_time = self.opalConfig["worker"]["worker_params"]["periodic_infra_update_time"]
        self.kvcevent_coalesce_time = self.opalConfig["worker"]["worker_params"]["kvcevent_coalesce_time"]

        # KVC events
        self._pending_kvc_events: list[KVCEvent] = []

        # Input queue from router (unbounded; backpressure is via GPU block limits)
        self._worker_local_queue = simpy.Store(self.simpy_env)

        # KVC manager for prefix matching and retrieval
        self._kvc_manager = OpalKVCacheEngine(self.opalEnv, self.opalConfig, self.id)

        # GPU model for timing calculations
        self.gpu_model = GPUModel(opal_env, opal_config)
        self.gpu_busy_time = 0

        # Output queues
        self.router_output_finished_req_queue = output_req_queue
        self.infra_update_queue = infra_update_queue

        # Model configuration
        self.model = self.opalConfig["model"]["model_params"]["name"]
        self.gpu = self.opalConfig["worker"]["hw"]["gpu"]
        self.model_config: OpalModelConfig = self.opalEnv.llm_model

        # Initialize vLLM scheduler configuration
        self.scheduler_config = self._init_scheduler_config()

        # GPU memory block management
        self.block_size = self.opalConfig["worker"]["vllm_params"].get("block_size", 16)
        self._init_gpu_memory_blocks()

        # KVC fetch memory tracking
        self.kvc_fetch_blocks_in_flight = 0  # Blocks currently being fetched

        # Invariant: free_gpu_blocks + self._apc.resident == total_gpu_blocks
        # (a registered block holds memory regardless of ref_count; ref_count
        # only governs eviction eligibility, not this capacity invariant).
        # Dedicated token database for APC: chunk_size == block_size, partial
        # last block excluded so a partially filled block is never hashed.
        _apc_token_db = OpalTokenDatabase(
            self.block_size,
            self._kvc_manager.token_database.metadata,
        )
        _apc_token_db.save_unfull_chunk = False
        # Pluggable eviction policy (this build ships: lru).
        _policy_name = self.opalConfig["worker"]["vllm_params"].get("apc_eviction_policy", "lru")
        _policy_ttl = float(self.opalConfig["worker"]["vllm_params"].get("apc_lru_ttl", 0.0))
        _apc_policy: BaseAPCPolicy = make_apc_policy(
            _policy_name,
            ttl=_policy_ttl,
            clock=lambda: self.simpy_env.now,
        )
        self._apc = GpuApc(_apc_policy, _apc_token_db, self.block_size)

        # Request queues by state
        self.waiting_requests: List[VLLMSchedulerRequest] = []
        self.running_requests: List[VLLMSchedulerRequest] = []
        self.completed_requests: List[VLLMSchedulerRequest] = []

        # Persistent batch state - the batch IS the running requests
        self.current_batch: Optional[BatchMetadata] = None

        # Scheduler busy state:
        # - True for all active states, including normal work processing and the
        #   code paths immediately before/after idle wakeup
        # - False ONLY while the scheduler coroutine is actually blocked inside
        #   its indefinite idle sleep (`yield timeout(float("inf"))`)
        #
        # IMPORTANT:
        # This flag must track the exact lifetime of the idle sleep, otherwise
        # queue_work()/intake may send stale interrupts while the scheduler is
        # already active again. Use try/finally around the idle sleep to keep
        # the flag accurate.
        self._scheduler_busy = True

        # Intake loop idle state:
        # - True ONLY while _check_new_requests() is actually blocked inside its
        #   indefinite idle sleep (`yield timeout(float("inf"))`)
        # - False in all active intake states
        #
        # IMPORTANT:
        # This flag must also track the exact lifetime of the idle sleep.
        # If it remains True after wakeup, queue_work() will keep sending
        # unnecessary interrupts even though intake is already draining the queue.
        self._check_new_requests_idle = False

        # Log initialization
        self.log.info(
            f"{self} vLLM Scheduler Worker initialized with:"
            f"\n  - max_num_seqs: {self.scheduler_config.max_num_seqs}"
            f"\n  - max_num_batched_tokens: {self.scheduler_config.max_num_batched_tokens}"
            f"\n  - chunked_prefill: {self.scheduler_config.chunked_prefill}"
            f"\n  - max_model_len: {self.scheduler_config.max_model_len}"
            f"\n  - prefill_chunk_size (derived): {self.scheduler_config.prefill_chunk_size}"
            f"\n  - block_size: {self.block_size} tokens"
            f"\n  - total_gpu_blocks: {self.total_gpu_blocks}"
            f"\n  - free_gpu_blocks: {self.free_gpu_blocks}"
            f"\n  - max_kvc_fetch_blocks: {self.max_kvc_fetch_blocks}"
            f"\n  - max_kvc_ready_requests: {self.scheduler_config.max_kvc_ready_requests}"
            f"\n  - GPU APC: {'ENABLED' if self.scheduler_config.enable_gpu_apc else 'DISABLED'}"
            f"\n  - policy={_policy_name}"
        )
        self.sanity_checks()
        # Start simulation processes
        self._run()

    def sanity_checks(self):
        # check if the max_num_seq and max_num_batched_tokens amount to the free gpu_blocks
        expected_blocks = (
            self.scheduler_config.max_num_seqs * self.scheduler_config.max_num_batched_tokens
        ) // self.block_size
        warn = False
        if expected_blocks > self.total_gpu_blocks:
            self.log.warning(
                f"The max schedul-able {expected_blocks} GPU blocks, but only {self.total_gpu_blocks} block available"
            )
            self.log.warning(
                f"Consider decreasing max_num_seqs {self.scheduler_config.max_num_seqs} or max_num_batched_tokens {self.scheduler_config.max_num_batched_tokens}"
            )
            warn = True

        if not warn:
            self.log.info(
                f"Sanity check passed: expected {expected_blocks} blocks, got {self.total_gpu_blocks} available"
            )

    def _init_gpu_memory_blocks(self):
        """Initialize GPU memory block management with tensor parallelism support."""
        gpu_memory_gb = self.opalConfig["worker"]["hw"]["memory_gb"]
        tp_degree = self.opalConfig["worker"]["hw"].get("tp", 1)

        # With tensor parallelism, KV cache is distributed across all TP ranks
        # So effective memory for KV cache is tp_degree * single_gpu_memory
        total_gpu_memory_bytes = int(gpu_memory_gb * tp_degree * 1024**3)

        model_params = self.model_config.model_params
        model_size_bytes = model_params * 2

        free_memory_bytes = total_gpu_memory_bytes - model_size_bytes
        block_size_bytes = self.block_size * self.model_config.key_value_bytes

        self.total_gpu_blocks = int(free_memory_bytes // block_size_bytes)
        self.free_gpu_blocks = self.total_gpu_blocks
        self.init_free_blocks = self.free_gpu_blocks

        max_seq_blocks = (self.model_config.max_position_embeddings + self.block_size - 1) // self.block_size
        if self.total_gpu_blocks < max_seq_blocks:
            self.log.error(f"GPU memory is insufficient: {self.total_gpu_blocks} KVC blocks available ")
            self.log.error(
                f"TP={tp_degree}, per-GPU mem={gpu_memory_gb} GB, total mem ={gpu_memory_gb * tp_degree} GB, "
            )
            self.log.error(
                f"Free for KV: {free_memory_bytes//2**30} GB after loading the model of size {model_size_bytes / 2**30:.2f} GB"
            )
            self.log.error(
                f"But {max_seq_blocks} blocks needed for max_model_len={self.scheduler_config.max_model_len} tokens / {self.block_size} blocks"
            )
            self.opalEnv.stop_simulation()

        # what we allow is to fetch 1 MAX kvc worth of data at max
        self.max_kvc_fetch_blocks = max_seq_blocks

        self.log.info(
            f"GPU memory blocks initialized: "
            f"tp_degree={tp_degree}, "
            f"per_gpu_memory={gpu_memory_gb}GB, "
            f"total_memory={gpu_memory_gb * tp_degree}GB, "
            f"model_size={model_size_bytes / 1024**3:.2f}GB, "
            f"free_memory={free_memory_bytes / 1024**3:.2f}GB, "
            f"block_size={self.block_size} tokens, "
            f"block_size_bytes={block_size_bytes / 1024:.2f}KB, "
            f"total_blocks={self.total_gpu_blocks}"
        )

    def _init_scheduler_config(self) -> VLLMSchedulerConfig:
        """Initialize vLLM scheduler configuration from worker config."""
        vllm_params = self.opalConfig["worker"]["vllm_params"]

        max_num_seqs = vllm_params.get("max_num_seqs")
        max_num_batched_tokens = vllm_params.get("max_num_batched_tokens")
        chunked_prefill = vllm_params.get("chunked_prefill")
        max_model_len = self.model_config.max_position_embeddings

        gpu_memory_gb = self.opalConfig["worker"]["hw"].get("memory_gb")
        gpu_memory_kvcache_bytes = int(gpu_memory_gb * 1024**3)
        max_kvc_ready_requests = vllm_params["max_kvc_ready_requests"]
        lookahead_reqs = vllm_params.get("lookahead_reqs", 256)  # Default to 256 if not specified
        enable_gpu_apc = vllm_params.get("enable_gpu_apc", False)

        config = VLLMSchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            chunked_prefill=chunked_prefill,
            gpu_memory_kvcache_bytes=gpu_memory_kvcache_bytes,
            max_kvc_ready_requests=max_kvc_ready_requests,
            lookahead_reqs=lookahead_reqs,
            enable_gpu_apc=enable_gpu_apc,
        )
        return config

    def _apc_admit_tokens(self, request: "VLLMSchedulerRequest", tokens_to_add: int) -> Optional[int]:
        """GPU-block cost of advancing `request` by tokens_to_add, sharing-aware.
        Evicts idle blocks to cover a shortfall; returns the capacity delta
        applied (>= 0), or None (no side effects) if it can't fit even after
        evicting. Token-budget / max_num_seqs checks are the caller's."""
        if tokens_to_add == 0:
            return 0

        if not self.scheduler_config.enable_gpu_apc:
            # No APC: plain per-block allocation.
            current_tokens = request.prompt_processed + request.decode_tokens_generated
            new_tokens = current_tokens + tokens_to_add
            current_blocks = (current_tokens + self.block_size - 1) // self.block_size
            new_blocks = (new_tokens + self.block_size - 1) // self.block_size
            blocks_needed = new_blocks - current_blocks
            if blocks_needed > self.free_gpu_blocks:
                return None
            self.free_gpu_blocks -= blocks_needed
            request.allocated_blocks += blocks_needed
            return blocks_needed

        # APC: price it, evict if short, then commit.
        resolution, hash_ids_ref = self._apc.resolve(request, tokens_to_add)
        if resolution.capacity_delta > self.free_gpu_blocks:
            self._apc_evict_blocks(resolution.capacity_delta - self.free_gpu_blocks)
        if resolution.capacity_delta > self.free_gpu_blocks:
            return None  # still no room after evicting
        self._apc.commit(request, resolution, hash_ids_ref, tokens_to_add)
        self.free_gpu_blocks -= resolution.capacity_delta
        request.allocated_blocks += resolution.capacity_delta
        return resolution.capacity_delta

    def _apc_evict_blocks(self, blocks_needed: int) -> int:
        """Evict idle blocks until blocks_needed are freed (or none remain),
        return them to free_gpu_blocks, and write each evicted prefix through to
        the CPU/DFS tier. GpuApc picks the victims; capacity + tier I/O (which
        need worker/KVC state) stay here."""
        freed, store_sources = self._apc.evict(blocks_needed)
        self.free_gpu_blocks += freed
        kvc_chunk_size = self._kvc_manager.token_database.chunk_size
        pending_stores: dict[int, tuple] = {}  # id(hash_ids_ref) -> (hash_ids_ref, max_end_idx)
        for hash_ids_ref, end_idx in store_sources:
            if end_idx % kvc_chunk_size != 0:
                continue
            ref_id = id(hash_ids_ref)
            if ref_id not in pending_stores or end_idx > pending_stores[ref_id][1]:
                pending_stores[ref_id] = (hash_ids_ref, end_idx)
        for hash_ids_ref, end_idx in pending_stores.values():
            def _store(h_ids):
                yield from self._kvc_manager.store(h_ids)
            self.simpy_env.process(_store(hash_ids_ref[:end_idx]))
        return freed

    def _select_preemption_candidate(self, requests: List[VLLMSchedulerRequest]) -> Optional[VLLMSchedulerRequest]:
        """
        Select a request to preempt from the given list.

        Current strategy: Select request with least work done (lowest work_score).

        Future strategies to consider (add as needed):
        - Prefill requests before decode requests
        - Requests with less progress first
        - Requests with no KVC hit first
        - Largest memory consumers first
        - Requests with highest preemption count (to avoid starvation)

        Args:
            requests: List of requests to consider for preemption

        Returns:
            The selected request, or None if no suitable candidate found
        """
        if not requests:
            return None

        candidate = None
        min_work_score = float("inf")

        for req in requests:
            # Only consider requests in valid running phases
            if req.phase not in [RequestPhase.PREFILL_CHUNKED, RequestPhase.DECODE]:
                continue

            # Skip requests very close to completion (within 5% of decode tokens)
            if req.phase == RequestPhase.DECODE:
                remaining_decode = req.remaining_decode_tokens()
                if remaining_decode <= max(1, int(req.output_tokens * 0.05)):
                    continue

            # Skip requests that have been preempted too many times (max 3)
            if req.preemption_count >= 3:
                self.log.debug(
                    f"Skipping request {req.request_id} for preemption: already preempted {req.preemption_count} times"
                )
                continue

            # Calculate work score: lower = less work done
            if req.phase in [RequestPhase.PREFILL_CHUNKED]:
                work_score = req.prompt_processed
            else:  # DECODE
                work_score = req.prompt_tokens + req.decode_tokens_generated

            if work_score < min_work_score:
                min_work_score = work_score
                candidate = req

        return candidate

    def _preempt_request(self, blocks_needed: int) -> bool:
        """
        Preempt a running request to free GPU blocks.

        Selects the request with the least work done (fewest tokens generated).
        Avoids preempting requests that are very close to completion.

        Args:
            blocks_needed: Number of blocks needed to make progress

        Returns:
            True if a request was preempted and enough blocks freed, False otherwise
        """
        if not self.running_requests:
            return False

        # Select candidate for preemption
        candidate = self._select_preemption_candidate(self.running_requests)

        if candidate is None:
            self.log.debug("No suitable candidate found for preemption")
            return False

        # Preempt the candidate
        blocks_to_free = candidate.allocated_blocks

        if blocks_to_free == 0:
            self.log.warning(f"Request {candidate.request_id} selected for preemption but has 0 allocated blocks")
            return False

        # Free GPU blocks
        self.free_gpu_blocks += blocks_to_free
        candidate.allocated_blocks = 0

        # Store progress before reset for logging
        old_phase = candidate.phase
        old_prefill_progress = candidate.prompt_processed
        old_decode_progress = candidate.decode_tokens_generated

        # Calculate work score for logging
        if candidate.phase in [RequestPhase.PREFILL_CHUNKED]:
            work_score = candidate.prompt_processed
        else:  # DECODE
            work_score = candidate.prompt_tokens + candidate.decode_tokens_generated

        # Move request back to FRONT of waiting queue and reset to initial state
        self.running_requests.remove(candidate)
        candidate.phase = RequestPhase.WAITING

        # Reset all progress - request will go through KVC lookup again
        candidate.prompt_processed = 0
        candidate.decode_tokens_generated = 0

        # Increment preemption counter
        candidate.preemption_count += 1

        # Insert at front of waiting queue for priority
        self.waiting_requests.insert(0, candidate)

        self.log.warning(
            f"PREEMPTED request {candidate.request_id}: "
            f"freed {blocks_to_free} blocks (work_score={work_score:.0f}, preemption_count={candidate.preemption_count}), "
            f"free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}, "
            f"phase was {old_phase.value}, "
            f"old_progress: prefill={old_prefill_progress}/{candidate.prompt_tokens}, "
            f"decode={old_decode_progress}/{candidate.output_tokens}, "
            f"reset to WAITING with progress=0"
        )

        # Store the preempted request ID so we can remove it from batch later
        self._preempted_request_ids = getattr(self, "_preempted_request_ids", set())
        self._preempted_request_ids.add(candidate.request_id)

        return blocks_to_free >= blocks_needed

    def queue_work(self, request: LLMRequest) -> Generator[None, None, None]:
        """
        Queue a request for processing - entry point for new work.

        This is called by the router when assigning work to this worker.
        It implements the interrupt-driven wake-up mechanism:

        1. Put request in the worker_local_queue
        2. Interrupt _check_new_requests process to wake it up immediately

        Without the interrupt, _check_new_requests would sleep indefinitely.
        The interrupt wakes it up so it can process the new request immediately.

        This eliminates the need for polling with small timeouts.
        """
        yield self._worker_local_queue.put(request)

        # Wake up _check_new_requests only if it is currently in the idle sleep.
        # If intake is already active, the request is already in the queue and
        # will be picked up without an extra interrupt.
        if self._check_new_requests_idle:
            try:
                self._check_new_request_process.interrupt()
            except RuntimeError:
                # Process may have already finished or not yet started
                pass

    def _run(self):
        """Start worker processes."""
        # Start scheduling loop first so _scheduling_loop_process is available
        self._scheduling_loop_process = self.simpy_env.process(self._vllm_scheduling_loop())
        # Then start request checker which may interrupt the scheduler
        self._check_new_request_process = self.simpy_env.process(self._check_new_requests())
        self.simpy_env.process(self._periodic_kvc_updates())

    def __str__(self):
        return f"{__class__.__name__}.{self.id}"

    def append_kvc_events(self, e: list[KVCEvent]):
        """Append KVC events for periodic reporting."""
        if len(e) > 0:
            self._pending_kvc_events = self._pending_kvc_events + e

    def _periodic_kvc_updates(self):
        """Periodically send KVC events to router."""
        router = self.opalEnv.registry.get_router()
        while not self.opalEnv.are_we_done():
            yield self.simpy_env.timeout(self.kvcevent_coalesce_time)
            self.simpy_env.process(router.queue_events(self._pending_kvc_events))
            self._pending_kvc_events = []

    def _periodic_infra_updates(self):
        """Periodically send infrastructure updates to router."""
        router = self.opalEnv.registry.get_router()
        while not self.opalEnv.are_we_done():
            yield self.simpy_env.timeout(self.periodic_infra_update_time)
            total_requests = len(self.waiting_requests) + len(self.running_requests)
            queue_occupancy = len(self.waiting_requests) / max(1, self.scheduler_config.max_num_seqs)
            gpu_utilization = min(1.0, len(self.running_requests) / max(1, self.scheduler_config.max_num_seqs))

            sys_event = SystemEvent(
                worker_id=self.id,
                load=gpu_utilization,
                ingress_queue_occupancy=queue_occupancy,
                mem_used=0.66,
                gpu_utilization=gpu_utilization,
            )
            self.simpy_env.process(router.queue_events([sys_event]))

    def _vllm_scheduling_loop(self):
        """
        Main continuous scheduling loop - interrupt-driven architecture.

        This loop implements an event-driven scheduler that:
        1. Processes work when requests are available (waiting or running)
        2. Sleeps indefinitely when idle, waking only on new request arrivals

        Interrupt Handling Strategy:
        - When IDLE (no work): Interrupts wake up the scheduler to check for new work
        - When ACTIVE (processing): New arrivals are queued, but the scheduler is NOT
          interrupted; the current scheduler step must run to completion atomically

        Why avoid interrupts during active work?
        - _scheduler_step() and _build_batch() contain multiple yield points (async operations)
        - If interrupted mid-execution, they could process the same request twice
        - Example: _build_batch() moves requests from waiting_requests to running_requests
          If interrupted after the move but before completion, the next call would fail
          with "ValueError: list.remove(x): x not in list"
        - More critically, if an interrupt lands during
          `yield self.simpy_env.timeout(batch_time)` inside `_scheduler_step()`,
          the batch execution is aborted before request state is updated. This causes
          long-running requests to restart the same batch over and over without making
          progress.

        Implementation detail:
        - `_scheduler_busy` is True whenever the scheduler is active or about to process work
        - `_scheduler_busy` is set to False only when entering the indefinite idle sleep
        - `_check_new_requests()` interrupts the scheduler only when `_scheduler_busy` is False

        This means new requests are never lost:
        - If idle, an interrupt wakes the scheduler immediately
        - If active, the request is appended to `waiting_requests` and will be picked up
          on the next scheduler iteration after current work completes

        Design Choice: `yield from` vs `yield simpy.process()`
        -------------------------------------------------------
        Q: Why use `yield from self._scheduler_step()` instead of
           `yield self.simpy_env.process(self._scheduler_step())`?

        Both approaches wait for _scheduler_step() to complete (synchronous execution),
        but they differ critically in interrupt handling:

        Option 1: `yield from self._scheduler_step()` (CURRENT CHOICE)
        - Inline execution: Runs in the same process context
        - Interrupt propagation: Interrupts to parent affect the inline code
        - Control: We can catch and ignore interrupts during work
        - Overhead: Single process in SimPy's event queue

        Option 2: `yield self.simpy_env.process(self._scheduler_step())`
        - Separate process: Spawns independent child process
        - Interrupt isolation: Interrupts to parent DON'T affect child
        - Control: Cannot catch interrupts happening inside child
        - Overhead: Two processes in SimPy's event queue (parent + child)

        Example: Interrupt During Batch Creation
        -----------------------------------------
        Scenario: New request arrives while _build_batch() is executing

        With `yield from` (Current):
        Time 0.0: Start _build_batch()
        Time 0.5: New request arrives → Interrupt sent
        Time 0.5: Interrupt caught by try-except → Ignored
        Time 1.0: _build_batch() completes atomically
        Time 1.0: Loop continues → Processes new request
        ✅ Correct: Batch creation completes without interruption

        With `yield simpy.process()`:
        Time 0.0: Spawn _build_batch() as child process
        Time 0.5: New request arrives → Interrupt sent to parent
        Time 0.5: Parent catches interrupt → Moves to next iteration
        Time 0.5: Parent starts NEW _build_batch() (child still running!)
        Time 1.0: First _build_batch() completes
        Time 1.5: Second _build_batch() completes
        ❌ Wrong: Two batches being built simultaneously, race condition!

        Conclusion: `yield from` is essential because:
        1. Allows us to catch and ignore interrupts during critical work
        2. Prevents race conditions from concurrent batch creation
        3. Simpler code with lower SimPy overhead
        4. Maintains atomic execution of scheduler operations
        """
        self.log.debug(f"{self} starting vLLM scheduling loop")
        self._sched_step_count = 0

        while not self.opalEnv.are_we_done():
            # self.log.debug(f"{self} scheduling loop iteration {self._sched_step_count}")
            self._sched_step_count += 1

            if self.waiting_requests or self.running_requests:
                # ACTIVE: Process work atomically without interruption.
                # The scheduler is considered busy whenever it is not in the
                # indefinite idle sleep below.
                self._scheduler_busy = True
                try:
                    yield from self._scheduler_step()
                except simpy.Interrupt:
                    # This should now be rare/unexpected because new work should
                    # only interrupt the scheduler while it is idle.
                    self.log.debug(
                        f"Scheduler interrupted during work - unexpected while busy: "
                        f"{len(self.waiting_requests)} waiting, {len(self.running_requests)} running, "
                        f"step={self._sched_step_count}"
                    )
                    continue
            else:
                # IDLE: Sleep indefinitely until interrupted by new request arrival.
                #
                # `_scheduler_busy` must be False ONLY for the exact lifetime of
                # this blocking sleep. Use finally to guarantee the flag is reset
                # even if the sleep is exited via interrupt.
                self._scheduler_busy = False
                try:
                    self.log.debug(f"Scheduler no work, sleeping for long")
                    yield self.simpy_env.timeout(float("inf"))
                except simpy.Interrupt:
                    # Woken up by new request arrival, continue to check queues
                    self.log.debug(f"Scheduler loop interrupted - checking for work")
                    continue
                finally:
                    self._scheduler_busy = True

        self.log.debug(f"{self} scheduling loop completed")

    def _check_new_requests(self):
        """
        Check for new requests from _worker_local_queue and add to waiting queue.

        This process implements the request intake side of the interrupt-driven architecture:
        1. Polls the worker_local_queue for new requests
        2. When a request arrives, adds it to waiting_requests
        3. Interrupts the scheduler to wake it up (if it's sleeping)
        4. When queue is empty, sleeps indefinitely until woken by queue_work()

        Interrupt Flow:
        - queue_work() puts request in queue → interrupts this process
        - This process wakes up → gets request → adds to waiting_requests
        - This process interrupts scheduler → scheduler wakes up and processes request

        This eliminates the old polling pattern where we'd check every 0.001s.
        """
        while not self.opalEnv.are_we_done():
            if len(self._worker_local_queue.items) > 0:
                # Get request from queue
                request: LLMRequest = yield self._worker_local_queue.get()
                request.stats.set_worker_arrival_time(self.simpy_env.now)
                sched_request: VLLMSchedulerRequest = VLLMSchedulerRequest(request)
                sched_request.phase = RequestPhase.WAITING

                # Add to waiting queue
                self.waiting_requests.append(sched_request)
                self.log.debug(f"New request {request.id} added to waiting queue (phase: WAITING)")

                # Wake up scheduler only if it is currently in the idle sleep.
                # If the scheduler is busy, the new request stays queued and
                # will be picked up on the next scheduler iteration.
                if not self._scheduler_busy:
                    try:
                        self._scheduling_loop_process.interrupt()
                    except RuntimeError:
                        # Scheduler may have finished or not started yet
                        continue
            else:
                # Queue is empty - sleep indefinitely until woken by queue_work().
                #
                # `_check_new_requests_idle` must be True ONLY while this exact
                # blocking sleep is active. Use finally to guarantee the flag is
                # reset immediately after wakeup, including the interrupt path.
                self._check_new_requests_idle = True
                try:
                    yield self.simpy_env.timeout(float("inf"))
                except simpy.Interrupt:
                    # Woken up by queue_work(), continue to check queue
                    self.log.debug("_check_new_requests interrupted by new work arrival")
                    continue
                finally:
                    self._check_new_requests_idle = False

    def _scheduler_step(self):
        """
        Execute one scheduler step: run current batch, advance states, refill on completions.

        1. Build batch if empty (from waiting queue)
        2. Execute batch (yield for batch_time)
        3. Update request states (prefill→decode, decode→completed)
        4. Move completed requests out, store KVC, free blocks
        5. Rebuild batch to fill vacated slots
        """
        step_start_time = self.simpy_env.now
        # self.log.debug(f"ENTRY: __scheduler_step, self.current_batch None? {self.current_batch is None}, ")
        # Step 1: Create initial batch if needed
        if self.current_batch is None or self.current_batch.is_empty():
            self.current_batch = yield from self._build_batch()

            if self.current_batch.is_empty():
                # Nothing to schedule - yield to allow async operations to complete
                yield self.simpy_env.timeout(0)
                # self.log.debug("No requests to schedule, yielding control")
                return

        # Log batch info -- enable the whole block for detailed debuggin
        # atr: do not remove this block
        # batch_info = []
        # for r in self.current_batch.prefill_requests:
        #     req = cast(VLLMSchedulerRequest, r)
        #     batch_info.append(
        #         f"req_{req.request_id}(state:{req.phase.value}, "
        #         f"prefill:{req.prompt_processed}/{req.prompt_tokens}, "
        #         f"decode:{req.decode_tokens_generated}/{req.output_tokens})"
        #     )
        # for r in self.current_batch.decode_requests:
        #     req = cast(VLLMSchedulerRequest, r)
        #     batch_info.append(
        #         f"req_{req.request_id}(state:{req.phase.value}, "
        #         f"prefill:{req.prompt_processed}/{req.prompt_tokens}, "
        #         f"decode:{req.decode_tokens_generated}/{req.output_tokens})"
        #     )
        # waiting_ready_ids = [r.request_id for r in self.waiting_requests if r.phase == RequestPhase.READY]
        # self.log.debug(
        #     f"Scheduler step_{self._sched_step_count}: "
        #     f"batch=[{', '.join(batch_info)}], "
        #     f"tokens={self.current_batch.total_tokens}/{self.scheduler_config.max_num_batched_tokens}, "
        #     f"free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}, "
        #     f"waiting_ready={len(waiting_ready_ids)} {waiting_ready_ids}"
        # )

        # Step 2: Execute batch
        batch_time = self._calculate_batch_time(self.current_batch)
        # self.log.debug(f"Batch execution time: {batch_time:.4f}s")

        yield self.simpy_env.timeout(batch_time)
        self.gpu_busy_time += batch_time

        # Step 3: Update request states
        self._update_request_states_and_stats(self.current_batch, batch_time)

        # Step 4: Check for completed requests
        completed_count = len([r for r in self.running_requests if r.phase == RequestPhase.COMPLETED])
        self._move_completed_requests()

        # Step 5: Handle completed requests (free memory, store KVC)
        if completed_count > 0:
            yield safe_process(self.simpy_env, self._handle_completed_requests())

            # Step 6: Try to add new requests to fill the slots
            yield safe_process(self.simpy_env, self._build_batch(self.current_batch))

    def _can_issue_kvc_fetch(self, kvc_blocks: int) -> bool:
        """
        Check if a KVC fetch can be issued for a waiting request.

        The idea of such rate-limiting is to avoid these requests hogging the GPU
        memory, thus leading the preemptions for running requests.

        Before I have a design with both control: the number of request and the total
        outstanding blocks that are committed to such fetching. The problem is that unless
        the outstanding block is actually equal to the "max_model_size" (in the worst case),
        it would lead livelock for the such long prefill requests. These requests will
        never be scheduled. For example, think of a prefill hit of 100% cache hit with 128K
        prompt.
          * An idea here could have been - to have a threshold above which such request will
            be compute scheduled. But that is odd to the overall KVCache size concept as
            it makes the most sense when we fetch it for long requests.

        So, given that the worst-case max single request fetching is nonetheless is the max
        prompt size, why not just control the number of such requests that can be pending.

        A downside of such design is consider an optimal case where there are sequence of
        very long prompts with high hit rates: (128K, 100% hit) * N. In this case we will
        run scheduling logic with only "N" sequence at a time. While optimally we could have
        scheduled the max prompts we can contain in the GPU memory.

        The notion of how many KVCache requests can be there is to be verified with vLLM
        (for example with slow storage). Does excessive KVCache fetching leads to excessive
        preemptions for the running request?


        Two conditions must be met:
        1. The number of KVC-ready requests (FETCH_KVC or READY with KVC) in waiting queue
           must not exceed max_kvc_ready_requests
        2. [DISABLED] The total blocks committed to KVC fetching for waiting requests only
           (in-flight or completed) must not exceed max_kvc_fetch_blocks

        Note: This limit applies ONLY to waiting requests, not running requests.

        Args:
            kvc_blocks: Number of blocks needed for the new KVC fetch

        Returns:
            True if the KVC fetch can be issued, False otherwise
        """
        # Count KVC-ready requests in waiting queue (FETCH_KVC or READY with allocated blocks from KVC)
        kvc_ready_count = 0
        waiting_kvc_blocks = 0

        for req in self.waiting_requests:
            if req.phase == RequestPhase.FETCH_KVC:
                kvc_ready_count += 1
                waiting_kvc_blocks += req.allocated_blocks
            elif req.phase == RequestPhase.READY and req.allocated_blocks > 0:
                # READY request with allocated blocks means it had a KVC hit
                kvc_ready_count += 1
                waiting_kvc_blocks += req.allocated_blocks

        # Check condition 1: max number of KVC-ready requests
        if kvc_ready_count >= self.scheduler_config.max_kvc_ready_requests:
            # kvc_pending_reqs = self.__get_kvc_ready_requests()
            # self.log.debug(
            #     f"KVC fetch blocked: {kvc_ready_count} KVC-ready requests in waiting queue "
            #     f">= max_kvc_ready_requests={self.scheduler_config.max_kvc_ready_requests}, current KVC pending requests: {kvc_pending_reqs}"
            # )
            return False

        # atr - see the comment above why this block is commented out.
        # # Check condition 2: max blocks committed to KVC fetching for waiting requests
        # if waiting_kvc_blocks + kvc_blocks > self.max_kvc_fetch_blocks:
        #     self.log.debug(
        #         f"KVC fetch blocked: waiting_kvc_blocks={waiting_kvc_blocks} + requested={kvc_blocks} "
        #         f"> max_kvc_fetch_blocks={self.max_kvc_fetch_blocks}"
        #     )
        #     return False

        return True

    def _build_batch(self, batch=None):
        """
        Build or rebuild a batch with correct memory accounting.

        Phases:
        1. Resume existing decode requests (highest priority, 1 token each)
        2. Resume existing prefill requests (chunked)
        3. Preempt any existing request that can no longer fit
        4. Add new READY requests from waiting queue (with lookahead limit)

        Args:
            batch: Existing batch to rebuild, or None to create new batch

        Returns:
            BatchMetadata: The batch (new or rebuilt)
        """
        start = time.perf_counter()
        req_touched = 0

        # Yield to allow other coroutines to run
        yield self.simpy_env.timeout(0.001)

        # Create new batch or reset existing batch
        is_new_batch = batch is None
        if is_new_batch:
            batch = BatchMetadata()
            self.log.debug(
                f"Creating initial batch: waiting={len(self.waiting_requests)}, running={len(self.running_requests)}"
            )
        else:
            # Reset token counts for existing batch - we'll recalculate
            # atr - token budget is a proxy for the compute effort
            batch: BatchMetadata = batch
            batch.total_tokens = 0
            batch.prefill_tokens = 0
            batch.decode_tokens = 0
            batch.request_tokens.clear()
            self.log.debug(
                f"Rebuilding existing batch: prefill={len(batch.prefill_requests)}, decode={len(batch.decode_requests)}, "
                f"waiting={len(self.waiting_requests)}, running={len(self.running_requests)}"
            )

        assert self.check_invariants()

        # Phase 1: Resume existing decode requests (highest priority)
        requests_to_preempt = []
        for req in batch.decode_requests[:]:  # Iterate over copy
            req = cast(VLLMSchedulerRequest, req)
            req_touched += 1

            # Decode always generates 1 token
            tokens_to_add = 1
            token_budget_ok = batch.total_tokens + tokens_to_add <= self.scheduler_config.max_num_batched_tokens
            delta = self._apc_admit_tokens(req, tokens_to_add) if token_budget_ok else None

            if delta is not None:
                batch.decode_tokens += tokens_to_add
                batch.total_tokens += tokens_to_add
                batch.request_tokens[req.request_id] = tokens_to_add
            
            else:
                # Cannot fit - mark for preemption
                requests_to_preempt.append(req)

        # Phase 2: Resume existing prefill requests
        for req in batch.prefill_requests[:]:  # Iterate over copy
            req = cast(VLLMSchedulerRequest, req)
            req_touched += 1

            # Calculate tokens to add for this prefill chunk
            ideal_chunk = schedule_prefill_chunk(req, self.scheduler_config)
            remaining_budget = self.scheduler_config.max_num_batched_tokens - batch.total_tokens
            tokens_to_add = min(ideal_chunk, remaining_budget)

            if tokens_to_add <= 0:
                # No token budget left - mark for preemption
                requests_to_preempt.append(req)
                self.log.debug(f"No token budget for prefill request {req.request_id}")
                continue

            delta = self._apc_admit_tokens(req, tokens_to_add)

            # Check if we can fit this request
            if delta is not None:
                batch.prefill_tokens += tokens_to_add
                batch.total_tokens += tokens_to_add
                batch.request_tokens[req.request_id] = tokens_to_add

            else:
                # Cannot fit - mark for preemption
                requests_to_preempt.append(req)
           

        # Phase 3: Preempt requests that cannot fit
        for req in requests_to_preempt:
            req = cast(VLLMSchedulerRequest, req)
            req_touched += 1

            # Remove from batch
            if req in batch.prefill_requests:
                batch.prefill_requests.remove(req)
            if req in batch.decode_requests:
                batch.decode_requests.remove(req)

            # Free GPU blocks
            if req.allocated_blocks > 0:
                self.free_gpu_blocks += req.allocated_blocks
                blocks_freed = req.allocated_blocks
                req.allocated_blocks = 0
            else:
                blocks_freed = 0

            # Store progress for logging
            old_phase = req.phase
            old_prefill_progress = req.prompt_processed
            old_decode_progress = req.decode_tokens_generated

            # Calculate work score for logging
            if req.phase in [RequestPhase.PREFILL_CHUNKED]:
                work_score = req.prompt_processed
            else:  # DECODE
                work_score = req.prompt_tokens + req.decode_tokens_generated

            # Move to front of waiting queue and reset
            self.running_requests.remove(req)
            req.phase = RequestPhase.WAITING
            req.prompt_processed = 0
            req.decode_tokens_generated = 0
            req.preemption_count += 1

            # Insert at front for priority
            self.waiting_requests.insert(0, req)

            self.log.warning(
                f"PREEMPTED request {req.request_id} during batch rebuild: "
                f"freed {blocks_freed} blocks (work_score={work_score:.0f}, preemption_count={req.preemption_count}), "
                f"free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}, "
                f"phase was {old_phase.value}, "
                f"old_progress: prefill={old_prefill_progress}/{req.prompt_tokens}, "
                f"decode={old_decode_progress}/{req.output_tokens}"
            )

        # Phase 4: Only add new waiting requests if we have capacity
        # Process waiting queue for new requests with lookahead limit
        lookahead_reqs = self.scheduler_config.lookahead_reqs  # Get from config
        requests_scanned = 0
        initial_batch_size = batch.num_sequences()
        requests_added_in_phase4 = 0

        # Debug: Count requests by phase
        phase_counts = {"WAITING": 0, "FETCH_KVC": 0, "READY": 0, "OTHER": 0}
        for r in self.waiting_requests:
            req_phase = cast(VLLMSchedulerRequest, r).phase
            if req_phase == RequestPhase.WAITING:
                phase_counts["WAITING"] += 1
            elif req_phase == RequestPhase.FETCH_KVC:
                phase_counts["FETCH_KVC"] += 1
            elif req_phase == RequestPhase.READY:
                phase_counts["READY"] += 1
            else:
                phase_counts["OTHER"] += 1

        self.log.debug(
            f"Phase 4 start: batch_size={initial_batch_size}, waiting_queue={len(self.waiting_requests)}, "
            f"phases={phase_counts}, free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}, "
            f"kvc_fetch_in_flight_blocks={self.kvc_fetch_blocks_in_flight}"
        )

        for request in self.waiting_requests[:]:
            req = cast(VLLMSchedulerRequest, request)
            req_touched += 1
            requests_scanned += 1

            # Stop scanning if we've reached lookahead limit
            # Only apply limit if batch started non-empty (regardless of how many we added)
            if initial_batch_size > 0 and requests_scanned > lookahead_reqs:
                self.log.warning(
                    f"Reached lookahead limit ({lookahead_reqs}), stopping waiting queue scan. "
                    f"Batch started with {initial_batch_size} sequences, now has {batch.num_sequences()}, "
                    f"added {requests_added_in_phase4} in this phase, {len(self.waiting_requests) - requests_scanned} requests not scanned."
                )
                break

            # Sanity check: COMPLETED requests should never be in waiting queue
            if req.phase == RequestPhase.COMPLETED:
                raise AssertionError(
                    f"BUG: Request {req.request_id} is COMPLETED but still in waiting_requests. "
                    f"This indicates a state management bug."
                )

            # Handle WAITING phase: perform KVC lookup
            if req.phase == RequestPhase.WAITING:
                req.llm_request.stats._3_start_processing_time = self.simpy_env.now

                # Step 1: GPU APC check (no yield — fresh on every scheduling step
                # so we always see the current APC state).
                apc_prefix = 0
                if self.scheduler_config.enable_gpu_apc:
                    apc_prefix = self._apc.lookup(req.hash_ids, claim=False)
                    self.log.info(
                        f"[KVC Lookup] [APC] | Request {req.request_id} | "
                        f"Found {apc_prefix}/{req.prompt_tokens} prefix tokens in GPU APC"
                    )

                # Step 2: tiered-storage lookup (cached after first call — may yield).
                # Always computed regardless of the APC result above, so that APC's
                # zero-I/O advantage can still be compared against the full prefix
                # length available in CPU/DFS tiers.
                if not hasattr(req, "_kvc_lookup_done"):
                    tiered_prefix = yield safe_process(
                        self.simpy_env, self._kvc_manager.lookup(tokens=req.hash_ids)
                    )
                    self.log.info(
                        f"[KVC Lookup] [Tiered] | Request {req.request_id} | "
                        f"Found {tiered_prefix}/{req.prompt_tokens} prefix tokens in tiered storage"
                    )
                    req._kvc_lookup_done = True
                    req._kvc_prefix_tokens = tiered_prefix
                else:
                    tiered_prefix = req._kvc_prefix_tokens

                # Step 3: use whichever source gives the longer prefix.
                # APC is preferred when tied because it requires no I/O.
                if apc_prefix >= tiered_prefix and apc_prefix > 0:
                    # APC wins: attach (share) the matched blocks and mark READY
                    # immediately (no I/O). Non-destructive -- these blocks stay
                    # resident/discoverable for any other concurrent request, and
                    # since they're shared, this request pays no GPU-block cost
                    # for the matched portion (only the simulated prefill compute
                    # for those tokens is skipped, which is the actual benefit).
                    apc_hit = self._apc.lookup(req.hash_ids, claim=True, request=req)
                    req.allocated_blocks = 0
                    req.prompt_processed = apc_hit
                    req.apc_resolved_tokens = apc_hit  # already hashed/attached -- nothing private here
                    req.phase = RequestPhase.READY
                    req.llm_request.stats._4_request_ready_time = self.simpy_env.now
                    req.llm_request.stats.kvc_hit_type = "apc"
                    req.llm_request.stats.set_prefix_hit_tokens(apc_hit)
                    req.llm_request.stats.set_kvc_hit_tokens_per_tier("HBM_APC", apc_hit)

                elif tiered_prefix > 0:
                    # Tiered storage wins: initiate async fetch from CPU/DFS.
                    num_prefix_tokens = tiered_prefix
                    kvc_blocks = (num_prefix_tokens + self.block_size - 1) // self.block_size

                    if not self._can_issue_kvc_fetch(kvc_blocks):
                        continue

                    if kvc_blocks > self.free_gpu_blocks:
                        idle_blocks = self._apc.evictable_count() if self.scheduler_config.enable_gpu_apc else 0
                        if kvc_blocks <= self.free_gpu_blocks + idle_blocks:
                            self._apc_evict_blocks(kvc_blocks - self.free_gpu_blocks)
                        else:
                            continue

                    self.free_gpu_blocks -= kvc_blocks
                    req.allocated_blocks = kvc_blocks
                    self.kvc_fetch_blocks_in_flight += kvc_blocks

                    if self.free_gpu_blocks < 0:
                        self.log.error(f"CRITICAL: free_gpu_blocks went negative!")
                        self.free_gpu_blocks += kvc_blocks
                        req.allocated_blocks = 0
                        self.kvc_fetch_blocks_in_flight -= kvc_blocks
                        assert False, "Free GPU blocks should never be negative"

                    req.phase = RequestPhase.FETCH_KVC
                    req.kvc_fetch_in_progress = True
                    self.simpy_env.process(self._async_kvc_retrieve(req, num_prefix_tokens))

                else:
                    # No prefix found in any tier — mark READY for full prefill.
                    req.phase = RequestPhase.READY
                    req.llm_request.stats._4_request_ready_time = self.simpy_env.now
                    req.llm_request.stats.kvc_hit_type = "none"

                # If we just marked it READY, fall through to process it
                if req.phase != RequestPhase.READY:
                    continue

            # Skip requests in FETCH_KVC phase
            if req.phase == RequestPhase.FETCH_KVC:
                continue

            # Only process READY requests for scheduling
            if req.phase != RequestPhase.READY:
                continue

            # Check batch limits
            if batch.num_sequences() >= self.scheduler_config.max_num_seqs:
                break

            # Handle 100% KVC hit case (prefill already complete)
            if req.is_prefill_complete():
                blocks_needed = self._calculate_blocks_needed(req, 0)

                if self._can_add_to_batch(batch, req, 0, blocks_needed):
                    if blocks_needed > 0:
                        if self.free_gpu_blocks >= blocks_needed:
                            req.allocated_blocks += blocks_needed
                            self.free_gpu_blocks -= blocks_needed
                        else:
                            assert False, "Not enough free blocks, how did this happen?"

                    batch.prefill_requests.append(req)
                    batch.request_tokens[req.request_id] = 0

                    # Move to running queue
                    self.waiting_requests.remove(req)
                    self.running_requests.append(req)
                    req.llm_request.stats._5_gpu_start_time = self.simpy_env.now
                    requests_added_in_phase4 += 1

                    self.log.debug(
                        f"Added READY request {req.request_id} with 100% KVC hit: "
                        f"0 prefill tokens, will transition to DECODE"
                    )
                    continue
                else:
                    self.log.debug(f"Cannot add 100% KVC hit request {req.request_id}")
                    break

            # Normal prefill case

            full_prompt_blocks = (req.prompt_tokens + self.block_size - 1) // self.block_size
            resolved_blocks = req.apc_resolved_tokens // self.block_size
            remaining_prompt_blocks = max(0, full_prompt_blocks - resolved_blocks)
            if remaining_prompt_blocks > self.free_gpu_blocks and self.scheduler_config.enable_gpu_apc:
                self._apc_evict_blocks(remaining_prompt_blocks - self.free_gpu_blocks)
            if remaining_prompt_blocks > self.free_gpu_blocks:
                break

            remaining_budget = self.scheduler_config.max_num_batched_tokens - batch.total_tokens
            ideal_chunk = schedule_prefill_chunk(req, self.scheduler_config)
            tokens_to_add = min(ideal_chunk, remaining_budget)

            if tokens_to_add <= 0:
                break

            blocks_needed = self._calculate_blocks_needed(req, tokens_to_add)

            if not self._can_add_to_batch(batch, req, tokens_to_add, blocks_needed):
                self.log.debug(
                    f"Cannot add request {req.request_id}: "
                    f"blocks_needed={blocks_needed}, free_blocks={self.free_gpu_blocks}"
                )
                break

            delta = self._apc_admit_tokens(req, tokens_to_add)
            if delta is None:
                break

            batch.prefill_requests.append(req)
            batch.prefill_tokens += tokens_to_add
            batch.total_tokens += tokens_to_add
            batch.request_tokens[req.request_id] = tokens_to_add

            # Move to running queue
            self.waiting_requests.remove(req)
            self.running_requests.append(req)
            req.llm_request.stats._5_gpu_start_time = self.simpy_env.now
            requests_added_in_phase4 += 1

            self.log.debug(
                f"Added READY request {req.request_id}: tokens={tokens_to_add}, "
                f"blocks_needed={blocks_needed}, free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}"
            )

        # Phase 4 summary logging
        end = time.perf_counter()
        self.log.debug(
            f"Phase 4 end: scanned={requests_scanned}, added={requests_added_in_phase4}, "
            f"final_batch_size={batch.num_sequences()}, phases_after={phase_counts}"
        )
        self.log.debug(
            f"Batch building time is {end - start:.2f} seconds, req touched {req_touched} || {self._get_queue_status()}"
        )

        return batch

    def _get_queue_status(self):

        return json.dumps(
            {
                "waiting": len(self.waiting_requests),
                "running": len(self.running_requests),
            }
        )

    def __get_kvc_ready_requests(self):
        kvc_pending_reqs = []
        for req in self.waiting_requests:
            if req.phase == RequestPhase.FETCH_KVC or (req.phase == RequestPhase.READY and req.allocated_blocks > 0):
                kvc_pending_reqs.append({"id": req.llm_request.id, "kvc_blocks": req.allocated_blocks})
        return kvc_pending_reqs

    def print_scheduler_state(self):
        """Print debug information about the current state."""
        self.log.debug("=========================")
        kvc_pending_reqs = self.__get_kvc_ready_requests()
        ready_count = sum([1 if r.phase == RequestPhase.READY else 0 for r in self.waiting_requests])
        self.log.debug(
            f"Queue status: waiting={len(self.waiting_requests)} (Ready: {ready_count}), running={len(self.running_requests)}, completed = {len(self.completed_requests)}, kvc_pending_reqs = {kvc_pending_reqs}"
        )

        self.log.debug(
            f"Scheduler step_{self._sched_step_count}: "
            f"tokens={self.current_batch.total_tokens}/{self.scheduler_config.max_num_batched_tokens}, "
            f"free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}, "
            f"waiting={len(self.waiting_requests)}, running = {len(self.running_requests)},"
            f"kvc_fetch_blocks_in_flight: {self.kvc_fetch_blocks_in_flight}"
        )
        active_used = sum([r.allocated_blocks for r in (self.running_requests + self.completed_requests)])
        # because a waiting request might have kvc committed
        kvc_committed = sum([r.allocated_blocks for r in (self.waiting_requests)])
        total_used = active_used + kvc_committed + self.kvc_fetch_blocks_in_flight
        self.log.debug(
            f"active_used: {active_used}, kvc_committed: {kvc_committed}, kvc_fetch_blocks_in_flight: {self.kvc_fetch_blocks_in_flight}, total: {total_used}"
        )
        self.log.debug("=========================")

    def check_invariants(self):
        """
        In normal operation this should be no-op.
        This for now is for detailed debugging.
        """
        enable_variant_checks = False
        if enable_variant_checks:
            self.memory_invariant_check()
            self.batch_token_invariant_check()
        return True

    def batch_token_invariant_check(self) -> bool:
        if self.current_batch is not None:
            if self.current_batch.is_empty():
                if self.current_batch.total_tokens != 0:
                    self.log.debug(
                        f"Current token budget {self.current_batch.total_tokens} is not empty for an empty batch"
                    )
                    assert False

            # sum should add up
            sum_tokens = sum(list(self.current_batch.request_tokens.values()))
            if sum_tokens != self.current_batch.total_tokens:
                self.log.debug(
                    f"Current token budget {self.current_batch.total_tokens} is not equal to the sum of tokens in the batch {sum_tokens}"
                )
                assert False

        return True

    def memory_invariant_check(self) -> bool:
        # if nothing is running create a new batch
        ready_count = sum([1 if r.phase == RequestPhase.READY else 0 for r in self.waiting_requests])
        self.log.debug(
            f"Queue status: waiting={len(self.waiting_requests)} (Ready: {ready_count}), running={len(self.running_requests)}, completed = {len(self.completed_requests)}"
        )

        if self.current_batch is None:
            ret_val = self.free_gpu_blocks == self.init_free_blocks
        else:
            if not self.scheduler_config.enable_gpu_apc:
                # now we have a dynamic setting
                active_used = sum([r.allocated_blocks for r in (self.running_requests + self.completed_requests)])
                # because a waiting request might have kvc committed
                kvc_committed = sum([r.allocated_blocks for r in (self.waiting_requests)])
                total_used = active_used + kvc_committed
                self.log.debug(
                    f"active_used: {active_used}, kvc_committed: {kvc_committed}, kvc_fetch_blocks_in_flight: {self.kvc_fetch_blocks_in_flight}, total_used: {total_used}, gpu_free: {self.free_gpu_blocks} | condition = {self.init_free_blocks == self.free_gpu_blocks + total_used}"
                )

                ret_val = self.init_free_blocks == self.free_gpu_blocks + total_used
            else:
                # Sharing-aware invariant
                ret_val = self.init_free_blocks == self.free_gpu_blocks + self._apc.resident + private_total
                self.log.debug(
                    f"apc_resident: {self._apc.resident}, apc_private: {private_total}, "
                    f"gpu_free: {self.free_gpu_blocks} | condition = {ret_val}"
                )
                if not ret_val:
                    self.log.debug(
                        f"APC invariant violated: free_gpu_blocks({self.free_gpu_blocks}) + "
                        f"len(policy)({self._apc.resident}) + private({private_total}) != "
                        f"init_free_blocks({self.init_free_blocks})"
                    )

        if ret_val == False:
            # We will abort, print the current state for debugging
            self.print_scheduler_state()
            self.opalEnv.stop_simulation()

        return ret_val

    def _calculate_blocks_needed(self, request: VLLMSchedulerRequest, tokens_to_add: int) -> int:
        """
        Calculate number of GPU memory blocks needed for this request.

        For chunked prefill, only allocate blocks for the tokens being processed in this step.
        Blocks grow incrementally as the request progresses through prefill and decode.
        """
        # Calculate blocks needed for current + new tokens
        current_tokens = request.prompt_processed + request.decode_tokens_generated
        new_tokens = current_tokens + tokens_to_add
        current_blocks = (current_tokens + self.block_size - 1) // self.block_size
        new_blocks = (new_tokens + self.block_size - 1) // self.block_size
        assert new_blocks >= current_blocks, f"new_blocks {new_blocks} < current_blocks {current_blocks}"
        return new_blocks - current_blocks

    def _can_add_to_batch(
        self, batch: BatchMetadata, request: VLLMSchedulerRequest, tokens_to_add: int, blocks_needed: int
    ) -> bool:
        """Check if request can be added to batch given all constraints."""
        if batch.num_sequences() >= self.scheduler_config.max_num_seqs:
            return False

        if batch.total_tokens + tokens_to_add > self.scheduler_config.max_num_batched_tokens:
            return False

        # STRICT CHECK: Ensure we have enough blocks
        if blocks_needed > self.free_gpu_blocks:
            return False

        # SANITY CHECK: Ensure free_blocks is non-negative
        if self.free_gpu_blocks < 0:
            self.log.error(
                f"CRITICAL: free_gpu_blocks is negative ({self.free_gpu_blocks})! "
                f"This indicates a memory accounting bug."
            )
            assert False, "Negative free_gpu_blocks detected - memory accounting error"

        return True

    def _async_kvc_retrieve(self, req: VLLMSchedulerRequest, num_prefix_tokens: int):
        """
        Async coroutine to retrieve KV cache data for a request.
        This only handles the data transfer part and runs without blocking the scheduler loop.

        GPU memory has already been accounted for before this is called.
        """
        # self.log.debug(f"Starting async KVC retrieve for request {req.request_id}")

        # Retrieve the KV cache (this yields and simulates the move time)
        moved_tensors = yield safe_process(self.simpy_env, self._kvc_manager.retrieve(req.hash_ids, num_prefix_tokens))
        """
        Original:     [T, T, T, F, F, F]
        Inverted:     [F, F, F, T, T, T]
        Cumsum:       [0, 0, 0, 1, 2, 3]
        Equals 0:     [T, T, T, F, F, F]
        Sum:          3 ← This is our answer!
        """
        actual_retrieved_tokens = int(((~moved_tensors).cumsum(axis=0) == 0).sum())
        self.log.debug(f"Fetched KVC tokens {actual_retrieved_tokens} for Request {req.request_id}")

        # the number of fetched KVC token are essentially prompt length that is now processed
        req.prompt_processed = actual_retrieved_tokens

        # Recalculate blocks based on actual retrieved tokens
        actual_kvc_blocks = (actual_retrieved_tokens + self.block_size - 1) // self.block_size
        estimated_kvc_blocks = req.allocated_blocks

        # atr: with num_prefix_tokens pass in the retrieved tokens function we are asking to restrict the fetching of
        # prefix cache to max num_prefix_tokens. This is a safety check to ensure that we are not fetching more than
        # the free memory we have. We were constantly running in the issue where at the time of fetching the prefix
        # we got something like 1 token, but then when we actually retrieve() kvc we got 512 ...this leads to this memory
        # race condition. This is a safety check to ensure that we are not fetching more than the free memory we have.
        # This would probably lead to a performance hit, but I do not know how to reconcile the issue of this.
        # The core issue being
        # lookup <-------- more tokens got generated -------> retrieve()
        # between these more tokens could have been generated. How to fix it
        assert actual_kvc_blocks == estimated_kvc_blocks

        # Adjust GPU blocks if actual differs from estimated
        if actual_kvc_blocks != estimated_kvc_blocks:
            block_diff = actual_kvc_blocks - estimated_kvc_blocks

            # STRICT CHECK: Ensure we have enough blocks for adjustment
            if block_diff > 0 and block_diff > self.free_gpu_blocks:
                # this can happen in between if someone has generated these new additional prefixes.
                # we must fetch for what we have committed memory for! TODO(atr) -- FIXME!
                # For now, lets trap this condition and we will fix later.
                self.log.error(
                    f"CRITICAL: Request {req.request_id} needs {block_diff} more blocks "
                    f"but only {self.free_gpu_blocks} available during KVC adjustment!"
                )
                self.opalEnv.stop_simulation()

            self.free_gpu_blocks -= block_diff
            req.allocated_blocks = actual_kvc_blocks

            # Also adjust in-flight counter
            self.kvc_fetch_blocks_in_flight += block_diff

            self.log.debug(
                f"Request {req.request_id}: adjusted GPU blocks from {estimated_kvc_blocks} to {actual_kvc_blocks} "
                f"(free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks})"
            )

        # Decrement in-flight KVC fetch counter
        self.kvc_fetch_blocks_in_flight -= actual_kvc_blocks

        # SANITY CHECK: Ensure in-flight counter doesn't go negative
        if self.kvc_fetch_blocks_in_flight < 0:
            self.log.error(
                f"CRITICAL: kvc_fetch_blocks_in_flight went negative ({self.kvc_fetch_blocks_in_flight})! "
                f"This indicates a tracking bug."
            )
            assert False  # This should never happen
            self.kvc_fetch_blocks_in_flight = 0

        # Transition to READY state
        req.kvc_fetch_in_progress = False
        req.llm_request.stats.set_prefix_hit_tokens(actual_retrieved_tokens)

        req.phase = RequestPhase.READY
        req.llm_request.stats._4_request_ready_time = self.simpy_env.now
        req.llm_request.stats.__kvc_hit_tokens = actual_retrieved_tokens

        self.log.debug(
            f"Request {req.request_id}: KVC retrieve complete, "
            f"retrieved {actual_retrieved_tokens}/{num_prefix_tokens} tokens "
            f"({actual_retrieved_tokens * 100 / num_prefix_tokens:.1f}%), "
            f"allocated {actual_kvc_blocks} blocks, now READY, "
            f"kvc_in_flight={self.kvc_fetch_blocks_in_flight}/{self.max_kvc_fetch_blocks}"
        )

    def _calculate_batch_time(self, batch: BatchMetadata):
        """Calculate execution time: max(prefill_latency, batched_decode_latency)."""
        max_prefill_time = 0.0
        max_decode_time = 0.0

        for request in batch.prefill_requests:
            tokens_to_process = batch.request_tokens.get(request.request_id, 0)

            # the total size we have to target to process is so far + new, while pretending that the so_far is in the cache
            prefill_time = self.gpu_model.get_prefill_latency(
                request.prompt_processed + tokens_to_process, request.prompt_processed
            )
            max_prefill_time = max(max_prefill_time, prefill_time)

        decode_batch = []
        # batch decode
        for request in batch.decode_requests:
            # For running decode requests, always generate 1 token per step
            # For newly transitioned requests, this is already set in _update_request_states_and_stats
            assert (request.request_id in batch.request_tokens) and (batch.request_tokens.get(request.request_id) == 1)
            decode_batch.append((request.prompt_tokens + request.decode_tokens_generated))

        decode_time = self.gpu_model.get_decode_latency_batch(decode_batch)

        batch_time = max(max_prefill_time, decode_time)
        return batch_time

    def _update_request_states_and_stats(self, batch: BatchMetadata, batch_time: float):
        """Advance request phases after batch execution and move prefill→decode within the batch."""
        current_time = self.simpy_env.now

        # Track requests that transition from prefill to decode
        prefill_to_decode = []

        for req in batch.prefill_requests:
            request = cast(VLLMSchedulerRequest, req)
            request.llm_request.stats.add_scheduler_timestamp(current_time)

            tokens_processed = batch.request_tokens.get(request.request_id)

            # Update phase transition
            if request.phase == RequestPhase.READY:
                request.phase = RequestPhase.PREFILL_CHUNKED
                self.log.debug(
                    f"Request {request.request_id}: transitioned READY -> "
                    f"{'PREFILL_CHUNKED' if self.scheduler_config.chunked_prefill else 'PREFILL'}"
                )

            request.prompt_processed += tokens_processed

            if request.is_prefill_complete():
                request.llm_request.stats.mark_prefill_done()
                request.phase = RequestPhase.DECODE
                prefill_to_decode.append(request)
                self.log.debug(f"Request {request.request_id}: prefill complete, transitioning to DECODE")

        # process all the decode states to see if any of them transition to finish
        for req in batch.decode_requests:
            request = cast(VLLMSchedulerRequest, req)
            request.llm_request.stats.add_scheduler_timestamp(current_time)
            # this entry __MUST__ exist as "1", no need to offer a default as "1"
            tokens_generated = batch.request_tokens.get(request.request_id)
            request.decode_tokens_generated += tokens_generated

            if request.is_decode_complete():
                request.phase = RequestPhase.COMPLETED
                self.log.debug(
                    f"Request {request.request_id}: decode complete, transitioning to COMPLETED "
                    f"(blocks will be freed after KVC store)"
                )

        # Move requests that completed prefill from prefill_requests to decode_requests in batch
        # atr - This should be done at the end, not to interfere with the loop above, which may pick up the
        # newly transitions request to the decode.
        for request in prefill_to_decode:
            batch.prefill_requests.remove(request)
            batch.decode_requests.append(request)
            # Update token accounting for next iteration (decode generates 1 token at a time)
            batch.request_tokens[request.request_id] = 1

    def _move_completed_requests(self):
        """
        Move completed requests from running to completed queue.
        Also removes them from the current batch.
        """
        newly_completed = [req for req in self.running_requests if req.phase == RequestPhase.COMPLETED]

        for req in newly_completed:
            self.running_requests.remove(req)
            self.completed_requests.append(req)

            # Remove from current batch
            if self.current_batch:
                if req in self.current_batch.prefill_requests:
                    # self.current_batch.prefill_requests.remove(req)
                    assert False, "This should not happen that a request is moved directly from prefill to complete"
                if req in self.current_batch.decode_requests:
                    self.current_batch.decode_requests.remove(req)

                # Remove from token tracking and update total_tokens
                if req.request_id in self.current_batch.request_tokens:
                    tokens_to_remove = self.current_batch.request_tokens[req.request_id]
                    self.current_batch.total_tokens -= tokens_to_remove
                    del self.current_batch.request_tokens[req.request_id]
                    self.log.debug(
                        f"Request {req.request_id}: removed {tokens_to_remove} tokens from batch, "
                        f"new total_tokens={self.current_batch.total_tokens}"
                    )

            self.log.debug(f"Request {req.request_id}: moved from running to completed queue and removed from batch")
            req.llm_request.stats._7_decode_done_time = self.simpy_env.now

    def _handle_completed_requests(self):
        """Handle completed requests: release GPU blocks, store KVC, and send to output queue."""
        # FIX: Race condition prevention for concurrent executions
        # This method is called via safe_process() which allows multiple concurrent instances.
        # Previously, we iterated over completed_requests[:] (a copy) but removed from the
        # original list during iteration. This caused "list.remove(x): x not in list" errors
        # when multiple instances tried to remove the same request.
        #
        # Solution: Move all completed requests to a local list and clear the shared list
        # atomically BEFORE any yield. This ensures each concurrent execution gets its own
        # exclusive set of requests to handle, preventing race conditions.
        # CRITICAL: Must grab the list before any yield to prevent race conditions!
        requests_to_handle = self.completed_requests[:]
        self.completed_requests.clear()

        # Small delay to allow other processes to run
        yield self.simpy_env.timeout(0.001)

        for request in requests_to_handle:
            self.log.debug(f"Storing KVC for request {request.request_id} " f"({len(request.hash_ids)} tokens)")
            # Store KVC data (this can take time but doesn't need GPU memory)
            yield safe_process(self.simpy_env, self._kvc_manager.store(request.hash_ids))

            if request.allocated_blocks > 0:
                self.free_gpu_blocks += request.allocated_blocks
                self.log.debug(
                    f"Request {request.request_id}: released {request.allocated_blocks} blocks after saving KVC store, "
                    f"free_blocks={self.free_gpu_blocks}/{self.total_gpu_blocks}"
                )
                request.allocated_blocks = 0

            yield self.router_output_finished_req_queue.put(request.llm_request)

            self.log.info(
                f"Request {request.request_id} retired on worker.{self.id}: "
                f"prefill={request.prompt_tokens} tokens, "
                f"decode={request.output_tokens} tokens, (prefix hit = {(request.llm_request.stats.get_prefix_hit_tokens() * 100 / request.prompt_tokens):.2f}%, sched steps = {request.llm_request.stats.get_scheduler_steps()})"
            )


# Made with Bob
