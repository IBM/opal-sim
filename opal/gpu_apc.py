# SPDX-License-Identifier: Apache-2.0
"""GPU Automatic Prefix Cache (APC) collaborator.

`GpuApc` encapsulates all GPU-APC *cache mechanics* for the vLLM worker:
the eviction policy, the block-granularity token database, the block->source
map, prefix lookups, and the sharing-aware admission/settle/release/evict
operations.

Deliberately knows NOTHING about worker scheduling state. In particular it
never touches `free_gpu_blocks` or the KVC tier manager -- capacity accounting
and tier write-through stay in the worker, which orchestrates:

    resolution, ref = apc.resolve(request, tokens_to_add)   # think (read-only)
    free_gpu_blocks += apc.evict(delta - free)[0]           # make room
    apc.commit(request, resolution, ref, tokens_to_add)     # act
    free_gpu_blocks -= resolution.capacity_delta

So every method here is pure with respect to worker capacity: mutating methods
return the deltas the worker must apply, rather than reaching back into it.
"""
from __future__ import annotations

import logging

from opal.apc_policy import (
    BaseAPCPolicy,
    BlockResolution,
    commit_apc_blocks,
    resolve_apc_blocks,
)

from opal.kvc_manager import OpalTokenDatabase

log = logging.getLogger("GpuApc")


class GpuApc:
    """Owns the GPU-APC policy + token db and the operations over them.

    Args:
        policy: the eviction policy (built via apc_policy.make_apc_policy).
        token_db: block-granularity OpalTokenDatabase (chunk_size == block_size,
            partial last block excluded).
        block_size: tokens per GPU block.
        
            
    """

    def __init__(
        self,
        policy: BaseAPCPolicy,
        token_db: "OpalTokenDatabase",
        block_size: int,
    ) -> None:
        self.policy = policy
        self.token_db = token_db
        self.block_size = block_size
        
        # block_hash -> (hash_ids ref, end_token_idx). Saved at insert so the
        # worker can write evicted prefixes through to a lower tier at the KVC
        # manager's chunk granularity. Populated by commit_apc_blocks.
        self.block_source: dict[int, tuple[list, int]] = {}

    # ── introspection (for the worker's logs / invariant checks) ──────────────

    @property
    def resident(self) -> int:
        """Number of blocks currently registered (== physical blocks held)."""
        return len(self.policy)

    def evictable_count(self) -> int:
        """Registered blocks with no live referrer (idle, reclaimable now)."""
        return self.policy.evictable_count()

    # ── hashing ───────────────────────────────────────────────────────────────

    def hash_ids_for(self, request, total_tokens_needed: int) -> list:
        """Token id sequence (prompt + decode-so-far) covering at least
        total_tokens_needed tokens, for incremental APC hashing. Decode tokens
        aren't appended to request.hash_ids until retirement, so this stitches
        in the ground-truth output tokens on demand.

        Cached and extended incrementally on the request (never rebuilt from
        scratch) -- called once per admission step, so a naive concatenation
        here would itself be O(total tokens) every decode step.
        """
        if total_tokens_needed <= request.prompt_tokens:
            return request.hash_ids
        if request.apc_hash_ids_cache is None:
            request.apc_hash_ids_cache = list(request.hash_ids)
        cache = request.apc_hash_ids_cache
        if len(cache) < total_tokens_needed:
            output_ids = request.llm_request.output_token_ids or []
            decode_needed = total_tokens_needed - request.prompt_tokens
            cache.extend(output_ids[len(cache) - request.prompt_tokens : decode_needed])
        return cache

    # ── lookup / claim ────────────────────────────────────────────────────────

    def lookup(self, hash_ids: list, claim: bool = False, request=None) -> int:
        """Longest prefix token count found consecutively in the GPU APC table.

        claim=True non-destructively increfs each matched block on `request`'s
        behalf (shared attach -- the block stays resident/discoverable for other
        concurrent requests) and records it in request.apc_owned_hashes for
        later decref at retirement. Returns matched prefix token count (0 miss).
        """
        assert not claim or request is not None, "claim=True requires `request`"
        matched_tokens = 0
        chain_hash = None
        for _start, end, block_hash in self.token_db.process_tokens(hash_ids):
            if block_hash not in self.policy:
                break
            if claim:
                self.policy.incref(block_hash)
                request.apc_owned_hashes.add(block_hash)
            else:
                self.policy.touch(block_hash)
            matched_tokens = end
            chain_hash = block_hash
        if claim:
            # Seed the prefix-hash chain so the NEXT incremental admission call
            # resumes hashing from here and produces the same hash a continuous
            # chain from token 0 would.
            request.apc_chain_hash = chain_hash
        return matched_tokens

    # ── admission (pure w.r.t. capacity) ──────────────────────────────────────

    def resolve(self, request, tokens_to_add: int) -> tuple[BlockResolution, list]:
        """Read-only: cost of advancing `request` by tokens_to_add, sharing-aware.

        Inspects the policy (membership only) but never mutates it. Returns the
        BlockResolution and the hash_ids ref it was computed against, which the
        matching commit() must be given.
        """
        current_tokens = request.apc_resolved_tokens
        hash_ids_ref = self.hash_ids_for(request, current_tokens + tokens_to_add)
        resolution = resolve_apc_blocks(
            self.token_db, self.policy, self.block_size,
            hash_ids_ref, current_tokens, tokens_to_add,
            chain_hash=request.apc_chain_hash,
        )
        return resolution, hash_ids_ref

    def commit(self, request, resolution: BlockResolution, hash_ids_ref: list, tokens_to_add: int) -> None:
        """Register/attach the blocks from a resolution and advance request state.

        Call exactly once per resolve() the worker decided to commit (after it
        has confirmed/evicted enough physical capacity).
        """
        commit_apc_blocks(self.policy, resolution, self.block_source, hash_ids_ref)
        for block_hash, _end in resolution.new_block_hashes:
            request.apc_owned_hashes.add(block_hash)
        for block_hash in resolution.attach_hashes:
            request.apc_owned_hashes.add(block_hash)
        request.apc_resolved_tokens += tokens_to_add
        request.apc_private_blocks += resolution.private_delta
        request.apc_chain_hash = resolution.chain_hash

    def settle(self, request, full_blocks: int) -> int:
        """Register `full_blocks` of already-held private content (e.g. just
        fetched from a lower tier, where no APC hashing happened) into the table
        at no new capacity cost. Advances apc_resolved_tokens by
        full_blocks*block_size. Returns blocks that turned out to duplicate an
        already-cached block (their private copy can be freed by the worker).
        """
        if full_blocks == 0:
            return 0
        current_tokens = request.apc_resolved_tokens
        assert current_tokens % self.block_size == 0
        settle_tokens = full_blocks * self.block_size
        hash_ids_ref = self.hash_ids_for(request, current_tokens + settle_tokens)
        resolution = resolve_apc_blocks(
            self.token_db, self.policy, self.block_size,
            hash_ids_ref, current_tokens, settle_tokens,
            chain_hash=request.apc_chain_hash,
        )
        assert not resolution.reserved_partial  # settle ranges are always block-aligned
        commit_apc_blocks(self.policy, resolution, self.block_source, hash_ids_ref)
        for block_hash, _end in resolution.new_block_hashes:
            request.apc_owned_hashes.add(block_hash)
        for block_hash in resolution.attach_hashes:
            request.apc_owned_hashes.add(block_hash)
        request.apc_resolved_tokens += settle_tokens
        request.apc_chain_hash = resolution.chain_hash
        return len(resolution.attach_hashes)

    # ── release ───────────────────────────────────────────────────────────────

    def release(self, request) -> int:
        """Release all of `request`'s claims (retirement/preemption): decref
        every block it depends on (they stay resident, reclaimed later by
        eviction) and reset its APC state. Returns the request's private,
        never-registered blocks, which the worker frees immediately.
        """
        for block_hash in request.apc_owned_hashes:
            self.policy.decref(block_hash)
        request.apc_owned_hashes.clear()
        freed = request.apc_private_blocks
        request.apc_private_blocks = 0
        request.apc_resolved_tokens = 0
        request.apc_chain_hash = None
        return freed

    # ── eviction (pure w.r.t. capacity and tiers) ─────────────────────────────

    
    def evict(self, blocks_needed: int) -> tuple[int, list[tuple[list, int]]]:
        """Evict idle blocks until `blocks_needed` are freed or none remain."""
        freed = 0
        store_sources: list[tuple[list, int]] = []
        for block_hash, _end in self.policy.evict(blocks_needed):
            source = self.block_source.pop(block_hash, None)
            if source is not None:
                store_sources.append(source)
            freed += 1
        return freed, store_sources


