# SPDX-License-Identifier: Apache-2.0
"""Pluggable eviction policies for the GPU Automatic Prefix Cache (APC).

Each policy implements BaseAPCPolicy. The worker holds one policy instance
(_apc_policy) and calls insert/touch/evict through it.

Block accounting (_apc_reserved_blocks, free_gpu_blocks) lives in the worker;
policies only manage the *identity* of which block hashes are cached and which
to evict next — they do not touch counters.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from collections import OrderedDict
from typing import Callable, Optional
from opal.kvc_manager import OpalTokenDatabase

class BaseAPCPolicy(abc.ABC):
    """Interface for GPU APC eviction policies.
    """

    def __init__(self) -> None:
        self.ref_counts: dict[int, int] = {}
        self.pinned_count: int = 0

    def incref(self, block_hash: int) -> int:
        """Add a reference (new owner or an attaching/claiming request). Returns new count."""
        old = self.ref_counts.get(block_hash, 0)
        n = old + 1
        self.ref_counts[block_hash] = n
        if old == 0:
            self.pinned_count += 1
            self.on_pin(block_hash)
        return n

    def decref(self, block_hash: int) -> int:
        """Remove a reference (an owner is done with this block). Returns new count.
        Never goes negative; decref on an untracked/already-zero hash is a no-op
        floor at 0 (defensive -- callers should not decref more than they incref'd,
        but this avoids corrupting the table on a bug elsewhere).
        """
        old = self.ref_counts.get(block_hash, 0)
        n = max(0, old - 1)
        self.ref_counts[block_hash] = n
        if old == 1:
            self.pinned_count -= 1
            self.on_unpin(block_hash)
        return n

    def on_pin(self, block_hash: int) -> None:
        """Hook: a block just became referenced (ref_count 0 -> 1).
        Policies that keep pinned blocks out of their eviction ordering (e.g. LRU)
        override this to remove the block from that structure. Default: no-op.
        """

    def on_unpin(self, block_hash: int) -> None:
        """Hook: a block just became idle (ref_count 1 -> 0).
        Counterpart to on_pin -- policies re-admit the block to their eviction
        ordering here. Not called on eviction/forced-remove (see forget_refcount),
        since the block is leaving the table entirely in those cases. Default: no-op.
        """

    def evictable_count(self) -> int:
        """How many resident blocks currently have no live referrer (ref_count == 0)."""
        return len(self) - self.pinned_count

    def forget_refcount(self, block_hash: int) -> None:
        """Drop ref-count bookkeeping for a hash that's leaving the table entirely
        (eviction or forced remove). Internal helper for subclasses."""
        old = self.ref_counts.pop(block_hash, 0)
        if old > 0:
            self.pinned_count -= 1

    @abc.abstractmethod
    def insert(self, block_hash: int, end_token_idx: int) -> None:
        """Add or refresh a cached block's identity. Acts as upsert. Does NOT
        change ref count -- callers must also call incref() for the owner."""


    @abc.abstractmethod
    def touch(self, block_hash: int) -> None:
        """Update recency / frequency on a read hit (lookup without claim)."""
        

    @abc.abstractmethod
    def evict(self, count: int) -> list[tuple[int, int]]:
        """Remove up to `count` eviction victims and return them in eviction order.

        Frees many blocks in a single request so the per-eviction bookkeeping
        (and the scan over the policy's eviction ordering) is amortized into one
        pass. Must only select idle blocks (ref_count == 0). May return fewer than
        `count` entries if the policy runs out of evictable blocks.
        """
        
    @abc.abstractmethod
    def __contains__(self, block_hash: int) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...


    @abc.abstractmethod
    def all_hashes(self):
        """Iterate over all currently-resident block hashes."""


# ─────────────────────────────────────────────────────────────────────────────
# LRU
# ─────────────────────────────────────────────────────────────────────────────

class LRUPolicy(BaseAPCPolicy):
    """Least-Recently-Used eviction with pinned blocks kept out of the scan.

    Two structures are maintained:

      * ``table``     -- every resident block (hash -> end_token_idx). Backs
                          discoverability (__contains__/all_hashes/len);
                          pinned and idle blocks alike stay here so prefix-cache
                          lookups still hit blocks that are currently in use.
      * ``evictable`` -- ONLY idle (ref_count == 0) blocks, ordered front = LRU.
                         This is the eviction queue we actually scan.

    Because pinned blocks are removed from ``evictable`` on pin (on_pin) and
    re-added on release (on_unpin), eviction never walks past in-use blocks:
    evict() is O(number of victims) rather than O(total resident blocks). A block
    released back to idle is treated as most-recently-used (appended to the MRU
    end), matching how a paged KV cache returns freed blocks to its free list.

    When ttl > 0, a block is immune to eviction for ttl seconds after its last
    access; evict() skips (but leaves in place) any idle block still inside its
    window, and returns fewer victims if none are yet eligible.
    """

    def __init__(self, ttl: float = 0.0, clock: Callable[[], float] = lambda: 0.0) -> None:
        super().__init__()
        self.table: dict[int, int] = {}                     # all resident blocks
        self.evictable: OrderedDict[int, None] = OrderedDict()  # idle blocks, front = LRU
        self._last_access: dict[int, float] = {}
        self._ttl = ttl
        self._clock = clock

    def insert(self, block_hash: int, end_token_idx: int) -> None:
        self.table[block_hash] = end_token_idx
        self._last_access[block_hash] = self._clock()
        if self.ref_counts.get(block_hash, 0) == 0:
            self.evictable[block_hash] = None
            self.evictable.move_to_end(block_hash)

    def touch(self, block_hash: int) -> None:
        self._last_access[block_hash] = self._clock()
        if block_hash in self.evictable:  # pinned blocks aren't in the queue
            self.evictable.move_to_end(block_hash)

    def on_pin(self, block_hash: int) -> None:
        # Block is now in use -- take it out of the eviction ordering.
        self.evictable.pop(block_hash, None)

    def on_unpin(self, block_hash: int) -> None:
        # Block is idle again -- re-admit as most-recently-used (if still resident).
        if block_hash in self.table:
            self.evictable[block_hash] = None
            self.evictable.move_to_end(block_hash)

    def evict(self, count: int) -> list[tuple[int, int]]:
        victims: list[tuple[int, int]] = []
        if count <= 0 or not self.evictable:
            return victims
        now = self._clock()
        ttl = self._ttl
        scanned = 0        # how many queue entries the pass actually touched
        ttl_skipped = 0    # idle-but-still-within-TTL entries we stepped over
        # Single pass from the LRU front. Only idle blocks live here, so we never
        # skip pinned entries -- at most we skip blocks still inside their TTL window.
        for block_hash in list(self.evictable):  # front = LRU
            scanned += 1
            if ttl > 0.0 and now < self._last_access[block_hash] + ttl:
                ttl_skipped += 1
                continue
            del self.evictable[block_hash]
            end_token_idx = self.table.pop(block_hash)
            del self._last_access[block_hash]
            self.forget_refcount(block_hash)
            victims.append((block_hash, end_token_idx))
            if len(victims) >= count:
                break
        return victims

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self.table

    def __len__(self) -> int:
        return len(self.table)

    def all_hashes(self):
        return iter(self.table)

# ─────────────────────────────────────────────────────────────────────────────
# Block resolution (ref-counted sharing across concurrent owners)
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BlockResolution:
    """Result of resolving how many *physical* GPU blocks a request needs for
    its next chunk of tokens, after accounting for content-identical blocks
    already resident (owned by this or another request) that can be shared
    instead of freshly allocated.
    """
    capacity_delta: int                      # net free_gpu_blocks change required (negative => blocks freed)
    new_block_hashes: list[tuple[int, int]]  # (hash, end_token_idx) needing insert+incref once committed
    attach_hashes: list[int]                 # hashes of already-cached blocks needing incref once committed
    reserved_partial: bool                   # True if a new private (not-yet-full) block slot is needed
    private_delta: int                       # net change in caller's *private* (unregistered) block count
    chain_hash: Optional[int]                # prefix-hash chain value as of current_tokens + tokens_to_add;
                                              # caller must save this and pass it back as chain_hash next call
                                              # (avoids re-deriving the whole chain from token 0 every time)


def resolve_apc_blocks(
    apc_token_db: "OpalTokenDatabase",
    apc_policy: BaseAPCPolicy,
    block_size: int,
    hash_ids_ref: list,
    current_tokens: int,
    tokens_to_add: int,
    chain_hash: Optional[int] = None,
) -> BlockResolution:
  
    new_tokens = current_tokens + tokens_to_add
    full_blocks_before = current_tokens // block_size
    full_blocks_after = new_tokens // block_size
    had_partial_before = (current_tokens % block_size) != 0
    has_partial_after = (new_tokens % block_size) != 0
    newly_full_count = full_blocks_after - full_blocks_before

    new_block_hashes: list[tuple[int, int]] = []
    attach_hashes: list[int] = []
    # Did the *former* private partial-tail block (if any) turn out to match
    # an already-cached block (deduped -> its private slot is freed), or did
    # it become a brand-new registry entry (promoted -> no NEW capacity, it
    # was already privately allocated)?
    former_partial_deduped = False
    former_partial_promoted = False
    out_chain_hash = chain_hash

    if newly_full_count > 0:
        token_limit = full_blocks_after * block_size
        start_idx = full_blocks_before * block_size
        for start, end, block_hash in apc_token_db.process_tokens(
            hash_ids_ref, start_idx=start_idx, prefix_hash=chain_hash, end_idx=token_limit
        ):
            out_chain_hash = block_hash
            block_idx = start // block_size
            is_former_partial = had_partial_before and block_idx == full_blocks_before
            if block_hash in apc_policy:
                attach_hashes.append(block_hash)
                if is_former_partial:
                    former_partial_deduped = True
            else:
                new_block_hashes.append((block_hash, end))
                if is_former_partial:
                    former_partial_promoted = True

    fresh_needed = len(new_block_hashes)
    capacity_delta = fresh_needed
    if former_partial_promoted:
        capacity_delta -= 1  # already privately allocated -- registering it costs nothing new
    if former_partial_deduped:
        capacity_delta -= 1  # private slot no longer needed at all -- freed back

    # A NEW partial-tail reservation is needed only if we now have a partial
    # tail we don't already hold a private slot for: either we had none
    # before, or the one we had just got resolved (completed) this step and
    # we've moved on to a fresh trailing block.
    still_same_partial_block = (
        had_partial_before and has_partial_after and full_blocks_before == full_blocks_after
    )
    reserved_partial = has_partial_after and not still_same_partial_block
    if reserved_partial:
        capacity_delta += 1

    private_delta = 0
    if reserved_partial:
        private_delta += 1
    if former_partial_promoted or former_partial_deduped:
        private_delta -= 1

    return BlockResolution(
        capacity_delta=capacity_delta,
        new_block_hashes=new_block_hashes,
        attach_hashes=attach_hashes,
        reserved_partial=reserved_partial,
        private_delta=private_delta,
        chain_hash=out_chain_hash,
    )


def commit_apc_blocks(
    apc_policy: BaseAPCPolicy,
    resolution: BlockResolution,
    apc_block_source: dict,
    hash_ids_ref: list,
) -> None:
    """Actually register/attach the blocks from a BlockResolution. Call exactly
    once per resolve_apc_blocks() call the caller decided to commit (i.e.
    after confirming/evicting for resolution.capacity_delta physical blocks).
    """
    for block_hash, end_idx in resolution.new_block_hashes:
        if block_hash not in apc_policy:
            apc_policy.insert(block_hash, end_idx)
            apc_block_source.setdefault(block_hash, (hash_ids_ref, end_idx))
        apc_policy.incref(block_hash)
    for block_hash in resolution.attach_hashes:
        apc_policy.incref(block_hash)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_apc_policy(
    name: str,
    ttl: float = 0.0,
    clock: Callable[[], float] = lambda: 0.0,
) -> BaseAPCPolicy:
    """Instantiate an APC eviction policy by name.

    Args:
        name: One of "lru", "lfu", "random", "continuum", "oracle", "prefix_oracle".
              Currently only "lru"
        ttl: Seconds a block is immune to eviction after its last access (LRU only).
        clock: Callable returning current simulation time in seconds (LRU).
    """
    match name:
        case "lru":
            return LRUPolicy(ttl=ttl, clock=clock)
        case _:
            raise ValueError(
                f"Unknown APC eviction policy: {name!r}. "
            )
