# SPDX-License-Identifier: Apache-2.0
"""Eviction policy for the GPU Automatic Prefix Cache (APC).

BaseAPCPolicy defines the interface; LRUPolicy implements it. The worker holds
one policy instance (_apc_policy) and calls insert/touch/incref/decref/evict
through it.

Block accounting (_apc_reserved_blocks, free_gpu_blocks) lives in the worker;
the policy only manages the *identity* of which block hashes are cached and
which to evict next — it does not touch counters.
"""
from __future__ import annotations

import abc
import dataclasses
import logging
from collections import OrderedDict
from typing import Callable, Optional

log = logging.getLogger("EvictionPolicy")

from opal.kvcache.kvc_manager import OpalTokenDatabase


class BaseAPCPolicy(abc.ABC):
    """Interface for GPU APC eviction policies.

    Each cached entry maps a block_hash (int) to end_token_idx (int). The worker
    calls:
      - insert : when a block is first hashed (becomes cache-discoverable)
      - incref : when a request starts using a block (fresh insert, or another
                 request attaching to / claiming an already-cached block)
      - decref : when a request is done with a block (retirement); the block
                 stays cached (still discoverable) but becomes evict-eligible
                 once its ref count reaches 0
      - touch  : on a read-only APC lookup hit (updates recency / frequency,
                 does not change ref count)
      - evict  : when the worker needs to free blocks (skips ref_count > 0)

    Ref counting lives here (not in the worker) since "is this block evictable"
    is intrinsic to cache identity, same as recency/frequency.
    """

    def __init__(self) -> None:
        # Subclasses must call super().__init__().
        # Ref counts for in-flight ownership; block_hash -> live referrer count.
        self._ref_counts: dict[int, int] = {}
        # Count of hashes with ref_count > 0, so evictable_count() is O(1) as
        # len(self) - _pinned_count. Kept in sync by incref/decref/_forget_refcount.
        self._pinned_count: int = 0

    def incref(self, block_hash: int) -> int:
        """Add a reference (new owner or an attaching/claiming request). Returns new count."""
        old = self._ref_counts.get(block_hash, 0)
        n = old + 1
        self._ref_counts[block_hash] = n
        if old == 0:
            self._pinned_count += 1
            self._on_pin(block_hash)
        return n

    def decref(self, block_hash: int) -> int:
        """Remove a reference (an owner is done with this block). Returns new count.

        Never goes negative; decref on an untracked/already-zero hash is a no-op
        floor at 0 (defensive -- callers should not decref more than they incref'd,
        but this avoids corrupting the table on a bug elsewhere).
        """
        old = self._ref_counts.get(block_hash, 0)
        n = max(0, old - 1)
        self._ref_counts[block_hash] = n
        if old == 1:
            self._pinned_count -= 1
            self._on_unpin(block_hash)
        return n

    def _on_pin(self, block_hash: int) -> None:
        """Hook: a block just became referenced (ref_count 0 -> 1).

        Policies that keep pinned blocks out of their eviction ordering (e.g. LRU)
        override this to remove the block from that structure. Default: no-op.
        """

    def _on_unpin(self, block_hash: int) -> None:
        """Hook: a block just became idle (ref_count 1 -> 0).

        Counterpart to _on_pin -- policies re-admit the block to their eviction
        ordering here. Not called on eviction (see _forget_refcount), since the
        block is leaving the table entirely in that case. Default: no-op.
        """

    def ref_count(self, block_hash: int) -> int:
        return self._ref_counts.get(block_hash, 0)

    def evictable_count(self) -> int:
        """How many resident blocks currently have no live referrer (ref_count == 0)."""
        return len(self) - self._pinned_count

    def _forget_refcount(self, block_hash: int) -> None:
        """Drop ref-count bookkeeping for a hash that's leaving the table entirely
        (eviction). Internal helper for subclasses."""
        old = self._ref_counts.pop(block_hash, 0)
        if old > 0:
            self._pinned_count -= 1

    @abc.abstractmethod
    def insert(self, block_hash: int, end_token_idx: int) -> None:
        """Add or refresh a cached block's identity. Acts as upsert. Does NOT
        change ref count -- callers must also call incref() for the owner."""
        ...

    @abc.abstractmethod
    def touch(self, block_hash: int) -> None:
        """Update recency / frequency on a read hit (lookup without claim)."""
        ...

    @abc.abstractmethod
    def evict(self, count: int) -> list[tuple[int, int]]:
        """Remove up to `count` eviction victims (ref_count == 0 only) and return
        them as (block_hash, end_token_idx) in eviction order. May return fewer
        than `count` if the policy runs out of evictable blocks."""
        ...

    @abc.abstractmethod
    def __contains__(self, block_hash: int) -> bool: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...


# ─────────────────────────────────────────────────────────────────────────────
# LRU
# ─────────────────────────────────────────────────────────────────────────────

class LRUPolicy(BaseAPCPolicy):
    """Least-Recently-Used eviction with pinned blocks kept out of the scan.

    Two structures are maintained:

      * ``_table``     -- every resident block (hash -> end_token_idx). Backs
                          discoverability (__contains__/len);
                          pinned and idle blocks alike stay here so prefix-cache
                          lookups still hit blocks that are currently in use.
      * ``_evictable`` -- ONLY idle (ref_count == 0) blocks, ordered front = LRU,
                          back = MRU. This is the eviction queue we actually scan.

    Because pinned blocks are removed from ``_evictable`` on pin (_on_pin) and
    re-added on release (_on_unpin), eviction never walks past in-use blocks:
    evict() is O(number of victims) rather than O(total resident blocks). A block
    released back to idle is treated as most-recently-used (appended to the MRU
    end), matching how a paged KV cache returns freed blocks to its free list.

    When ttl > 0, a block is immune to eviction for ttl seconds after its last
    access; evict() skips (but leaves in place) any idle block still inside its
    window, and returns fewer victims if none are yet eligible.
    """

    def __init__(self, ttl: float = 0.0, clock: Callable[[], float] = lambda: 0.0) -> None:
        super().__init__()
        self._table: dict[int, int] = {}                     # all resident blocks
        self._evictable: OrderedDict[int, None] = OrderedDict()  # idle blocks, front = LRU
        self._last_access: dict[int, float] = {}
        self._ttl = ttl
        self._clock = clock

    def _on_pin(self, block_hash: int) -> None:
        # Block is now in use -- take it out of the eviction ordering.
        self._evictable.pop(block_hash, None)

    def _on_unpin(self, block_hash: int) -> None:
        # Block is idle again -- re-admit as most-recently-used (if still resident).
        if block_hash in self._table:
            self._evictable[block_hash] = None
            self._evictable.move_to_end(block_hash)

    def insert(self, block_hash: int, end_token_idx: int) -> None:
        self._table[block_hash] = end_token_idx
        self._last_access[block_hash] = self._clock()
        # A freshly-hashed block is idle until the owner increfs it; keep the
        # eviction queue in sync only while the block is actually idle.
        if self.ref_count(block_hash) == 0:
            self._evictable[block_hash] = None
            self._evictable.move_to_end(block_hash)

    def touch(self, block_hash: int) -> None:
        self._last_access[block_hash] = self._clock()
        if block_hash in self._evictable:  # pinned blocks aren't in the queue
            self._evictable.move_to_end(block_hash)

    def evict(self, count: int) -> list[tuple[int, int]]:
        victims: list[tuple[int, int]] = []
        if count <= 0 or not self._evictable:
            return victims
        now = self._clock()
        ttl = self._ttl
        scanned = 0        # how many queue entries the pass actually touched
        ttl_skipped = 0    # idle-but-still-within-TTL entries we stepped over
        # Single pass from the LRU front. Only idle blocks live here, so we never
        # skip pinned entries -- at most we skip blocks still inside their TTL window.
        for block_hash in list(self._evictable):  # front = LRU
            scanned += 1
            if ttl > 0.0 and now < self._last_access[block_hash] + ttl:
                ttl_skipped += 1
                continue
            del self._evictable[block_hash]
            end_token_idx = self._table.pop(block_hash)
            del self._last_access[block_hash]
            self._forget_refcount(block_hash)
            victims.append((block_hash, end_token_idx))
            if len(victims) >= count:
                break
        # Proof-of-optimization log: `scanned` should stay ~= evicted (+ttl_skipped)
        # and be INDEPENDENT of `pinned`. Pinned blocks are not in the queue, so the
        # pass never walks them -- that's the O(victims) vs old O(resident) win.
        log.debug(
            "[LRU.evict] requested=%d evicted=%d | scanned=%d ttl_skipped=%d "
            "| eviction_queue=%d pinned=%d resident=%d (pinned NOT scanned)",
            count, len(victims), scanned, ttl_skipped,
            len(self._evictable), self._pinned_count, len(self._table),
        )
        return victims

    def __contains__(self, block_hash: int) -> bool:
        return block_hash in self._table

    def __len__(self) -> int:
        return len(self._table)


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
    """Determine the physical block cost of advancing a request's known token
    range from current_tokens to current_tokens + tokens_to_add, accounting
    for blocks whose content already exists elsewhere in the GPU APC (shared,
    ref-counted -- see BaseAPCPolicy.incref/decref).

    Pure / read-only with respect to apc_policy: only membership (`in`) is
    inspected, nothing is inserted or increfed here. The caller must actually
    commit (see commit_apc_blocks) once it has confirmed capacity_delta
    physical blocks are available (evicting first if necessary).

    chain_hash must be the BlockResolution.chain_hash returned by the most
    recent prior call for this same token sequence (None for the first call,
    when current_tokens == 0) -- this lets hashing resume from current_tokens
    instead of re-deriving the prefix-chained hash for the whole sequence
    from scratch on every call. Without it, repeated incremental calls over a
    growing sequence (one per decode step, for example) cost O(n) each,
    O(n^2) overall -- intractable for long contexts.

    Every block boundary fully crossed by [current_tokens, current_tokens +
    tokens_to_add) is "newly full" and becomes hashable; the trailing partial
    block (if the new total isn't block-aligned) is never hashed -- its
    content isn't finalized yet, so it always needs (or keeps) a private,
    unshared slot until a future call completes it.
    """
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
        for start, end, block_hash in apc_token_db.process_tokens_from(
            hash_ids_ref, start_idx, chain_hash, end_idx=token_limit
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
        name: Currently only "lru".
        ttl: Seconds a block is immune to eviction after its last access.
        clock: Callable returning current simulation time in seconds.
    """
    if name == "lru":
        return LRUPolicy(ttl=ttl, clock=clock)
    raise ValueError(f"Unknown APC eviction policy: {name!r}.")
