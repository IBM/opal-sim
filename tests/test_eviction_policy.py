# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GPU APC eviction policy: LRUPolicy ref-counting/eviction
ordering, and the resolve_apc_blocks/commit_apc_blocks block-accounting math.

These are pure, simpy-free units -- no worker, scheduler, or simulation
environment is needed.
"""
from opal.kvcache.eviction_policy import LRUPolicy, commit_apc_blocks, resolve_apc_blocks
from opal.kvcache.kvc_manager import OpalEngineMetadata, OpalTokenDatabase

BLOCK_SIZE = 4


def make_token_db(chunk_size: int = BLOCK_SIZE) -> OpalTokenDatabase:
    metadata = OpalEngineMetadata("test-model", 1, 0, "vllm", "fp16", (1, 2, chunk_size, 1, 1))
    return OpalTokenDatabase(chunk_size, metadata)


class TestLRUPolicyRefCounting:
    def test_incref_pins_block_out_of_eviction_queue(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        assert policy.evictable_count() == 1

        policy.incref(1)
        assert policy.ref_count(1) == 1
        assert policy.evictable_count() == 0
        assert policy.evict(1) == []  # pinned -- nothing to evict

    def test_decref_readmits_block_as_most_recently_used(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.insert(2, 8)
        policy.incref(1)

        policy.decref(1)
        assert policy.ref_count(1) == 0
        assert policy.evictable_count() == 2
        # 1 was idle, then pinned, then released -> now MRU, so 2 (never
        # touched) is evicted first.
        assert policy.evict(1) == [(2, 8)]

    def test_decref_on_untracked_hash_is_noop_floored_at_zero(self):
        policy = LRUPolicy()
        assert policy.decref(999) == 0
        assert policy.ref_count(999) == 0
        assert policy.evictable_count() == 0

    def test_multiple_increfs_require_matching_decrefs_before_evictable(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.incref(1)
        policy.incref(1)
        assert policy.ref_count(1) == 2

        policy.decref(1)
        assert policy.evictable_count() == 0  # still pinned (ref_count 1)

        policy.decref(1)
        assert policy.ref_count(1) == 0
        assert policy.evictable_count() == 1


class TestLRUPolicyEvictionOrder:
    def test_evict_returns_victims_in_lru_order(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.insert(2, 8)
        policy.insert(3, 12)

        assert policy.evict(2) == [(1, 4), (2, 8)]
        assert len(policy) == 1
        assert 3 in policy

    def test_evict_skips_pinned_blocks_regardless_of_position(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.insert(2, 8)
        policy.insert(3, 12)
        policy.incref(2)  # pin the middle (otherwise-LRU-ish) block

        victims = policy.evict(3)  # ask for more than available idle blocks
        assert victims == [(1, 4), (3, 12)]
        assert 2 in policy  # pinned block survives
        assert policy.ref_count(2) == 1

    def test_evict_returns_fewer_than_requested_when_queue_short(self):
        policy = LRUPolicy()
        policy.insert(1, 4)

        assert policy.evict(5) == [(1, 4)]
        assert len(policy) == 0

    def test_evict_on_empty_policy_returns_empty_list(self):
        policy = LRUPolicy()
        assert policy.evict(3) == []

    def test_touch_moves_block_to_most_recently_used(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.insert(2, 8)

        policy.touch(1)  # 1 was LRU, now becomes MRU
        assert policy.evict(1) == [(2, 8)]

    def test_evicted_block_forgets_refcount_bookkeeping(self):
        policy = LRUPolicy()
        policy.insert(1, 4)
        policy.incref(1)
        policy.decref(1)  # idle again, ref-count entry dropped internally

        policy.evict(1)
        assert 1 not in policy
        assert policy.ref_count(1) == 0
        assert policy.evictable_count() == 0


class TestLRUPolicyTTL:
    def test_ttl_immune_block_is_skipped_until_window_elapses(self):
        clock_box = [0.0]
        policy = LRUPolicy(ttl=10.0, clock=lambda: clock_box[0])
        policy.insert(1, 4)  # last_access = 0.0

        clock_box[0] = 5.0
        assert policy.evict(1) == []  # still within the 10s TTL window
        assert 1 in policy

        clock_box[0] = 10.1
        assert policy.evict(1) == [(1, 4)]

    def test_ttl_only_skips_immune_entries_not_eligible_ones(self):
        clock_box = [0.0]
        policy = LRUPolicy(ttl=10.0, clock=lambda: clock_box[0])
        policy.insert(1, 4)  # accessed at t=0 -- eligible after t=10

        clock_box[0] = 20.0
        policy.insert(2, 8)  # accessed at t=20 -- still immune

        victims = policy.evict(2)
        assert victims == [(1, 4)]  # only the eligible one comes back
        assert 2 in policy


class TestResolveApcBlocks:
    def test_sub_block_growth_reserves_one_private_slot(self):
        db = make_token_db()
        policy = LRUPolicy()
        tokens = list(range(20))

        r = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, current_tokens=0, tokens_to_add=3)

        assert r.capacity_delta == 1
        assert r.private_delta == 1
        assert r.new_block_hashes == []
        assert r.attach_hashes == []
        assert r.chain_hash is None  # no full block crossed yet

    def test_repeated_growth_within_same_partial_block_does_not_double_reserve(self):
        db = make_token_db()
        policy = LRUPolicy()
        tokens = list(range(20))

        r1 = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, 0, 1)
        r2 = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, 1, 1, chain_hash=r1.chain_hash)

        assert r1.capacity_delta == 1 and r1.private_delta == 1
        assert r2.capacity_delta == 0 and r2.private_delta == 0

    def test_completing_a_private_partial_block_promotes_it_at_zero_net_cost(self):
        db = make_token_db()
        policy = LRUPolicy()
        tokens = list(range(20))

        partial = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, 0, 3)
        completed = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, 3, 1, chain_hash=partial.chain_hash)

        # No new physical block needed: the private slot just gets registered.
        assert completed.capacity_delta == 0
        assert completed.private_delta == -1
        assert len(completed.new_block_hashes) == 1
        assert completed.attach_hashes == []

    def test_attaching_to_already_cached_full_block_is_free(self):
        db = make_token_db()
        policy = LRUPolicy()
        seq_a = list(range(4)) + [100]
        seq_b = list(range(4)) + [200]  # shares the first block's content with seq_a

        ra = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_a, 0, 4)
        commit_apc_blocks(policy, ra, {}, seq_a)

        rb = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_b, 0, 4)

        assert rb.capacity_delta == 0
        assert rb.new_block_hashes == []
        assert rb.attach_hashes == [ra.new_block_hashes[0][0]]

    def test_completing_a_partial_block_that_dedupes_frees_the_private_slot(self):
        db = make_token_db()
        policy = LRUPolicy()
        seq_a = list(range(4))

        # Another request already registered the identical full block.
        pre_existing = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_a, 0, 4)
        commit_apc_blocks(policy, pre_existing, {}, seq_a)

        # This request grows its own private partial tail into the same content.
        partial = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_a, 0, 3)
        completed = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_a, 3, 1, chain_hash=partial.chain_hash)

        # The private slot is released AND no new physical block was needed
        # (net: one full block returned to the free pool).
        assert completed.capacity_delta == -1
        assert completed.private_delta == -1
        assert completed.new_block_hashes == []
        assert completed.attach_hashes == [pre_existing.new_block_hashes[0][0]]

    def test_multiple_full_blocks_crossed_in_one_call(self):
        db = make_token_db()
        policy = LRUPolicy()
        tokens = list(range(20))

        r = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, current_tokens=0, tokens_to_add=8)

        assert r.capacity_delta == 2
        assert len(r.new_block_hashes) == 2
        assert r.attach_hashes == []
        assert r.private_delta == 0  # no partial tail (8 is block-aligned)

    def test_incremental_chain_hash_resumption_matches_one_shot_derivation(self):
        """Resuming with the returned chain_hash must yield the exact same
        per-block hashes as deriving from token 0 in a single call -- this is
        the O(n) vs O(n^2) correctness guarantee resolve_apc_blocks relies on.
        """
        db = make_token_db()
        tokens = list(range(12))  # 3 full blocks

        one_shot = resolve_apc_blocks(db, LRUPolicy(), BLOCK_SIZE, tokens, 0, 12)
        one_shot_hashes = [h for h, _ in one_shot.new_block_hashes]

        incremental_hashes = []
        chain_hash = None
        current = 0
        policy = LRUPolicy()
        for _ in range(3):
            r = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, current, 4, chain_hash=chain_hash)
            incremental_hashes.extend(h for h, _ in r.new_block_hashes)
            chain_hash = r.chain_hash
            current += 4

        assert incremental_hashes == one_shot_hashes
        assert one_shot.chain_hash == chain_hash


class TestCommitApcBlocks:
    def test_new_block_hashes_are_inserted_and_increfed(self):
        db = make_token_db()
        policy = LRUPolicy()
        tokens = list(range(4))
        block_source: dict = {}

        r = resolve_apc_blocks(db, policy, BLOCK_SIZE, tokens, 0, 4)
        commit_apc_blocks(policy, r, block_source, tokens)

        block_hash = r.new_block_hashes[0][0]
        assert block_hash in policy
        assert policy.ref_count(block_hash) == 1
        assert block_source[block_hash] == (tokens, r.new_block_hashes[0][1])

    def test_attach_hashes_are_increfed_without_reinserting(self):
        db = make_token_db()
        policy = LRUPolicy()
        seq_a = list(range(4)) + [100]
        seq_b = list(range(4)) + [200]
        block_source: dict = {}

        ra = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_a, 0, 4)
        commit_apc_blocks(policy, ra, block_source, seq_a)
        block_hash = ra.new_block_hashes[0][0]

        rb = resolve_apc_blocks(db, policy, BLOCK_SIZE, seq_b, 0, 4)
        commit_apc_blocks(policy, rb, block_source, seq_b)

        assert policy.ref_count(block_hash) == 2
        # block_source keeps the first owner's reference (setdefault semantics).
        assert block_source[block_hash] == (seq_a, ra.new_block_hashes[0][1])
