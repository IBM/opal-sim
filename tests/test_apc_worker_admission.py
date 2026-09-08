# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the GPU-APC worker glue in opal/worker/vllm_worker.py:
_apc_admit_tokens, _apc_release, and _apc_lookup.

These methods are entangled with LLMWorkerVLLMScheduler's simpy-driven state
(free_gpu_blocks, the KVC manager, the simulation clock), so a full worker is
too heavy for a unit test. Instead we bind the real, unmodified methods onto a
minimal stand-in object that only carries the attributes those methods
actually touch (verified against the source, not guessed) -- everything below
runs the production block-accounting logic, just without a simpy environment.
"""
from types import SimpleNamespace

from opal.kvcache.eviction_policy import LRUPolicy
from opal.kvcache.kvc_manager import OpalEngineMetadata, OpalTokenDatabase
from opal.worker.vllm_worker import LLMWorkerVLLMScheduler

BLOCK_SIZE = 4


def make_token_db() -> OpalTokenDatabase:
    metadata = OpalEngineMetadata("test-model", 1, 0, "vllm", "fp16", (1, 2, BLOCK_SIZE, 1, 1))
    return OpalTokenDatabase(BLOCK_SIZE, metadata)


def _drain(gen):
    """Stand-in for simpy_env.process: run the (store) generator to completion
    synchronously -- we don't care about simulated timing here, only that the
    real _apc_evict_blocks code path doesn't error when it fires a store."""
    try:
        while True:
            next(gen)
    except StopIteration:
        pass


def _fake_store(hash_ids):
    yield from ()


class FakeAPCWorker:
    """Binds the real worker methods under test; everything else is a stub."""

    _apc_hash_ids_for = LLMWorkerVLLMScheduler._apc_hash_ids_for
    _apc_admit_tokens = LLMWorkerVLLMScheduler._apc_admit_tokens
    _apc_release = LLMWorkerVLLMScheduler._apc_release
    _apc_lookup = LLMWorkerVLLMScheduler._apc_lookup
    _apc_evict_blocks = LLMWorkerVLLMScheduler._apc_evict_blocks


def make_worker(total_blocks: int) -> FakeAPCWorker:
    w = FakeAPCWorker()
    w.scheduler_config = SimpleNamespace(enable_gpu_apc=True)
    w.block_size = BLOCK_SIZE
    w.total_gpu_blocks = total_blocks
    w.free_gpu_blocks = total_blocks
    w._apc_token_db = make_token_db()
    w._apc_policy = LRUPolicy()
    w._apc_block_source = {}
    w._kvc_manager = SimpleNamespace(
        token_database=SimpleNamespace(chunk_size=BLOCK_SIZE),
        store=_fake_store,
    )
    w.simpy_env = SimpleNamespace(process=_drain)
    return w


def make_request(request_id: int, tokens: list) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        hash_ids=tokens,
        _apc_hash_ids_ref=tokens,  # pre-set so _apc_hash_ids_for is a pure pass-through
        apc_resolved_tokens=0,
        apc_owned_hashes=set(),
        apc_chain_hash=None,
        apc_private_blocks=0,
        allocated_blocks=0,
        _apc_prompt_chain=None,
    )


class TestApcAdmitTokens:
    def test_admits_a_fresh_full_block_and_tracks_ownership(self):
        worker = make_worker(total_blocks=4)
        request = make_request(1, list(range(4)))

        delta = worker._apc_admit_tokens(request, 4)

        assert delta == 1
        assert worker.free_gpu_blocks == 3
        assert request.allocated_blocks == 1
        assert request.apc_resolved_tokens == 4
        assert len(request.apc_owned_hashes) == 1
        owned_hash = next(iter(request.apc_owned_hashes))
        assert worker._apc_policy.ref_count(owned_hash) == 1

    def test_partial_tail_growth_reserves_a_private_block(self):
        worker = make_worker(total_blocks=4)
        request = make_request(1, list(range(5)))  # 1 full block + 1 trailing token

        delta = worker._apc_admit_tokens(request, 5)

        assert delta == 2  # 1 shared block + 1 private partial-tail reservation
        assert request.apc_private_blocks == 1
        assert request.allocated_blocks == 2
        assert len(request.apc_owned_hashes) == 1  # only the full block is registered

    def test_evicts_idle_blocks_to_cover_a_shortfall(self):
        worker = make_worker(total_blocks=1)
        owner_a = make_request(1, list(range(4)))
        worker._apc_admit_tokens(owner_a, 4)
        hash_a = next(iter(owner_a.apc_owned_hashes))
        worker._apc_release(owner_a)  # block becomes idle but stays resident
        assert hash_a in worker._apc_policy

        owner_b = make_request(2, [100, 101, 102, 103])  # distinct content -> new block
        delta = worker._apc_admit_tokens(owner_b, 4)

        assert delta == 1
        assert hash_a not in worker._apc_policy  # evicted to make room
        assert worker.free_gpu_blocks == 0
        assert next(iter(owner_b.apc_owned_hashes)) in worker._apc_policy

    def test_returns_none_and_leaves_state_untouched_when_capacity_cannot_be_freed(self):
        worker = make_worker(total_blocks=1)
        owner_a = make_request(1, list(range(4)))
        worker._apc_admit_tokens(owner_a, 4)  # pins the only block (still owned)
        hash_a = next(iter(owner_a.apc_owned_hashes))

        # Second request's first block dedupes against A's (attach), but its
        # second block is genuinely new and there is no free/evictable capacity.
        owner_b = make_request(2, list(range(4)) + [999, 998, 997, 996])
        result = worker._apc_admit_tokens(owner_b, 8)

        assert result is None
        assert worker.free_gpu_blocks == 0
        assert owner_b.apc_owned_hashes == set()  # speculative attach incref was undone
        assert owner_b.apc_resolved_tokens == 0
        assert worker._apc_policy.ref_count(hash_a) == 1  # A's ownership is unaffected


class TestApcRelease:
    def test_frees_private_blocks_and_decrefs_owned_hashes(self):
        worker = make_worker(total_blocks=3)
        request = make_request(1, list(range(5)))
        worker._apc_admit_tokens(request, 5)  # 1 shared block (owned) + 1 private
        owned_hash = next(iter(request.apc_owned_hashes))

        freed = worker._apc_release(request)

        assert freed == 1  # only the private slot is returned immediately
        assert worker.free_gpu_blocks == 2  # 1 (post-admit) + 1 private freed
        assert request.apc_owned_hashes == set()
        assert request.apc_private_blocks == 0
        assert request.allocated_blocks == 0
        # The shared block stays resident (still cached) but becomes idle/evictable.
        assert owned_hash in worker._apc_policy
        assert worker._apc_policy.ref_count(owned_hash) == 0

    def test_calling_release_twice_is_a_safe_noop_the_second_time(self):
        worker = make_worker(total_blocks=3)
        request = make_request(1, list(range(5)))
        worker._apc_admit_tokens(request, 5)

        worker._apc_release(request)
        free_after_first = worker.free_gpu_blocks

        freed_again = worker._apc_release(request)

        assert freed_again == 0
        assert worker.free_gpu_blocks == free_after_first  # no double-free


class TestApcLookup:
    def test_lookup_without_claim_is_read_only(self):
        worker = make_worker(total_blocks=4)
        owner = make_request(9, list(range(8)))  # 2 resident blocks
        worker._apc_admit_tokens(owner, 8)
        resident_hashes = set(owner.apc_owned_hashes)
        worker._apc_release(owner)  # idle, but still resident/discoverable

        seeker = make_request(3, list(range(8)) + [777, 778, 779, 780])
        matched = worker._apc_lookup(seeker.hash_ids, claim=False, request=seeker)

        assert matched == 8
        assert seeker.apc_owned_hashes == set()  # nothing claimed
        for h in resident_hashes:
            assert worker._apc_policy.ref_count(h) == 0  # untouched -- still idle

    def test_lookup_with_claim_pins_matched_blocks_and_sets_chain_hash(self):
        worker = make_worker(total_blocks=4)
        owner = make_request(9, list(range(8)))
        worker._apc_admit_tokens(owner, 8)
        worker._apc_release(owner)

        seeker = make_request(4, list(range(8)) + [777, 778, 779, 780])
        matched = worker._apc_lookup(seeker.hash_ids, claim=True, request=seeker)

        assert matched == 8
        assert len(seeker.apc_owned_hashes) == 2
        assert seeker.apc_chain_hash is not None
        for h in seeker.apc_owned_hashes:
            assert worker._apc_policy.ref_count(h) == 1

    def test_lookup_stops_at_the_first_missing_block_no_gap_matching(self):
        worker = make_worker(total_blocks=4)
        resident = make_request(10, list(range(4)))
        worker._apc_admit_tokens(resident, 4)

        # block 0 resident, block 1 missing, block 2 would coincidentally also
        # be absent -- lookup must not skip the gap and match past it.
        seeker_tokens = list(range(4)) + [500, 501, 502, 503] + [600, 601, 602, 603]
        seeker = make_request(5, seeker_tokens)

        matched = worker._apc_lookup(seeker.hash_ids, claim=False, request=seeker)

        assert matched == 4

    def test_lookup_caches_the_prompt_chain_per_request(self):
        worker = make_worker(total_blocks=4)
        call_count = []
        original = worker._apc_token_db.process_tokens

        def counting_process_tokens(*args, **kwargs):
            call_count.append(1)
            return original(*args, **kwargs)

        worker._apc_token_db.process_tokens = counting_process_tokens
        seeker = make_request(6, list(range(4)))

        worker._apc_lookup(seeker.hash_ids, claim=False, request=seeker)
        worker._apc_lookup(seeker.hash_ids, claim=False, request=seeker)

        assert len(call_count) == 1  # second call reused seeker._apc_prompt_chain
