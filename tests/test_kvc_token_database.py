# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OpalTokenDatabase.process_tokens_from -- the resumable,
chunk-aligned hashing entry point that resolve_apc_blocks relies on to avoid
re-deriving the whole prefix-hash chain from token 0 on every call.
"""
import pytest

from opal.kvcache.kvc_manager import OpalEngineMetadata, OpalTokenDatabase

CHUNK_SIZE = 4


def make_token_db(chunk_size: int = CHUNK_SIZE) -> OpalTokenDatabase:
    metadata = OpalEngineMetadata("test-model", 1, 0, "vllm", "fp16", (1, 2, chunk_size, 1, 1))
    return OpalTokenDatabase(chunk_size, metadata)


class TestProcessTokensFrom:
    def test_matches_process_tokens_when_starting_from_zero(self):
        db = make_token_db()
        tokens = list(range(10))  # 2 full chunks + 1 trailing partial chunk

        assert list(db.process_tokens_from(tokens, 0, None)) == list(db.process_tokens(tokens))

    def test_rejects_non_chunk_aligned_start_idx(self):
        db = make_token_db()
        tokens = list(range(10))

        with pytest.raises(AssertionError, match="not chunk-aligned"):
            list(db.process_tokens_from(tokens, 2, None))

    def test_end_idx_truncates_the_final_chunk(self):
        db = make_token_db()
        tokens = list(range(10))

        result = list(db.process_tokens_from(tokens, 0, None, end_idx=6))

        assert [(s, e) for s, e, _ in result] == [(0, 4), (4, 6)]

    def test_resuming_with_correct_prior_hash_matches_one_shot_chain(self):
        """The whole point of process_tokens_from: resuming from a later
        start_idx with the true prior chunk's hash must reproduce exactly the
        hashes a full derivation from token 0 would have produced.
        """
        db = make_token_db()
        tokens = list(range(12))  # 3 full chunks

        one_shot = list(db.process_tokens_from(tokens, 0, None))

        first_chunk = list(db.process_tokens_from(tokens, 0, None, end_idx=4))
        prior_hash = first_chunk[-1][2]
        resumed_rest = list(db.process_tokens_from(tokens, 4, prior_hash))

        assert first_chunk + resumed_rest == one_shot

    def test_resuming_without_the_prior_hash_diverges_from_the_true_chain(self):
        """Guards the resumption contract itself: omitting prefix_hash resets
        the chain (falls back to the init hash) instead of continuing it, so
        callers that lose track of chain_hash silently get wrong block
        identities rather than an error.
        """
        db = make_token_db()
        tokens = list(range(12))

        one_shot = list(db.process_tokens_from(tokens, 0, None))
        resumed_without_prefix = list(db.process_tokens_from(tokens, 4, None))

        assert resumed_without_prefix[0][2] != one_shot[1][2]

    def test_absolute_token_offsets_preserved_when_resuming_mid_sequence(self):
        db = make_token_db()
        tokens = list(range(12))

        result = list(db.process_tokens_from(tokens, 4, None, end_idx=12))

        assert [(s, e) for s, e, _ in result] == [(4, 8), (8, 12)]
