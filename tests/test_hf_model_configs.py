# SPDX-License-Identifier: Apache-2.0
"""
Validates OpalModel's param/MoE/KV-cache estimates against manually-sourced
documented ground truth for real Hugging Face model configs, downloaded live
(config.json only, no weights) via OpalModel.from_huggingface -- the same
transformers.AutoConfig-based path environment.py's get_model() uses in
production. No fixtures are checked into this repo (model configs are each
covered by their own upstream license).

Ground-truth figures below are hand-sourced from each model's official card/
technical report, not derived from the code under test:
  - Llama-3.1-8B-Instruct: 8.03B total params (Meta model card)
  - DeepSeek-V3: 671B total / 37B active (arXiv:2412.19437, HF model card)
  - Qwen3-30B-A3B: 30.5B total / 3.3B active (HF model card)
  - Nemotron-H-8B-Base-8K: ~8B total is the marketing name, but NVIDIA's own
    config implies ~9-10B when summed layer-by-layer (52 layers, MLP layers
    included) -- see the hand-calculation this test's assertion is based on.

A tolerance (not exact equality) is used throughout since "documented"
figures are themselves rounded/approximate and our estimate is derived from
config dimensions alone, not real weight counts.
"""

import pytest

from opal.llm_inference.config_loader import ModelConfigLoader, estimate_params
from opal.llm_inference.opal_model import OpalModel

pytestmark = pytest.mark.network


def assert_within(actual: float, expected: float, tol: float, label: str):
    lo, hi = expected * (1 - tol), expected * (1 + tol)
    assert lo <= actual <= hi, f"{label}: expected ~{expected:,.0f} (+/-{tol:.0%}), got {actual:,.0f}"


class TestDenseModel:
    """Llama-3.1-8B-Instruct: plain GQA, no MoE, no MLA."""

    @pytest.fixture(scope="class")
    def model(self):
        return OpalModel.from_huggingface("meta-llama/Llama-3.1-8B-Instruct")

    def test_param_count(self, model):
        assert_within(model.total_params, 8.03e9, tol=0.02, label="total_params")
        assert model.total_params == model.active_params  # dense: no MoE routing
        assert model.moe is None

    def test_kv_cache_per_token(self, model):
        # GQA: 8 kv heads, head_dim 128, bf16 (2 bytes), K+V, x32 layers
        expected = 2 * 8 * 128 * 2 * 32
        assert model.key_value_bytes == expected
        assert model.get_kvc_bytes(1000) == expected * 1000


class TestMoEModel:
    """Qwen3-30B-A3B: standard MoE, full attention (no MLA)."""

    @pytest.fixture(scope="class")
    def model(self):
        return OpalModel.from_huggingface("Qwen/Qwen3-30B-A3B")

    def test_param_count(self, model):
        assert_within(model.total_params, 30.5e9, tol=0.03, label="total_params")
        assert_within(model.active_params, 3.3e9, tol=0.05, label="active_params")
        assert model.active_params < model.total_params

    def test_moe_breakdown(self, model):
        assert model.moe is not None
        assert model.moe.num_experts == 128
        assert model.moe.experts_per_tok == 8


class TestMLAModel:
    """DeepSeek-V3: MoE + multi-head latent attention (compressed KV cache)."""

    @pytest.fixture(scope="class")
    def model(self):
        return OpalModel.from_huggingface("deepseek-ai/DeepSeek-V3")

    def test_param_count(self, model):
        assert_within(model.total_params, 671e9, tol=0.02, label="total_params")
        assert_within(model.active_params, 37e9, tol=0.10, label="active_params")

    def test_moe_breakdown(self, model):
        assert model.moe is not None
        assert model.moe.num_experts == 256
        assert model.moe.experts_per_tok == 8

    def test_mla_kv_cache_is_compressed(self, model):
        # MLA caches one compressed latent (kv_lora_rank) + a small decoupled
        # RoPE key (qk_rope_head_dim), not full per-head K/V -- this is the
        # whole point of MLA, so assert it directly against the config's own
        # dimensions rather than a hardcoded byte count.
        kv_lora_rank = model.config_dict["kv_lora_rank"]
        qk_rope_head_dim = model.config_dict["qk_rope_head_dim"]
        expected = (kv_lora_rank + qk_rope_head_dim) * model.attention_layers * model.kv_cache_bytes_per_elem
        assert model.key_value_bytes == expected

        # sanity: MLA's per-token footprint is far below a naive GQA/MHA
        # formula would give for the same head count -- confirms the MLA
        # branch actually fired instead of silently falling back to GQA
        num_kv_heads = model.config_dict.get("num_key_value_heads", model.num_attention_heads)
        naive_gqa_dim = 2 * num_kv_heads * model.kv_head_size
        assert kv_lora_rank + qk_rope_head_dim < naive_gqa_dim


class TestHybridMambaModel:
    """Nemotron-H-8B-Base-8K: hybrid Mamba-2 + attention + MLP layers."""

    def test_param_count_via_from_huggingface(self):
        model = OpalModel.from_huggingface("nvidia/Nemotron-H-8B-Base-8K")
        assert_within(model.total_params, 10.1e9, tol=0.05, label="total_params")

    def test_param_count_via_config_loader(self, tmp_path):
        # Exercises the same estimate_params() logic as the test above, but
        # via the raw-JSON path (ModelConfigLoader), which never touches
        # transformers.AutoConfig -- this is the actual regression test for
        # the MLP-param-zeroing bug fixed in config_loader.py: dense hybrid
        # models were undercounted at ~3.76B instead of ~10.1B because MLP
        # params were zeroed for every pattern layer, not just mamba/
        # attention ones.
        loader = ModelConfigLoader(config_dir=str(tmp_path))
        cfg = loader.load("nvidia/Nemotron-H-8B-Base-8K")
        estimate = estimate_params(cfg)
        assert_within(estimate.total, 10.1e9, tol=0.05, label="total_params")
