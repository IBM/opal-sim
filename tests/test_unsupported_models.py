# SPDX-License-Identifier: Apache-2.0
"""
Negative-path tests for opal.llm_inference's strict per-family validation:
synthetic config dicts (no network, no fixtures) that should be rejected
with UnsupportedModelError rather than silently guessed at, one per
assumption enumerated in the strict-validation pass (see
opal/llm_inference/config_loader.py and inference_model_factory.py).
"""

import pytest

from opal.llm_inference.config_loader import (
    UnsupportedModelError,
    count_attention_layers,
    detect_architecture,
    estimate_params,
)
from opal.llm_inference.inference_model_factory import ModelFactory
from opal.llm_inference.roofline_inference_model import GPUConfig

HW = GPUConfig(peak_flops_fp16=989.5e12, peak_flops_fp8=1979e12, peak_bandwidth=8e12)

DENSE_BASE = {
    "model_type": "granite",
    "hidden_size": 4096,
    "intermediate_size": 12800,
    "num_hidden_layers": 4,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 49159,
    "tie_word_embeddings": True,
}

HYBRID_BASE = {
    "model_type": "nemotron_h",
    "hidden_size": 4096,
    "num_hidden_layers": 4,
    "layers_block_type": ["M", "M", "*", "-"],
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 1000,
    "tie_word_embeddings": False,
    "intermediate_size": 8192,
    "num_experts_per_tok": 1,
    "num_experts": 0,
}


class TestUnrecognizedModelType:
    def test_detect_architecture_rejects_unknown_model_type(self):
        cfg = {**DENSE_BASE, "model_type": "totally_made_up_model"}
        with pytest.raises(UnsupportedModelError):
            detect_architecture(cfg)

    def test_estimate_params_rejects_unknown_model_type(self):
        cfg = {**DENSE_BASE, "model_type": "totally_made_up_model"}
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)

    def test_model_factory_rejects_unknown_model_type(self):
        cfg = {**DENSE_BASE, "model_type": "totally_made_up_model"}
        with pytest.raises(UnsupportedModelError):
            ModelFactory().create_from_config(cfg, "made-up-model", HW)


class TestHybridMissingMambaFields:
    """model_type is in the hybrid_mamba_moe allowlist and the pattern marks
    layers as mamba, but none of the Mamba-2 state-space fields are present
    -- must raise instead of silently defaulting to 16/4/2."""

    def test_estimate_params_rejects_missing_mamba_fields(self):
        cfg = {**HYBRID_BASE}
        assert "state_size" not in cfg and "ssm_state_size" not in cfg and "mamba_d_state" not in cfg
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)

    def test_model_factory_rejects_hybrid_with_no_pattern_and_no_counts(self):
        # a hybrid model_type with neither num_mamba_layers/num_attention_layers
        # nor any layer-type pattern at all -- nothing to derive the split from
        cfg = {
            "model_type": "nemotron_h",
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 1000,
        }
        with pytest.raises(UnsupportedModelError):
            ModelFactory().create_from_config(cfg, "no-pattern-hybrid", HW)


class TestMoEMissingFields:
    def test_estimate_params_rejects_moe_missing_experts_per_tok(self):
        cfg = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 1000,
            "tie_word_embeddings": False,
            "num_experts": 8,
            "moe_intermediate_size": 512,
            # num_experts_per_tok deliberately omitted
        }
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)

    def test_estimate_params_rejects_moe_missing_expert_size(self):
        cfg = {
            "model_type": "qwen3_moe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "vocab_size": 1000,
            "tie_word_embeddings": False,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            # neither moe_intermediate_size nor intermediate_size present
        }
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)


class TestDenseMissingIntermediateSize:
    def test_estimate_params_rejects_dense_missing_intermediate_size(self):
        cfg = {**DENSE_BASE}
        del cfg["intermediate_size"]
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)


class TestAttentionMissingHeads:
    def test_estimate_params_rejects_attention_layer_missing_num_heads(self):
        cfg = {**DENSE_BASE}
        del cfg["num_attention_heads"]
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)


class TestMLAMissingLatentField:
    def test_model_factory_rejects_mla_without_latent_field(self):
        # has_mla/has_sparse_attn field checks never fire (no kv_lora_rank/
        # mla_latent_dim at all), so detect_architecture correctly falls back
        # to full_attention here -- this test instead exercises the
        # _build_mla/_build_sparse_mla defensive check directly, for a config
        # that claims MLA via model_type but is missing the latent dim field.
        from opal.llm_inference.inference_model_factory import ModelFactory as MF

        cfg = {
            "model_type": "deepseek_v3",
            "hidden_size": 4096,
            "num_hidden_layers": 2,
            "vocab_size": 1000,
            "tie_word_embeddings": False,
            "intermediate_size": 8192,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 512,
            # kv_lora_rank/mla_latent_dim AND num_attention_heads deliberately
            # omitted -- kv_cache_dim_per_layer's GQA fallback also needs
            # num_attention_heads, so this is what actually makes it return
            # None (a real deepseek_v3 config always has both kv_lora_rank
            # and num_attention_heads; this tests the defensive check itself)
        }
        factory = MF()
        cfg_eff = cfg
        with pytest.raises(UnsupportedModelError):
            factory._build_mla(cfg, cfg_eff, HW, cfg["hidden_size"], cfg["num_hidden_layers"], "no-latent-mla", {})


class TestHybridNoPatternNoModelTypeHeuristic:
    def test_count_attention_layers_rejects_hybrid_with_no_pattern(self):
        cfg = {
            "model_type": "granitemoehybrid",
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 1000,
        }
        with pytest.raises(UnsupportedModelError):
            count_attention_layers(cfg)


class TestGraniteHybridPatternContradiction:
    def test_estimate_params_rejects_granite_pattern_with_ffn_token(self):
        # granitemoehybrid is expected to use the 2-way (mamba/attention-only)
        # pattern convention; a pattern carrying an explicit FFN/MoE token
        # contradicts that assumption and must raise, not silently pick a
        # convention.
        cfg = {
            "model_type": "granitemoehybrid",
            "hidden_size": 4096,
            "num_hidden_layers": 4,
            "layer_types": ["mamba", "mamba", "attention", "-"],
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 1000,
            "tie_word_embeddings": False,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 512,
        }
        with pytest.raises(UnsupportedModelError):
            estimate_params(cfg)


class TestSanityPositivePath:
    """Confirms the strict checks above don't false-positive on the
    well-formed bases they're derived from."""

    def test_dense_base_does_not_raise(self):
        estimate = estimate_params(DENSE_BASE)
        assert estimate.total > 0
        assert estimate.moe is None

    def test_hybrid_base_does_not_raise(self):
        cfg = {**HYBRID_BASE, "state_size": 128, "conv_kernel": 4, "expand": 2}
        estimate = estimate_params(cfg)
        assert estimate.total > 0
