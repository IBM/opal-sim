# SPDX-License-Identifier: Apache-2.0
"""
ModelFactory: given a Hugging Face model name, loads its config.json
(ModelConfigLoader, falling back to a HF download cached into Opal's
model-configs/<name>/ layout), detects its architecture family (hybrid
Mamba+MoE, MLA, MLA+sparse-attention/DSA, or plain full attention), and
instantiates the matching LLMRooflineModel subclass -- passing the config
dict through for its own generic parse (opal_model.parse_opal_model_fields)
plus whatever architecture-specific extras (n_attn_layers, mla_latent_dim,
topk, ...) aren't derivable from that generic parse.
"""

from __future__ import annotations

from typing import Optional

from opal.llm_inference.config_loader import (
    ModelConfigLoader,
    UnsupportedModelError,
    _HYBRID_MAMBA_MOE_MODEL_TYPES,
    detect_architecture,
    effective_llm_config,
    get_config_field,
    kv_cache_dim_per_layer,
)
from opal.llm_inference.roofline_inference_model import (
    GPUConfig,
    LLMRooflineModel,
    FullAttentionRooflineModel,
    MLAAttentionRooflineModel,
    SparseAttentionRooflineModel,
    HybridMambaMoERooflineModel,
)

# --------------------------------------------------------------------------
# The factory
# --------------------------------------------------------------------------


class ModelFactory:
    """
    Usage:
        hw = GPUConfig(peak_flops_fp16=989.5e12, peak_flops_fp8=1979e12, peak_bandwidth=8e12)
        factory = ModelFactory(config_dir="model-configs")
        model = factory.create("nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4", hw)
        result = model.estimate(batch)   # batch: opal.worker.vllm_worker.BatchMetadata
    """

    def __init__(self, config_dir: str = "model-configs"):
        self.loader = ModelConfigLoader(config_dir)

    def create(
        self,
        model_name: str,
        hw: GPUConfig,
        active_params_override: Optional[float] = None,
        a_override: Optional[float] = None,
        b_override: Optional[float] = None,
        force_refresh: bool = False,
    ) -> LLMRooflineModel:
        raw_cfg = self.loader.load(model_name, force_refresh=force_refresh)
        return self.create_from_config(
            raw_cfg,
            model_name,
            hw,
            active_params_override=active_params_override,
            a_override=a_override,
            b_override=b_override,
        )

    def create_from_config(
        self,
        raw_cfg: dict,
        model_name: str,
        hw: GPUConfig,
        active_params_override: Optional[float] = None,
        a_override: Optional[float] = None,
        b_override: Optional[float] = None,
    ) -> LLMRooflineModel:
        """Same as create(), but takes an already-loaded config dict directly
        instead of doing its own config.json lookup -- used by
        inference_engine to build a model from the same config OpalModel
        already parsed for a worker, rather than fetching/parsing it twice.

        raw_cfg is passed straight through to the concrete LLMRooflineModel
        subclass's constructor (parses total/active params, bytes_per_elem,
        moe, etc. itself); this method's own job is just: detect
        architecture, and compute the architecture-specific extras that
        aren't derivable from that generic parse."""
        # unwrap VLM wrapper configs for architecture detection/field lookups
        # below; raw_cfg itself stays wrapped for the concrete model's own parse
        cfg = effective_llm_config(raw_cfg)
        arch = detect_architecture(cfg)

        model_type = str(cfg.get("model_type", "")).lower()
        hidden = get_config_field(cfg, ["hidden_size", "d_model", "n_embd"])
        if hidden is None:
            raise UnsupportedModelError(
                f"create_from_config: model_type {model_type!r} ({model_name}) has "
                f"none of hidden_size/d_model/n_embd -- cannot size this model "
                f"without it, and defaulting to 4096 would be an unverified guess"
            )
        layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern", "layer_types"])
        n_layers = get_config_field(cfg, ["num_hidden_layers", "n_layer", "num_layers"])
        if n_layers is None:
            if layer_pattern is None:
                raise UnsupportedModelError(
                    f"create_from_config: model_type {model_type!r} ({model_name}) has "
                    f"none of num_hidden_layers/n_layer/num_layers and no layer-type "
                    f"pattern to derive a count from -- defaulting to 32 would be an "
                    f"unverified guess"
                )
            n_layers = len(layer_pattern)

        # roofline-specific extras threaded into every LLMRooflineModel
        # subclass; n_layers supports the legacy-b-parity path
        kwargs = dict(n_layers=n_layers)
        if b_override is not None:
            kwargs["b"] = b_override
        if a_override is not None:
            kwargs["a"] = a_override

        if arch == "hybrid_mamba_moe":
            model = self._build_hybrid_mamba_moe(raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs)
        elif arch == "sparse_mla":
            model = self._build_sparse_mla(raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs)
        elif arch == "mla":
            model = self._build_mla(raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs)
        else:
            model = FullAttentionRooflineModel(
                raw_cfg,
                model_name,
                hw,
                n_attn_layers=n_layers,
                **kwargs,
            )

        # active_params_override forces a specific active-param count
        # independent of the config parse; drop .moe since its breakdown is
        # no longer consistent with an overridden active_params
        if active_params_override is not None:
            model.active_params = active_params_override
            model.moe = None

        model.name = model_name
        return model

    # ---- per-architecture construction helpers ----

    def _build_hybrid_mamba_moe(
        self, raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs
    ) -> HybridMambaMoERooflineModel:
        model_type = str(cfg.get("model_type", "")).lower()
        n_mamba = get_config_field(cfg, ["num_mamba_layers"], None)
        n_attn = get_config_field(cfg, ["num_attention_layers"], None)

        if n_mamba is None or n_attn is None:
            pattern = get_config_field(cfg, ["hybrid_override_pattern", "layers_block_type", "layer_types"], None)
            if pattern is not None:
                pattern_list = list(pattern) if not isinstance(pattern, str) else list(pattern)
                n_attn = sum(1 for p in pattern_list if str(p).lower() in ("*", "attention", "attn"))
                n_mamba = sum(1 for p in pattern_list if str(p).lower() in ("m", "mamba"))
            elif model_type == "jamba":
                # Jamba has no pattern field at all -- attention layers are
                # every attn_layer_period-th layer (same convention
                # estimate_params/count_attention_layers use for this model_type).
                attn_period = cfg.get("attn_layer_period", 8)
                n_attn = sum(1 for i in range(n_layers) if i % attn_period == 0)
                n_mamba = n_layers - n_attn
            else:
                raise UnsupportedModelError(
                    f"_build_hybrid_mamba_moe: model_type {model_type!r} ({model_name}) "
                    f"has no num_mamba_layers/num_attention_layers and no layer-type "
                    f"pattern to derive the mamba/attention split from -- guessing a "
                    f"10% attention-layer fraction here would be an unverified assumption"
                )

        if model_type not in _HYBRID_MAMBA_MOE_MODEL_TYPES:
            raise UnsupportedModelError(
                f"_build_hybrid_mamba_moe: model_type {model_type!r} ({model_name}) is "
                f"not a verified Mamba-2 family ({sorted(_HYBRID_MAMBA_MOE_MODEL_TYPES)}) "
                f"-- cannot size its SSM state without a verified per-family formula"
            )
        d_state = get_config_field(cfg, ["mamba_d_state", "ssm_state_size", "state_size"])
        if d_state is None:
            raise UnsupportedModelError(
                f"_build_hybrid_mamba_moe: model_type {model_type!r} ({model_name}) has "
                f"none of mamba_d_state/ssm_state_size/state_size -- defaulting to 128 "
                f"would be an unverified guess"
            )
        # mamba_scan_const is a fixed FLOPs-per-element constant for the
        # selective scan itself, not read from any known config field in real
        # Mamba-2 configs -- it's a genuine roofline-model constant, not a
        # stand-in for a missing config value, so it's only passed through
        # when the config explicitly overrides it; otherwise
        # HybridMambaMoERooflineModel's own constructor default (3.0) applies.
        scan_const = get_config_field(cfg, ["mamba_scan_const"])
        if scan_const is not None:
            kwargs = {**kwargs, "scan_const": scan_const}

        return HybridMambaMoERooflineModel(
            raw_cfg,
            model_name,
            hw,
            n_attn_layers=n_attn,
            n_mamba_layers=n_mamba,
            d_state=d_state,
            **kwargs,
        )

    def _build_sparse_mla(self, raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs) -> SparseAttentionRooflineModel:
        # kv_cache_dim_per_layer includes qk_rope_head_dim when present (the
        # decoupled RoPE key MLA caches alongside the compressed latent) --
        # ~12.5% higher for GLM-5.2 than a bare kv_lora_rank lookup. Computed
        # here explicitly (MLA-specific) rather than defaulting from
        # OpalModel's own architecture-agnostic accessor.
        latent = kv_cache_dim_per_layer(cfg, hidden)
        if latent is None:
            raise UnsupportedModelError(
                f"_build_sparse_mla: {model_name} was detected as MLA+sparse-attention "
                f"but has neither kv_lora_rank nor mla_latent_dim -- this isn't really "
                f"MLA, so defaulting the latent dim to 512 would be an unverified guess"
            )
        topk = get_config_field(cfg, ["index_topk", "dsa_topk", "nsa_topk"])
        if topk is None:
            raise UnsupportedModelError(
                f"_build_sparse_mla: {model_name} was detected as MLA+sparse-attention "
                f"(via index_n_heads or similar) but has none of index_topk/dsa_topk/"
                f"nsa_topk -- defaulting topk to 2048 would be an unverified guess"
            )
        share_period = get_config_field(cfg, ["index_share_period", "indexer_share_period", "index_topk_freq"])
        if share_period is None:
            raise UnsupportedModelError(
                f"_build_sparse_mla: {model_name} was detected as MLA+sparse-attention "
                f"but has none of index_share_period/indexer_share_period/"
                f"index_topk_freq -- defaulting the indexer share period to 4 would be "
                f"an unverified guess"
            )
        indexer_n_heads = get_config_field(cfg, ["index_n_heads", "indexer_n_heads"])
        indexer_head_dim = get_config_field(cfg, ["index_head_dim", "indexer_head_dim"])
        indexer_flops = get_config_field(cfg, ["indexer_flops_per_token"])
        if indexer_flops is None:
            if indexer_n_heads is None or indexer_head_dim is None:
                raise UnsupportedModelError(
                    f"_build_sparse_mla: {model_name} has no indexer_flops_per_token "
                    f"and no index_n_heads/index_head_dim to derive it from -- "
                    f"defaulting to 2*hidden*256 would be an unverified guess"
                )
            # 2x for the indexer's own Q/K projections, sized like a small
            # extra attention head group -- matches DeepSeek-V3.2's published
            # indexer shape (n_heads * head_dim), not a fixed constant
            indexer_flops = 2 * hidden * (indexer_n_heads * indexer_head_dim)

        return SparseAttentionRooflineModel(
            raw_cfg,
            model_name,
            hw,
            n_attn_layers=n_layers,
            mla_latent_dim=latent,
            topk=topk,
            indexer_flops_per_token=indexer_flops,
            indexer_share_period=share_period,
            **kwargs,
        )

    def _build_mla(self, raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs) -> MLAAttentionRooflineModel:
        latent = kv_cache_dim_per_layer(cfg, hidden)
        if latent is None:
            raise UnsupportedModelError(
                f"_build_mla: {model_name} was detected as MLA but has neither "
                f"kv_lora_rank nor mla_latent_dim -- this isn't really MLA, so "
                f"defaulting the latent dim to 512 would be an unverified guess"
            )
        return MLAAttentionRooflineModel(
            raw_cfg,
            model_name,
            hw,
            n_attn_layers=n_layers,
            mla_latent_dim=latent,
            **kwargs,
        )
