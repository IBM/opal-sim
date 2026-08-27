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

        hidden = get_config_field(cfg, ["hidden_size", "d_model", "n_embd"], 4096)
        layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern"])
        n_layers = get_config_field(cfg, ["num_hidden_layers", "n_layer", "num_layers"])
        if n_layers is None:
            n_layers = len(layer_pattern) if layer_pattern is not None else 32

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
        n_mamba = get_config_field(cfg, ["num_mamba_layers"], None)
        n_attn = get_config_field(cfg, ["num_attention_layers"], None)

        if n_mamba is None or n_attn is None:
            pattern = get_config_field(cfg, ["hybrid_override_pattern", "layers_block_type"], None)
            if pattern is not None:
                pattern_list = list(pattern) if not isinstance(pattern, str) else list(pattern)
                n_attn = sum(1 for p in pattern_list if str(p).lower() in ("*", "attention", "attn"))
                n_mamba = sum(1 for p in pattern_list if str(p).lower() in ("m", "mamba"))
            else:
                # heuristic fallback: hybrids typically keep only a small
                # fraction of layers as full attention
                n_attn = max(1, round(n_layers * 0.1))
                n_mamba = n_layers - n_attn

        d_state = get_config_field(cfg, ["mamba_d_state", "ssm_state_size", "state_size"], 128)
        scan_const = get_config_field(cfg, ["mamba_scan_const"], 3.0)

        return HybridMambaMoERooflineModel(
            raw_cfg,
            model_name,
            hw,
            n_attn_layers=n_attn,
            n_mamba_layers=n_mamba,
            d_state=d_state,
            scan_const=scan_const,
            **kwargs,
        )

    def _build_sparse_mla(self, raw_cfg, cfg, hw, hidden, n_layers, model_name, kwargs) -> SparseAttentionRooflineModel:
        # kv_cache_dim_per_layer includes qk_rope_head_dim when present (the
        # decoupled RoPE key MLA caches alongside the compressed latent) --
        # ~12.5% higher for GLM-5.2 than a bare kv_lora_rank lookup. Computed
        # here explicitly (MLA-specific) rather than defaulting from
        # OpalModel's own architecture-agnostic accessor.
        latent = kv_cache_dim_per_layer(cfg, hidden) or 512
        topk = get_config_field(cfg, ["index_topk", "dsa_topk", "nsa_topk"], 2048)
        share_period = get_config_field(cfg, ["index_share_period", "indexer_share_period", "index_topk_freq"], 4)
        indexer_flops = get_config_field(cfg, ["indexer_flops_per_token"], 2 * hidden * 256)

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
        latent = kv_cache_dim_per_layer(cfg, hidden) or 512
        return MLAAttentionRooflineModel(
            raw_cfg,
            model_name,
            hw,
            n_attn_layers=n_layers,
            mla_latent_dim=latent,
            **kwargs,
        )
