# SPDX-License-Identifier: Apache-2.0
"""
Model config loading + architecture detection + parameter estimation for the
llm_inference roofline models.

Consolidates what opal.llm_model.OpalModelConfig and the old
inference_model_factory.ConfigLoader each did separately: config.json lookup
(Opal's model-configs/<name>/config.json layout, HF download fallback),
architecture-family detection, total/active param estimation from config
dimensions alone, and dtype -> bytes-per-element mapping (including NVFP4).
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from opal.llm_inference.opal_model import MoEParams

# --------------------------------------------------------------------------
# Config field lookup
# --------------------------------------------------------------------------


def get_config_field(cfg: dict, keys: List[str], default: Any = None) -> Any:
    """First present key from a list of aliases (model families name the
    same concept differently, e.g. 'hidden_size' vs 'd_model' vs 'n_embd')."""
    for k in keys:
        if k in cfg and cfg[k] is not None:
            return cfg[k]
    return default


def effective_llm_config(cfg: dict) -> dict:
    """Unwrap a multimodal/VLM wrapper config (e.g. Kimi-K2.7-Code) down to
    its text backbone under 'text_config'. No-op for ordinary flat configs."""
    if "hidden_size" not in cfg and "d_model" not in cfg and isinstance(cfg.get("text_config"), dict):
        return cfg["text_config"]
    return cfg


# --------------------------------------------------------------------------
# Config loading: Opal's on-disk layout first, Hugging Face download as
# fallback (cached into that same layout).
# --------------------------------------------------------------------------


class ModelConfigLoader:
    """Fetches a model's config.json from Opal's on-disk
    model-configs/<name>/config.json layout, falling back to Hugging Face
    (and caching the result back into that layout) if not found locally."""

    def __init__(self, config_dir: str = "model-configs"):
        self.config_dir = Path(config_dir)

    def _model_dir_name(self, model_name: str) -> str:
        return model_name.rstrip("/").split("/")[-1]

    def _local_config_path(self, model_name: str) -> Path:
        return self.config_dir / self._model_dir_name(model_name) / "config.json"

    def load(self, model_name: str, force_refresh: bool = False) -> dict:
        path = self._local_config_path(model_name)
        if path.exists() and not force_refresh:
            with open(path, "r") as f:
                return json.load(f)

        cfg = self._download(model_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        return cfg

    def _download(self, model_name: str) -> dict:
        try:
            from huggingface_hub import hf_hub_download

            local_file = hf_hub_download(repo_id=model_name, filename="config.json")
            with open(local_file, "r") as f:
                return json.load(f)
        except ImportError:
            pass  # huggingface_hub not installed, fall through to raw HTTP
        except Exception as e:
            print(
                f"[ModelConfigLoader] huggingface_hub download failed ({e}); " f"falling back to a direct HTTPS fetch"
            )

        url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "opal-llm-inference"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            raise RuntimeError(
                f"Could not fetch config.json for '{model_name}' from Hugging Face "
                f"(tried huggingface_hub and {url}): {e}"
            ) from e


# --------------------------------------------------------------------------
# Architecture detection
# --------------------------------------------------------------------------


def detect_architecture(cfg: dict) -> str:
    """Returns one of: 'hybrid_mamba_moe', 'sparse_mla', 'mla',
    'full_attention', via heuristic hallmark-field checks. Extend this if a
    new model family doesn't fit these patterns."""
    cfg = effective_llm_config(cfg)
    model_type = str(cfg.get("model_type", "")).lower()
    architectures = [str(a).lower() for a in cfg.get("architectures", [])]
    tags = [model_type] + architectures

    has_mamba = (
        any("mamba" in t for t in tags)
        or get_config_field(cfg, ["hybrid_override_pattern", "layers_block_type", "mamba_d_state", "ssm_state_size"])
        is not None
    )
    has_mla = get_config_field(cfg, ["kv_lora_rank", "q_lora_rank", "mla_latent_dim"]) is not None
    has_sparse_attn = get_config_field(cfg, ["index_topk", "dsa_topk", "nsa_topk", "index_n_heads"]) is not None

    if has_mamba:
        return "hybrid_mamba_moe"
    if has_mla and has_sparse_attn:
        return "sparse_mla"
    if has_mla:
        return "mla"
    return "full_attention"


# --------------------------------------------------------------------------
# Total- and active-params-per-token estimation
# --------------------------------------------------------------------------


@dataclass
class ParamEstimate:
    total: float
    active: float
    moe: Optional[MoEParams] = None  # None for non-MoE (dense) models


def estimate_params(cfg: dict) -> ParamEstimate:
    """Layer-by-layer total/active param estimate from config dimensions
    alone (no weight download). Handles dense, MoE (with shared experts),
    LatentMoE, and Mamba/hybrid per-layer-pattern architectures -- validated
    against real configs (DeepSeek-V3, Kimi-K2.x, GLM-5.x, Nemotron-H/Ultra)
    to within a few percent of published total/active param counts.

    `active` is what LLMRooflineModel.active_params should be set to (drives
    FLOPs); `total` is the weight-footprint/capacity figure. For MoE
    configs, `.moe` carries the breakdown LLMRooflineModel.bytes_weights
    needs for batch-size-dependent weight-read bytes (see MoEParams):
    shared_active_params + moe_layers * num_experts * expert_params_per_layer
    equals total exactly, and collapses to active at a single token.
    """
    cfg = effective_llm_config(cfg)
    model_type = str(cfg.get("model_type", "")).lower()
    num_experts = get_config_field(cfg, ["num_experts", "n_routed_experts", "num_local_experts"], 0)
    num_shared_experts = get_config_field(cfg, ["n_shared_experts", "num_shared_experts"], 0)
    num_dense_layers = get_config_field(cfg, ["num_dense_layers", "first_k_dense_replace"], 0)
    is_moe = num_experts > 0
    # LatentMoE (e.g. Nemotron-3-Ultra): experts operate on a projected-down
    # latent dim with a 2-matrix (up+down) shape instead of standard 3-matrix
    # SwiGLU -- confirmed against real safetensors tensor names
    moe_latent_size = cfg.get("moe_latent_size")

    hidden_size = cfg.get("hidden_size") or cfg.get("d_model")
    layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern"])
    num_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    if num_layers is None and layer_pattern is not None:
        num_layers = len(layer_pattern)
    vocab_size = cfg.get("vocab_size")
    tie_embeddings = cfg.get("tie_word_embeddings", False)

    embedding_params = vocab_size * hidden_size
    output_params = 0 if tie_embeddings else (vocab_size * hidden_size)

    total_transformer_params = 0
    active_transformer_params = 0

    moe_layers = 0
    expert_params_per_layer = 0.0  # same shape assumed for every MoE layer
    experts_per_tok_val = 0

    for i in range(num_layers):
        # per-layer type from the published pattern when present (covers
        # 2-way mamba/attention and 3-way mamba/attention/FFN-only splits
        # like Nemotron-H's); else fall back to a global model_type heuristic
        pattern_token = None
        if layer_pattern is not None and i < len(layer_pattern):
            pattern_token = str(layer_pattern[i]).lower()
            is_layer_mamba = pattern_token in ("m", "mamba")
            is_layer_attention = pattern_token in ("*", "attention", "attn")
            # anything else: FFN-only layer, no sequence-mixing weights
        elif "jamba" in model_type:
            attn_period = cfg.get("attn_layer_period", 8)
            is_layer_mamba = i % attn_period != 0
            is_layer_attention = not is_layer_mamba
        else:
            is_layer_mamba = "mamba" in model_type
            is_layer_attention = not is_layer_mamba

        if is_layer_mamba:
            d_state = cfg.get("state_size") or cfg.get("ssm_state_size", 16)
            d_conv = cfg.get("conv_kernel", 4)
            expand = cfg.get("expand", 2)
            d_inner = int(expand * hidden_size)

            in_proj = hidden_size * (d_inner * 2)
            out_proj = d_inner * hidden_size
            conv_params = d_inner * d_conv

            dt_rank = (
                math.ceil(hidden_size / 16) if cfg.get("time_step_rank") == "auto" else cfg.get("time_step_rank", 1)
            )
            ssm_params = (d_inner * d_state) + (d_inner * dt_rank) + (dt_rank * hidden_size)

            comm_params = in_proj + out_proj + conv_params + ssm_params
        elif is_layer_attention:
            num_heads = cfg.get("num_attention_heads", 1)
            num_kv = cfg.get("num_key_value_heads", num_heads)
            head_dim = cfg.get("head_dim", hidden_size // num_heads)

            q_proj = hidden_size * (num_heads * head_dim)
            kv_proj = 2 * (num_kv * head_dim) * hidden_size
            o_proj = hidden_size * hidden_size
            comm_params = q_proj + kv_proj + o_proj
        else:
            comm_params = 0  # FFN-only layer: no sequence-mixing weights

        if pattern_token is not None:
            # a published pattern is exclusive: each layer is exactly one of
            # {mamba, attention, MoE-FFN}, confirmed against real tensor names
            current_layer_is_moe = is_moe and not is_layer_mamba and not is_layer_attention
        else:
            current_layer_is_moe = is_moe and (i >= num_dense_layers)
            if "jamba" in model_type:
                exp_period = cfg.get("expert_layer_period", 2)
                current_layer_is_moe = i % exp_period == 0

        if current_layer_is_moe:
            k = cfg.get("num_experts_per_tok", 1)
            router = hidden_size * num_experts

            if moe_latent_size is not None:
                expert_size = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or (hidden_size * 4)
                per_expert_params = 2 * moe_latent_size * expert_size
                latent_proj_params = 2 * hidden_size * moe_latent_size
                shared_expert_params = num_shared_experts * per_expert_params
                always_active = latent_proj_params + router + shared_expert_params
                layer_total_mlp = always_active + num_experts * per_expert_params
                layer_active_mlp = always_active + k * per_expert_params
            else:
                # standard 3-matrix SwiGLU experts on hidden_size (DeepSeek-V3/
                # Kimi-K2.x/GLM-5.x convention), plus always-active shared experts
                expert_size = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or (hidden_size * 4)
                per_expert_params = 3 * hidden_size * expert_size
                shared_expert_params = num_shared_experts * per_expert_params
                layer_total_mlp = num_experts * per_expert_params + shared_expert_params + router
                layer_active_mlp = k * per_expert_params + shared_expert_params + router

            moe_layers += 1
            expert_params_per_layer = per_expert_params
            experts_per_tok_val = k
        elif pattern_token is not None:
            layer_total_mlp = 0
            layer_active_mlp = 0
        else:
            intermediate_size = cfg.get("intermediate_size", hidden_size * 4)
            mlp_params = 3 * hidden_size * intermediate_size
            layer_total_mlp = mlp_params
            layer_active_mlp = mlp_params

        norms = 2 * hidden_size
        total_transformer_params += comm_params + layer_total_mlp + norms
        active_transformer_params += comm_params + layer_active_mlp + norms

    total = embedding_params + total_transformer_params + output_params
    active = embedding_params + active_transformer_params + output_params

    moe = None
    if is_moe and moe_layers > 0:
        shared_active_params = active - (moe_layers * experts_per_tok_val * expert_params_per_layer)
        moe = MoEParams(
            num_experts=int(num_experts),
            experts_per_tok=int(experts_per_tok_val),
            moe_layers=moe_layers,
            expert_params_per_layer=expert_params_per_layer,
            shared_active_params=shared_active_params,
        )

    return ParamEstimate(total=total, active=active, moe=moe)


# --------------------------------------------------------------------------
# dtype -> bytes/element
# --------------------------------------------------------------------------


def guess_bytes_per_elem(cfg: dict) -> float:
    blob = json.dumps(cfg).lower()
    if "fp4" in blob:
        return 0.5  # NVFP4 / MXFP4 etc.: 4-bit packed storage
    if "fp8" in blob:
        return 1.0
    if "bf16" in blob or "fp16" in blob or "float16" in blob:
        return 2.0
    return 2.0  # default assumption


def guess_kv_cache_bytes_per_elem(cfg: dict) -> float:
    """KV cache is sometimes quantized to a different precision than the
    weights (e.g. Nemotron-3-Ultra: NVFP4 weights, FP8 KV cache) -- prefer
    quantization_config.kv_cache_scheme when present. Checks both the raw
    config and its unwrapped backbone, since that field describes the whole
    checkpoint and lives at the outer wrapper level for VLM configs."""
    for candidate in (cfg, effective_llm_config(cfg)):
        kv_scheme = candidate.get("quantization_config", {}).get("kv_cache_scheme")
        if isinstance(kv_scheme, dict) and kv_scheme.get("num_bits"):
            return kv_scheme["num_bits"] / 8
    return guess_bytes_per_elem(cfg)


# --------------------------------------------------------------------------
# Layers that actually carry a growing, token-indexed KV cache
# --------------------------------------------------------------------------


def count_attention_layers(cfg: dict) -> int:
    """Number of layers carrying a growing per-token KV cache: all layers
    for a dense model, else only the pattern-marked attention layers for a
    hybrid (Mamba layers carry fixed-size recurrent state instead; MoE-FFN-
    only layers carry none). Using num_hidden_layers unconditionally
    overcounts KV-cache memory for hybrids -- 9x for Nemotron-3-Ultra."""
    cfg = effective_llm_config(cfg)
    layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern"])
    if layer_pattern is None:
        return cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    return sum(1 for tok in layer_pattern if str(tok).lower() in ("*", "attention", "attn"))


# --------------------------------------------------------------------------
# K+V footprint per token per attention layer
# --------------------------------------------------------------------------


def kv_cache_dim_per_layer(cfg: dict, hidden: int) -> Optional[int]:
    """K+V elements stored/read per token per attention layer. MLA models
    (GLM-5.2, Kimi-K2.x, DeepSeek-V3 family) cache one compressed latent
    (kv_lora_rank) plus a small decoupled RoPE key (qk_rope_head_dim) shared
    across heads -- no x2, unlike GQA/MHA's 2 * num_key_value_heads *
    head_dim, where K/V are genuinely separate tensors. Using the GQA
    formula for an MLA model overcounts by ~40x (confirmed against
    GLM-5.2-NVFP4); using d_model instead of the GQA-reduced head count
    overcounts by 4x for Llama-3.1-8B's 32:8 head ratio. Returns None if
    neither field is present, so callers fall back to their own default
    (2 * hidden, i.e. full MHA)."""
    cfg = effective_llm_config(cfg)
    mla_latent = get_config_field(cfg, ["kv_lora_rank", "mla_latent_dim"])
    if mla_latent is not None:
        rope_dim = get_config_field(cfg, ["qk_rope_head_dim"], 0)
        return mla_latent + rope_dim

    num_heads = get_config_field(cfg, ["num_attention_heads"])
    if num_heads is None:
        return None
    num_kv_heads = get_config_field(cfg, ["num_key_value_heads"], num_heads)
    head_dim = get_config_field(cfg, ["head_dim"], hidden // num_heads)
    return 2 * num_kv_heads * head_dim
