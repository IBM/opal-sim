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


class UnsupportedModelError(ValueError):
    """Raised whenever a model config cannot be mapped to a supported
    architecture family, or matches a family but is missing a field that
    family's math depends on. A ValueError subclass so existing `except
    ValueError` callers still catch it, but callers can also catch it
    specifically to report "this model isn't supported" rather than a
    generic parse error."""


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


# model_type -> family, for every model_type this porting effort has verified
# end-to-end against real config.json fields and (where available) published
# param counts. A model_type present here is trusted outright; a model_type
# NOT present here is rejected outright (see detect_architecture) rather than
# silently falling through to the field-presence heuristic below -- an
# unrecognized model_type is exactly the case where guessing is unsafe.
_FULL_ATTENTION_MODEL_TYPES = {
    "llama",  # Llama-3.1-8B-Instruct: dense GQA
    "granite",  # granite-3.3-8b-instruct, dense GQA (bundled model-configs)
    "qwen3_moe",  # Qwen3-30B-A3B: MoE, full attention (no MLA)
}
_MLA_MODEL_TYPES = {
    "deepseek_v3",  # DeepSeek-V3: MoE + MLA
}
_HYBRID_MAMBA_MOE_MODEL_TYPES = {
    "nemotron_h",  # Nemotron-H-8B-Base-8K, Nemotron-3-Ultra: mamba+attention(+MoE-FFN)
    "granitemoehybrid",  # Granite-4.0-H: mamba/attention + always-on MoE-FFN
    "jamba",  # Jamba: mamba+attention+MoE-FFN on a fixed period, no pattern field
}
_KNOWN_MODEL_TYPES = _FULL_ATTENTION_MODEL_TYPES | _MLA_MODEL_TYPES | _HYBRID_MAMBA_MOE_MODEL_TYPES


def detect_architecture(cfg: dict) -> str:
    """Returns one of: 'hybrid_mamba_moe', 'sparse_mla', 'mla',
    'full_attention'.

    model_type is the primary signal: if present, it MUST be in one of the
    allowlists above, or this raises UnsupportedModelError -- silently
    falling through to the field-presence heuristic for an unrecognized
    model_type is exactly how the Granite-4 'every layer is full attention'
    bug happened. Only when model_type is missing entirely (e.g. a raw
    text_config block with no top-level model_type) does the field-presence
    heuristic run unguarded, since that's a legitimate gap in the allowlist
    approach rather than a genuinely-unrecognized model."""
    cfg = effective_llm_config(cfg)
    model_type = str(cfg.get("model_type", "")).lower()
    architectures = [str(a).lower() for a in cfg.get("architectures", [])]
    tags = [model_type] + architectures

    has_mamba = (
        any("mamba" in t for t in tags)
        or get_config_field(
            cfg, ["hybrid_override_pattern", "layers_block_type", "layer_types", "mamba_d_state", "ssm_state_size"]
        )
        is not None
    )
    has_mla = get_config_field(cfg, ["kv_lora_rank", "q_lora_rank", "mla_latent_dim"]) is not None
    has_sparse_attn = get_config_field(cfg, ["index_topk", "dsa_topk", "nsa_topk", "index_n_heads"]) is not None

    if model_type:
        if model_type not in _KNOWN_MODEL_TYPES:
            raise UnsupportedModelError(
                f"detect_architecture: model_type {model_type!r} is not a recognized "
                f"architecture family. Known model_types: {sorted(_KNOWN_MODEL_TYPES)}. "
                f"If this is a genuinely new family, verify its math against real "
                f"config/weight shapes and add its model_type to the appropriate "
                f"allowlist in config_loader.py rather than letting it fall through "
                f"to a heuristic guess."
            )
        if model_type in _HYBRID_MAMBA_MOE_MODEL_TYPES:
            return "hybrid_mamba_moe"
        if model_type in _MLA_MODEL_TYPES:
            return "sparse_mla" if has_sparse_attn else "mla"
        # _FULL_ATTENTION_MODEL_TYPES: fall through to the field checks below
        # only to distinguish 'mla'/'sparse_mla' should this model_type ever
        # legitimately carry MLA fields too; today none of them do.

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
    against real configs (DeepSeek-V3, Kimi-K2.x, GLM-5.x, Nemotron-H/Ultra,
    Granite-4.0-H) to within a few percent of published total/active param
    counts.

    `active` is what LLMRooflineModel.active_params should be set to (drives
    FLOPs); `total` is the weight-footprint/capacity figure. For MoE
    configs, `.moe` carries the breakdown LLMRooflineModel.bytes_weights
    needs for batch-size-dependent weight-read bytes (see MoEParams):
    shared_active_params + moe_layers * num_experts * expert_params_per_layer
    equals total exactly, and collapses to active at a single token.
    """
    cfg = effective_llm_config(cfg)
    model_type = str(cfg.get("model_type", "")).lower()
    arch = detect_architecture(cfg)  # single source of truth for family
    num_experts = get_config_field(cfg, ["num_experts", "n_routed_experts", "num_local_experts"], 0)
    num_shared_experts = get_config_field(cfg, ["n_shared_experts", "num_shared_experts"], 0)
    num_dense_layers = get_config_field(cfg, ["num_dense_layers", "first_k_dense_replace"], 0)
    # LatentMoE (e.g. Nemotron-3-Ultra): experts operate on a projected-down
    # latent dim with a 2-matrix (up+down) shape instead of standard 3-matrix
    # SwiGLU -- confirmed against real safetensors tensor names
    moe_latent_size = cfg.get("moe_latent_size")

    hidden_size = cfg.get("hidden_size") or cfg.get("d_model")
    layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern", "layer_types"])
    # Two different hybrid conventions share these same pattern fields, and
    # the split is a per-model_type fact (not re-derived from pattern shape
    # alone, so there's one source of truth -- detect_architecture's
    # allowlists -- rather than two independent heuristics that could drift):
    #  - Nemotron-H/-Ultra (model_type nemotron_h): the pattern is exclusive
    #    per layer -- mamba XOR attention XOR MoE-FFN-only (a distinct
    #    '-'/'mlp' token marks the MoE-FFN-only layers). MoE-ness is read off
    #    the pattern itself, not num_experts/n_routed_experts alone: some
    #    transformers AutoConfig classes (e.g. NemotronHConfig) carry a
    #    nonzero n_routed_experts class default even for configs that never
    #    set it, which would otherwise misclassify a dense pattern model.
    #  - Granite-4.0-H(-Hybrid) (model_type granitemoehybrid): the pattern is
    #    only 2-way (mamba XOR attention for sequence-mixing) -- every layer,
    #    whichever type, additionally runs its own always-on MoE-FFN block
    #    afterward (confirmed against GraniteMoeHybridDecoderLayer.forward:
    #    mamba/attention then unconditionally block_sparse_moe + shared_mlp).
    #    Here num_local_experts alone is the right MoE signal, since the
    #    pattern carries no dedicated FFN-only/MoE token to read instead.
    if arch == "hybrid_mamba_moe" and model_type == "granitemoehybrid":
        moe_on_every_layer = layer_pattern is not None and num_experts > 0
        is_moe = num_experts > 0
        if layer_pattern is not None and any(
            str(tok).lower() in ("e", "moe", "-", "mlp", "ffn") for tok in layer_pattern
        ):
            raise UnsupportedModelError(
                f"estimate_params: model_type {model_type!r} is expected to use "
                f"Granite-4-Hybrid's 2-way (mamba/attention only) pattern "
                f"convention, but {layer_pattern!r} carries an explicit FFN/MoE "
                f"token -- this contradicts the assumption moe_on_every_layer "
                f"depends on for this model_type; verify which convention this "
                f"config actually uses before proceeding"
            )
    else:
        pattern_has_ffn_token = layer_pattern is not None and any(
            str(tok).lower() in ("e", "moe", "-", "mlp", "ffn") for tok in layer_pattern
        )
        if pattern_has_ffn_token:
            is_moe = any(str(tok).lower() in ("e", "moe") for tok in layer_pattern)
            moe_on_every_layer = False
        else:
            is_moe = num_experts > 0
            moe_on_every_layer = layer_pattern is not None and is_moe
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
            is_layer_ffn_only = pattern_token in ("-", "mlp", "ffn")
            if not (is_layer_mamba or is_layer_attention or is_layer_ffn_only):
                raise ValueError(
                    f"estimate_params: layer {i} has unrecognized pattern token "
                    f"{pattern_token!r} in {layer_pattern!r} -- add it to the "
                    f"mamba/attention/FFN-only alias sets instead of silently "
                    f"guessing its layer type"
                )
        elif "jamba" in model_type:
            attn_period = cfg.get("attn_layer_period", 8)
            is_layer_mamba = i % attn_period != 0
            is_layer_attention = not is_layer_mamba
        else:
            is_layer_mamba = "mamba" in model_type
            is_layer_attention = not is_layer_mamba

        if is_layer_mamba:
            # Mamba-2 state-space params: required-with-real-values for any
            # model_type in the hybrid_mamba_moe allowlist, not defaulted --
            # a silent default here is exactly the unverified-assumption
            # pattern the Granite-4 bug fell into.
            d_state = get_config_field(cfg, ["state_size", "ssm_state_size", "mamba_d_state"])
            d_conv = get_config_field(cfg, ["conv_kernel", "mamba_d_conv"])
            expand = get_config_field(cfg, ["expand", "mamba_expand"])
            if d_state is None or d_conv is None or expand is None:
                raise UnsupportedModelError(
                    f"estimate_params: layer {i} is classified as a Mamba layer "
                    f"(model_type={model_type!r}) but the config is missing one of "
                    f"state_size/ssm_state_size/mamba_d_state, conv_kernel/"
                    f"mamba_d_conv, or expand/mamba_expand -- cannot compute this "
                    f"layer's param count without these, and a default value here "
                    f"would be an unverified guess"
                )
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
            num_heads = cfg.get("num_attention_heads")
            if num_heads is None:
                raise UnsupportedModelError(
                    f"estimate_params: layer {i} is classified as an attention "
                    f"layer (model_type={model_type!r}) but the config has no "
                    f"num_attention_heads -- cannot compute this layer's param "
                    f"count without it"
                )
            num_kv = cfg.get("num_key_value_heads", num_heads)
            head_dim = cfg.get("head_dim", hidden_size // num_heads)

            q_proj = hidden_size * (num_heads * head_dim)
            kv_proj = 2 * (num_kv * head_dim) * hidden_size
            o_proj = hidden_size * hidden_size
            comm_params = q_proj + kv_proj + o_proj
        else:
            comm_params = 0  # FFN-only layer: no sequence-mixing weights

        if moe_on_every_layer:
            # Granite-4-Hybrid style: MoE-FFN runs on every layer regardless
            # of its mamba/attention sequence-mixing type (additive, not
            # exclusive) -- see moe_on_every_layer's definition above.
            current_layer_is_moe = True
        elif pattern_token is not None:
            # Nemotron-H/-Ultra style: the pattern is exclusive -- each layer
            # is exactly one of {mamba, attention, MoE-FFN}, confirmed
            # against real tensor names.
            current_layer_is_moe = is_moe and not is_layer_mamba and not is_layer_attention
        else:
            current_layer_is_moe = is_moe and (i >= num_dense_layers)
            if "jamba" in model_type:
                exp_period = cfg.get("expert_layer_period", 2)
                current_layer_is_moe = i % exp_period == 0

        if current_layer_is_moe:
            k = cfg.get("num_experts_per_tok")
            if k is None:
                raise UnsupportedModelError(
                    f"estimate_params: layer {i} is classified as MoE "
                    f"(model_type={model_type!r}) but the config has no "
                    f"num_experts_per_tok -- cannot compute active-vs-total params "
                    f"without it"
                )
            expert_size = get_config_field(cfg, ["moe_intermediate_size", "intermediate_size"])
            if expert_size is None:
                raise UnsupportedModelError(
                    f"estimate_params: layer {i} is classified as MoE "
                    f"(model_type={model_type!r}) but the config has neither "
                    f"moe_intermediate_size nor intermediate_size -- cannot compute "
                    f"expert size without it, and hidden_size*4 here would be an "
                    f"unverified guess"
                )
            router = hidden_size * num_experts
            # Granite-4-Hybrid's always-on shared_mlp is sized by
            # shared_intermediate_size, not the n_shared_experts convention
            # DeepSeek-V3/Kimi-K2.x/GLM-5.x use -- both are "always-active
            # alongside the routed experts," just parameterized differently.
            shared_intermediate_size = cfg.get("shared_intermediate_size")

            if moe_latent_size is not None:
                per_expert_params = 2 * moe_latent_size * expert_size
                latent_proj_params = 2 * hidden_size * moe_latent_size
                shared_expert_params = num_shared_experts * per_expert_params
                always_active = latent_proj_params + router + shared_expert_params
                layer_total_mlp = always_active + num_experts * per_expert_params
                layer_active_mlp = always_active + k * per_expert_params
            else:
                # standard 3-matrix SwiGLU experts on hidden_size (DeepSeek-V3/
                # Kimi-K2.x/GLM-5.x/Granite-4-Hybrid convention), plus
                # always-active shared experts/shared_mlp
                per_expert_params = 3 * hidden_size * expert_size
                if shared_intermediate_size is not None:
                    shared_expert_params = 3 * hidden_size * shared_intermediate_size
                else:
                    shared_expert_params = num_shared_experts * per_expert_params
                layer_total_mlp = num_experts * per_expert_params + shared_expert_params + router
                layer_active_mlp = k * per_expert_params + shared_expert_params + router

            if moe_on_every_layer:
                # comm_params (mamba or attention) already computed above --
                # the MoE-FFN block is additive on top of it, not instead of it
                layer_total_mlp += comm_params
                layer_active_mlp += comm_params
                comm_params = 0

            moe_layers += 1
            expert_params_per_layer = per_expert_params
            experts_per_tok_val = k
        elif pattern_token is not None and (is_layer_mamba or is_layer_attention):
            # mamba/attention layers already have their own comm_params above;
            # don't also add a generic MLP term for them
            layer_total_mlp = 0
            layer_active_mlp = 0
        else:
            intermediate_size = cfg.get("intermediate_size")
            if intermediate_size is None:
                raise UnsupportedModelError(
                    f"estimate_params: layer {i} is a plain dense MLP layer "
                    f"(model_type={model_type!r}) but the config has no "
                    f"intermediate_size -- hidden_size*4 here would be an "
                    f"unverified guess"
                )
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
    model_type = str(cfg.get("model_type", "")).lower()
    layer_pattern = get_config_field(cfg, ["layers_block_type", "hybrid_override_pattern", "layer_types"])
    num_hidden_layers = cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")
    if layer_pattern is None:
        if model_type in _HYBRID_MAMBA_MOE_MODEL_TYPES:
            if model_type == "jamba":
                # Jamba has no pattern field at all -- attention layers are
                # every attn_layer_period-th layer, same period estimate_params
                # uses (see its "jamba" in model_type branch).
                attn_period = cfg.get("attn_layer_period", 8)
                return sum(1 for i in range(num_hidden_layers) if i % attn_period == 0)
            raise UnsupportedModelError(
                f"count_attention_layers: model_type {model_type!r} is a hybrid "
                f"Mamba/attention family but the config has none of "
                f"layers_block_type/hybrid_override_pattern/layer_types -- "
                f"treating every layer as full attention here would overcount "
                f"KV-cache memory, and there's no verified per-layer pattern to "
                f"derive the real count from"
            )
        return num_hidden_layers
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
