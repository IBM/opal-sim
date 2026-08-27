# SPDX-License-Identifier: Apache-2.0
"""
OpalModel: the base "what is this model" representation -- config loading,
dimension/dtype accessors, and total/active parameter counts -- shared by
every timing engine in opal.llm_inference.

environment.py builds one plain OpalModel per simulation (self.llm_model) as
the shared model representation kvc_manager.py/vllm_worker.py read for
memory-block sizing; the timing engine itself (an LLMRooflineModel subclass,
or the legacy GPUModel) is a separate object (self.inference_engine) -- see
opal.llm_inference.inference_engine.build_inference_engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Optional

from opal.core.datatypes import STR_DTYPE_TO_BYTES


@dataclass
class MoEParams:
    """Params (not bytes) that LLMRooflineModel.bytes_weights needs to model
    MoE weight-read bytes growing with batch size. shared_active_params is
    active_params minus the expert-routed FFN term, so
    shared_active_params + moe_layers * num_experts * expert_params_per_layer
    == total_params by construction."""

    num_experts: int
    experts_per_tok: int
    moe_layers: int
    expert_params_per_layer: float
    shared_active_params: float


def _model_name_from_url(hf_url: str) -> str:
    return hf_url.rstrip("/").split("/")[-1]


def _resolve_num_attention_heads(config: dict) -> int:
    if "num_attention_heads" in config:
        return int(config["num_attention_heads"])
    if "n_head" in config:
        return int(config["n_head"])
    raise ValueError("OpalModel: cannot determine num_attention_heads for this config")


def _resolve_num_hidden_layers(config: dict) -> int:
    if "num_hidden_layers" in config:
        return int(config["num_hidden_layers"])
    if "n_layer" in config:
        return int(config["n_layer"])
    if "layers_block_type" in config or "hybrid_override_pattern" in config:
        # some hybrid configs (e.g. nemotron_h) have no num_hidden_layers at all --
        # only the per-layer type pattern, whose length is the layer count
        pattern = config.get("layers_block_type") or config.get("hybrid_override_pattern")
        return len(pattern)
    raise ValueError("OpalModel: cannot determine num_hidden_layers for this config")


def _resolve_kv_head_dim(config: dict, num_attention_heads: int) -> int:
    # code copied and adapted from config.py file from vLLM
    if "head_dim" in config:
        return config["head_dim"]
    return config["hidden_size"] // num_attention_heads


def parse_opal_model_fields(config_dict: dict, model_name: str) -> Dict[str, Any]:
    """Parses a raw (possibly VLM-wrapped) HF config.json into the kwargs
    OpalModel.__init__ accepts. Shared by OpalModel.from_config_dict and
    LLMRooflineModel.__init__ so both go through one parse instead of two
    copies of it.

    Some real configs (e.g. Kimi-K2.x-NVFP4) are multimodal wrappers: the LLM
    backbone fields live under text_config. Dims/dtype below read the
    unwrapped backbone view; config_dict itself stays raw/wrapped in the
    returned fields, since quantization_config lives at the outer level.
    """
    from opal.llm_inference.config_loader import (
        count_attention_layers,
        effective_llm_config,
        estimate_params,
        guess_bytes_per_elem,
        guess_kv_cache_bytes_per_elem,
        kv_cache_dim_per_layer as _kv_cache_dim_per_layer,
    )

    backbone_cfg = effective_llm_config(config_dict)

    vocab_size = int(backbone_cfg["vocab_size"])
    hidden_size = int(backbone_cfg["hidden_size"])
    num_attention_heads = _resolve_num_attention_heads(backbone_cfg)
    num_hidden_layers = _resolve_num_hidden_layers(backbone_cfg)
    kv_head_size = _resolve_kv_head_dim(backbone_cfg, num_attention_heads)
    num_key_value_heads = int(backbone_cfg["num_key_value_heads"])
    max_position_embeddings = int(backbone_cfg.get("max_position_embeddings"))

    # transformers >= 4.55 renamed "torch_dtype" to "dtype" in .to_dict() output
    dtype_name = backbone_cfg.get("torch_dtype") or backbone_cfg.get("dtype")
    if dtype_name not in STR_DTYPE_TO_BYTES:
        raise ValueError(f"Unknown dtype '{dtype_name}'. Supported types: {list(STR_DTYPE_TO_BYTES.keys())}")
    torch_dtype_bytes = STR_DTYPE_TO_BYTES[dtype_name]
    torch_dtype_name = dtype_name

    # prefer the config's actual quantization scheme (e.g. NVFP4 = 0.5 B/param)
    # over torch_dtype, which reflects the unquantized base dtype only
    weight_bytes_per_elem = guess_bytes_per_elem(config_dict)

    # only `attention_layers` of `num_hidden_layers` carry a growing KV cache
    # (all of them for dense models, a small subset for hybrids); KV-cache
    # precision can differ from weight precision, hence the separate guess
    attention_layers = count_attention_layers(config_dict)
    kv_cache_bytes_per_elem = guess_kv_cache_bytes_per_elem(config_dict)
    kv_dim_per_layer = _kv_cache_dim_per_layer(config_dict, hidden_size)

    pe = estimate_params(config_dict)

    return dict(
        hidden_size=hidden_size,
        active_params=pe.active,
        total_params=pe.total,
        bytes_per_elem=weight_bytes_per_elem,
        moe=pe.moe,
        name=model_name,
        config_dict=config_dict,
        vocab_size=vocab_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        kv_head_size=kv_head_size,
        max_position_embeddings=max_position_embeddings,
        torch_dtype_name=torch_dtype_name,
        torch_dtype_bytes=torch_dtype_bytes,
        attention_layers=attention_layers,
        kv_cache_bytes_per_elem=kv_cache_bytes_per_elem,
        kv_cache_dim_per_layer=kv_dim_per_layer,
    )


class OpalModel:
    """The base "what is this model" representation: config loading,
    dimension/dtype/KV-cache accessors, and total/active parameter counts.

    Construct via from_huggingface/from_config_dir/from_config_dict (parses a
    real config.json, see parse_opal_model_fields), or directly with just
    hidden_size/active_params -- every other field defaults to None and is
    unused by the roofline math (opal's own tests build models this way to
    exercise formulas against clean illustrative dimensions)."""

    def __init__(
        self,
        *,
        hidden_size: int,
        active_params: float,
        bytes_per_elem: float = 2.0,
        name: Optional[str] = None,
        total_params: Optional[float] = None,
        moe: Optional[MoEParams] = None,
        config_dict: Optional[dict] = None,
        vocab_size: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        kv_head_size: Optional[int] = None,
        max_position_embeddings: Optional[int] = None,
        torch_dtype_name: Optional[str] = None,
        torch_dtype_bytes: Optional[float] = None,
        attention_layers: Optional[int] = None,
        kv_cache_bytes_per_elem: Optional[float] = None,
        kv_cache_dim_per_layer: Optional[int] = None,
    ):
        self.logger = logging.getLogger(type(self).__name__)
        self.config_dict = config_dict
        self.name = name
        self.model_name = name
        self.hidden_size = hidden_size
        self.active_params = active_params
        self.total_params = total_params if total_params is not None else active_params
        self.weight_bytes_per_elem = bytes_per_elem
        self.moe = moe
        self.vocab_size = vocab_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.kv_head_size = kv_head_size
        self.max_position_embeddings = max_position_embeddings
        self.torch_dtype_name = torch_dtype_name
        self.torch_dtype_bytes = torch_dtype_bytes
        self.attention_layers = attention_layers
        self.kv_cache_bytes_per_elem = kv_cache_bytes_per_elem
        self.kv_cache_dim_per_layer = kv_cache_dim_per_layer
        # plain multiply, not bit-shift: key_value_bytes can be fractional
        # under sub-byte KV-cache quantization
        if attention_layers is not None and kv_cache_bytes_per_elem is not None:
            dim = kv_cache_dim_per_layer if kv_cache_dim_per_layer is not None else 2 * hidden_size
            self.key_value_bytes = dim * attention_layers * kv_cache_bytes_per_elem
        else:
            self.key_value_bytes = None

    # ---- construction from a real config.json ----

    @classmethod
    def from_huggingface(cls, hf_url: str) -> "OpalModel":
        from transformers import AutoConfig

        hg_config = AutoConfig.from_pretrained(hf_url)
        config_dict = hg_config.to_dict()
        model_name = _model_name_from_url(hf_url)
        config_dict["model_name"] = model_name
        return cls.from_config_dict(config_dict, model_name)

    @classmethod
    def from_config_dir(cls, config_dir: str) -> "OpalModel":
        config_file = Path(config_dir) / "config.json"
        with open(config_file) as f:
            config_dict = json.load(f)
        if "model_name" not in config_dict:
            config_dict["model_name"] = _model_name_from_url(str(config_dir))
        return cls.from_config_dict(config_dict, config_dict["model_name"])

    @classmethod
    def from_config_dict(cls, config_dict: dict, model_name: str) -> "OpalModel":
        return cls(**parse_opal_model_fields(config_dict, model_name))

    # ---- basic accessors ----

    def get_kvc_bytes(self, tokens: int) -> int:
        """Size in bytes of the KV cache for `tokens` tokens."""
        return tokens * self.key_value_bytes

    def get_kvc_tokens(self, size_bytes: int) -> int:
        """Number of tokens that fit in `size_bytes` of KV cache."""
        return size_bytes // self.key_value_bytes

    def get_model_params(self) -> float:
        return self.total_params

    def get_active_params(self) -> float:
        return self.active_params

    def get_model_name(self) -> str:
        return self.model_name

    def __str__(self):
        if self.config_dict is not None:
            return pformat(self.config_dict)
        return (
            f"{type(self).__name__}(name={self.name!r}, hidden_size={self.hidden_size}, "
            f"active_params={self.active_params:,.0f})"
        )

    def toJSON(self):
        return json.dumps(self.config_dict, indent=4)

    # ---- timing interface, implemented by LLMRooflineModel / GPUModel ----

    def estimate(self, batch) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not implement estimate()")


def get_config_folder_for_model(config_dir: str, model_name: str):
    root_path = Path(config_dir)
    print(f"Scanning model/model_params/config_dir {config_dir} for the model {model_name}")
    # rglob("*") recursively finds all files and folders
    for path in root_path.rglob("*"):
        if path.is_dir():
            print(f"\t...processing location: {path.resolve()}")
            if model_name in path.parts:
                # we found a match, return
                return path
    print(f"WARNING: no suitable local folder found in {config_dir} for the model {model_name}")
    return None


def get_model(
    *,
    name: str | None = None,
    config_dir: str | None = None,
    hf_url: str | None = None,
) -> OpalModel:
    if config_dir is not None and hf_url is not None:
        raise ValueError(
            f"Provide exactly one config source either hf_url {hf_url} or config_dir {config_dir}, not both."
        )

    if hf_url is not None:
        return OpalModel.from_huggingface(hf_url)
    else:
        return OpalModel.from_config_dir(get_config_folder_for_model(config_dir, model_name=name))
