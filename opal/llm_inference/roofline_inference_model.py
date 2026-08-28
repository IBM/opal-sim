# SPDX-License-Identifier: Apache-2.0
"""
Roofline timing models for different LLM attention/backbone architectures,
sharing common logic in a base class and specializing only what differs.

Hierarchy:
    OpalModel
     +-- LLMRooflineModel (ABC)          -- common: dense/MoE FFN flops+bytes,
          |                                 weight-read bytes, roofline combine
          +-- FullAttentionRooflineModel     -- dense causal attention
          |    +-- MLAAttentionRooflineModel     -- KV cache sized by MLA latent dim
          |         +-- SparseAttentionRooflineModel -- + top-k sparse attention (DSA)
          +-- HybridMambaMoERooflineModel    -- + Mamba-2 scan, few dense attn layers

ModelFactory (inference_model_factory.py) builds one of the four concrete
classes directly from a real config.json; detect_architecture() picks the
family. Models operate on opal.vllm_worker.BatchMetadata (prefill_requests/
decode_requests + request_tokens).

`a`/`b`: `a` (default 4.0) is the FLOP cost of one QK^T+softmax*V pairwise
interaction, matching the legacy GPUModel's `worker.inference_params.a`. `b`
reproduces that engine's fixed-shape `b * d_model^2` dense/FFN term when
explicitly requested; left unset, the dense/FFN term uses
`2 * active_params` instead, driven by real config dimensions.

MoE weight-read bytes depend on batch composition: different tokens route to
different experts, so bytes read grows from ~experts_per_tok at batch size 1
toward num_experts as the batch grows. MoEParams + bytes_weights below model
that with a coupon-collector union-bound estimate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from opal.llm_inference.opal_model import OpalModel, parse_opal_model_fields

if TYPE_CHECKING:
    # unguarded import would be circular once vllm_worker imports this module
    from opal.worker.vllm_worker import BatchMetadata


# --------------------------------------------------------------------------
# Shared data structures
# --------------------------------------------------------------------------


@dataclass
class GPUConfig:
    """GPU compute/bandwidth roofline parameters. Peak FLOPS is dtype-specific
    (e.g. H100-class cards are ~2x fp8 TFLOPS vs fp16) -- peak_flops_for picks
    the right one using the same 4.0/2.0/1.0/0.5 bytes-per-element convention
    as config_loader.guess_bytes_per_elem. peak_flops_fp*/peak_bandwidth are
    expected to already be pre-scaled by tp; callers don't multiply again."""

    num_gpus: int = 1
    tp: int = 1
    peak_flops_fp32: float = 0.0
    peak_flops_fp16: float = 0.0
    peak_flops_fp8: float = 0.0
    peak_flops_fp4: float = 0.0
    peak_bandwidth: float = 0.0
    compute_efficiency: float = 0.4
    memory_efficiency: float = 0.6

    def peak_flops_for(self, bytes_per_elem: float) -> float:
        if bytes_per_elem >= 4.0:
            return self.peak_flops_fp32
        if bytes_per_elem >= 2.0:
            return self.peak_flops_fp16
        if bytes_per_elem >= 1.0:
            return self.peak_flops_fp8
        return self.peak_flops_fp4


def _expected_distinct_experts(num_tokens: int, num_experts: int, experts_per_tok: int) -> float:
    """Expected distinct experts touched per MoE layer when num_tokens
    independent tokens each pick experts_per_tok of num_experts uniformly at
    random (coupon-collector union bound)."""
    if num_tokens <= 0 or num_experts <= 0:
        return 0.0
    if experts_per_tok >= num_experts:
        return float(num_experts)
    p_miss_one_token = 1.0 - (experts_per_tok / num_experts)
    return num_experts * (1.0 - p_miss_one_token**num_tokens)


# --------------------------------------------------------------------------
# BatchMetadata aggregation helpers -- mirror the (n, p) / context-length
# semantics vllm_worker's scheduler already uses: prefill n = prompt_processed
# + tokens_this_step, p = prompt_processed (prefix-cache-covered); decode
# context = prompt_tokens + decode_tokens_generated.
# --------------------------------------------------------------------------


def _num_prefill_new_tokens(batch: BatchMetadata) -> int:
    return sum(batch.request_tokens.get(r.request_id, 0) for r in batch.prefill_requests)


def _sum_prefill_pairwise(batch: BatchMetadata) -> int:
    """sum of n*(n+1) - p*(p+1) across prefill requests: quadratic pairwise
    interaction count, net of pairs already covered by a cached prefix."""
    total = 0
    for r in batch.prefill_requests:
        tokens = batch.request_tokens.get(r.request_id, 0)
        n = r.prompt_processed + tokens
        p = r.prompt_processed
        total += n * (n + 1) - p * (p + 1)
    return total


def _sum_decode_context(batch: BatchMetadata) -> int:
    return sum(r.prompt_tokens + r.decode_tokens_generated for r in batch.decode_requests)


def _num_decode_requests(batch: BatchMetadata) -> int:
    return len(batch.decode_requests)


# --------------------------------------------------------------------------
# Base class: everything shared across architectures
# --------------------------------------------------------------------------


class LLMRooflineModel(OpalModel, ABC):
    """Common roofline logic on top of OpalModel's config/dimension/param
    accessors. Subclasses specialize only the attention FLOP/byte formulas
    (abstract methods below) and, optionally, add extra cost via the
    flops_extra/bytes_extra hooks (Mamba-2 scan, sparse-attention indexer)."""

    def __init__(
        self,
        config_dict: dict,
        model_name: str,
        hw: GPUConfig,
        b: Optional[float] = None,
        n_layers: Optional[int] = None,
    ):
        super().__init__(**parse_opal_model_fields(config_dict, model_name))
        self.hw = hw
        self.b = b  # legacy dense-FLOP coefficient override (None -> use active_params)
        self.n_layers = n_layers  # total hidden-layer count; only required when `b` is set

    # ---- roofline-notation aliases for OpalModel's config-layer fields ----

    @property
    def d_model(self) -> int:
        return self.hidden_size

    @property
    def bytes_per_elem(self) -> float:
        return self.weight_bytes_per_elem

    # ---- common dense/MoE FFN + weight-read terms ----

    def flops_dense(self, batch: BatchMetadata) -> float:
        # prefix tokens already ran through every dense/FFN layer previously;
        # only new prefill tokens + decode tokens need this compute
        new_tokens = _num_prefill_new_tokens(batch) + _num_decode_requests(batch)
        if self.b is not None:
            if self.n_layers is None:
                raise ValueError("LLMRooflineModel: 'b' legacy dense-FLOP override requires 'n_layers' to also be set")
            return self.b * self.n_layers * (self.d_model**2) * new_tokens
        return 2 * self.active_params * new_tokens

    def bytes_weights(self, batch: BatchMetadata) -> float:
        if self.moe is None:
            return self.active_params * self.bytes_per_elem
        num_tokens = _num_prefill_new_tokens(batch) + _num_decode_requests(batch)
        expected_experts = _expected_distinct_experts(num_tokens, self.moe.num_experts, self.moe.experts_per_tok)
        moe_bytes = self.moe.moe_layers * expected_experts * self.moe.expert_params_per_layer * self.bytes_per_elem
        shared_bytes = self.moe.shared_active_params * self.bytes_per_elem
        return shared_bytes + moe_bytes

    # ---- architecture-specific attention: must be implemented by subclasses ----

    @abstractmethod
    def flops_attention_prefill(self, batch: BatchMetadata) -> float: ...

    @abstractmethod
    def flops_attention_decode(self, batch: BatchMetadata) -> float: ...

    @abstractmethod
    def bytes_kv_write(self, batch: BatchMetadata) -> float: ...

    @abstractmethod
    def bytes_kv_read(self, batch: BatchMetadata) -> float: ...

    # ---- extension hooks: no-ops by default ----

    def flops_extra(self, batch: BatchMetadata) -> float:
        return 0.0

    def bytes_extra(self, batch: BatchMetadata) -> float:
        return 0.0

    # ---- totals + roofline combination ----

    def total_flops(self, batch: BatchMetadata) -> float:
        return (
            self.flops_dense(batch)
            + self.flops_attention_prefill(batch)
            + self.flops_attention_decode(batch)
            + self.flops_extra(batch)
        )

    def total_bytes(self, batch: BatchMetadata) -> float:
        return (
            self.bytes_weights(batch) + self.bytes_kv_write(batch) + self.bytes_kv_read(batch) + self.bytes_extra(batch)
        )

    def estimate(self, batch: BatchMetadata) -> dict:
        flops = self.total_flops(batch)
        nbytes = self.total_bytes(batch)
        peak_flops = self.hw.peak_flops_for(self.weight_bytes_per_elem)
        time_compute = flops / (self.hw.compute_efficiency * peak_flops)
        time_memory = nbytes / (self.hw.memory_efficiency * self.hw.peak_bandwidth)
        time_s = max(time_compute, time_memory)
        return {
            "flops": flops,
            "bytes": nbytes,
            "t_compute_s": time_compute,
            "t_memory_s": time_memory,
            "time_s": time_s,
        }


# --------------------------------------------------------------------------
# Full (dense) causal attention: quadratic prefill, linear decode
# --------------------------------------------------------------------------


class FullAttentionRooflineModel(LLMRooflineModel):
    """Standard dense causal attention over all prior tokens. kv_cache_dim is
    the per-token K+V size stored/read (2 * d_model for MHA/GQA-at-full-width,
    overridden to a small latent dim by MLA)."""

    def __init__(
        self,
        config_dict: dict,
        model_name: str,
        hw: GPUConfig,
        n_attn_layers: int,
        kv_cache_dim: Optional[int] = None,
        a: float = 4.0,
        b: Optional[float] = None,
        n_layers: Optional[int] = None,
    ):
        super().__init__(config_dict, model_name, hw, b=b, n_layers=n_layers)
        self.n_attn_layers = n_attn_layers
        # default: OpalModel's config-derived GQA-aware per-layer K+V
        # footprint, else full K+V per token per layer = 2 * d_model
        if kv_cache_dim is not None:
            self.kv_cache_dim = kv_cache_dim
        elif self.kv_cache_dim_per_layer is not None:
            self.kv_cache_dim = self.kv_cache_dim_per_layer
        else:
            self.kv_cache_dim = 2 * self.d_model
        self.a = a

    def flops_attention_prefill(self, batch: BatchMetadata) -> float:
        # n*(n+1) - p*(p+1): skips pairs already covered by a cached prefix
        return self.a * self.n_attn_layers * self.d_model * _sum_prefill_pairwise(batch)

    def flops_attention_decode(self, batch: BatchMetadata) -> float:
        return self.a * self.n_attn_layers * self.d_model * _sum_decode_context(batch)

    def bytes_kv_write(self, batch: BatchMetadata) -> float:
        new_tokens = _num_prefill_new_tokens(batch) + _num_decode_requests(batch)
        return self.n_attn_layers * self.kv_cache_dim * self.bytes_per_elem * new_tokens

    def bytes_kv_read(self, batch: BatchMetadata) -> float:
        return self.n_attn_layers * self.kv_cache_dim * self.bytes_per_elem * _sum_decode_context(batch)


# --------------------------------------------------------------------------
# MLA specialization: same access pattern, much smaller KV cache footprint
# --------------------------------------------------------------------------


class MLAAttentionRooflineModel(FullAttentionRooflineModel):
    """Multi-head Latent Attention: K/V compressed into a small latent vector
    per token. Only the KV byte footprint changes; FLOP shape is unchanged
    since MLA still attends over the full prior context."""

    def __init__(
        self,
        config_dict: dict,
        model_name: str,
        hw: GPUConfig,
        n_attn_layers: int,
        mla_latent_dim: int,
        a: float = 4.0,
        b: Optional[float] = None,
        n_layers: Optional[int] = None,
    ):
        super().__init__(
            config_dict, model_name, hw, n_attn_layers, kv_cache_dim=mla_latent_dim, a=a, b=b, n_layers=n_layers
        )
        self.mla_latent_dim = mla_latent_dim


# --------------------------------------------------------------------------
# Sparse attention (DSA-style) specialization: linear-in-C, top-k budget
# --------------------------------------------------------------------------


class SparseAttentionRooflineModel(MLAAttentionRooflineModel):
    """MLA + top-k sparse attention (DeepSeek Sparse Attention / GLM
    IndexShare): each query attends to only `topk` selected prior tokens, so
    attention FLOPs/bytes become linear rather than quadratic. Indexer cost
    is amortized every indexer_share_period layers."""

    def __init__(
        self,
        config_dict: dict,
        model_name: str,
        hw: GPUConfig,
        n_attn_layers: int,
        mla_latent_dim: int,
        topk: int,
        indexer_flops_per_token: float,
        indexer_share_period: int = 1,
        a: float = 4.0,
        b: Optional[float] = None,
        n_layers: Optional[int] = None,
    ):
        super().__init__(config_dict, model_name, hw, n_attn_layers, mla_latent_dim, a=a, b=b, n_layers=n_layers)
        self.topk = topk
        self.indexer_flops_per_token = indexer_flops_per_token
        self.indexer_share_period = max(1, indexer_share_period)

    def flops_attention_prefill(self, batch: BatchMetadata) -> float:
        # linear in new tokens only: cached-prefix tokens' top-k selection is reused
        new_tokens = _num_prefill_new_tokens(batch)
        return self.a * self.n_attn_layers * self.d_model * self.topk * new_tokens

    def flops_attention_decode(self, batch: BatchMetadata) -> float:
        return self.a * self.n_attn_layers * self.d_model * self.topk * _num_decode_requests(batch)

    def bytes_kv_write(self, batch: BatchMetadata) -> float:
        new_tokens = _num_prefill_new_tokens(batch) + _num_decode_requests(batch)
        return self.n_attn_layers * self.kv_cache_dim * self.bytes_per_elem * new_tokens

    def bytes_kv_read(self, batch: BatchMetadata) -> float:
        # only the selected top-k entries are read back, not the full context
        return self.n_attn_layers * self.kv_cache_dim * self.bytes_per_elem * self.topk * _num_decode_requests(batch)

    def flops_extra(self, batch: BatchMetadata) -> float:
        # indexer runs once every indexer_share_period layers, reused by the rest
        n_indexer_runs = max(1, self.n_attn_layers // self.indexer_share_period)
        new_tokens = _num_prefill_new_tokens(batch) + _num_decode_requests(batch)
        return n_indexer_runs * self.indexer_flops_per_token * new_tokens


# --------------------------------------------------------------------------
# Hybrid Mamba-2 + MoE + (small) dense attention specialization
# --------------------------------------------------------------------------


class HybridMambaMoERooflineModel(FullAttentionRooflineModel):
    """Adds Mamba-2 selective-scan cost via flops_extra/bytes_extra, on top
    of a small number of dense attention layers reused unchanged from
    FullAttentionRooflineModel."""

    def __init__(
        self,
        config_dict: dict,
        model_name: str,
        hw: GPUConfig,
        n_attn_layers: int,
        n_mamba_layers: int,
        d_state: int,
        scan_const: float = 3.0,
        kv_cache_dim: Optional[int] = None,
        a: float = 4.0,
        b: Optional[float] = None,
        n_layers: Optional[int] = None,
    ):
        super().__init__(
            config_dict, model_name, hw, n_attn_layers, kv_cache_dim=kv_cache_dim, a=a, b=b, n_layers=n_layers
        )
        self.n_mamba_layers = n_mamba_layers
        self.d_state = d_state
        self.scan_const = scan_const

    def flops_extra(self, batch: BatchMetadata) -> float:
        # a cached prefix's recurrent state carries forward; only new tokens need the scan
        prefill_term = (
            self.n_mamba_layers * self.scan_const * self.d_model * self.d_state * _num_prefill_new_tokens(batch)
        )
        decode_term = self.n_mamba_layers * self.scan_const * self.d_model * self.d_state * _num_decode_requests(batch)
        return prefill_term + decode_term

    def bytes_extra(self, batch: BatchMetadata) -> float:
        # recurrent state read/write per decode step; prefill's state stays resident
        return self.n_mamba_layers * self.d_state * self.d_model * self.bytes_per_elem * _num_decode_requests(batch)
