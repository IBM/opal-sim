# LLM Model Roofline Analysis

How `opal.llm_inference` turns a Hugging Face `config.json` into total/active
parameter counts, FLOPs-per-token, KV-cache size, and (for hybrid models) SSM
state size -- conceptually, with pointers to the exact Python that implements
each piece. This is the *model math* doc; for *why* the package is shaped
this way (migration rationale from the internal fork), see
`wiki/LLM-Inference-Roofline-Migration-Plan.md`.

All formulas here are estimates from config dimensions alone (no weight
download) -- validated to within a few percent of published total/active
param counts for one real model per architecture family, see
`tests/test_hf_model_configs.py`. Every worked example below uses a real
model's `config.json`, with numbers taken directly from a live run of
`OpalModel.from_huggingface(...)`. Configs that don't cleanly map to a
verified family, or are missing a field that family's math depends on, raise
`UnsupportedModelError` rather than silently guessing -- see
`tests/test_unsupported_models.py` for the negative-path tests that pin this
down.

## Table of contents

1. [Where the math lives](#1-where-the-math-lives)
2. [The four families, at a glance](#2-the-four-families-at-a-glance)
3. [Core idea: total vs. active params](#3-core-idea-total-vs-active-params)
4. [Strict validation: `UnsupportedModelError`](#4-strict-validation-unsupportedmodelerror)
5. [Dense models (Llama-style GQA)](#5-dense-models-llama-style-gqa)
6. [MoE (DeepSeek-V3 / Kimi-K2.x / GLM-5.x style)](#6-moe-deepseek-v3--kimi-k2x--glm-5x-style)
7. [MLA (Multi-head Latent Attention)](#7-mla-multi-head-latent-attention)
8. [Sparse attention (DSA / IndexShare)](#8-sparse-attention-dsa--indexshare)
9. [Hybrid Mamba (+MoE, +attention)](#9-hybrid-mamba-moe-attention)
10. [Dtype / quantization bytes-per-element](#10-dtype--quantization-bytes-per-element)
11. [Multimodal (VLM) wrapper configs](#11-multimodal-vlm-wrapper-configs)
12. [Summary: five worked models side by side](#12-summary-five-worked-models-side-by-side)

## 1. Where the math lives

- `opal/llm_inference/config_loader.py`
  - `detect_architecture(cfg)` (line 142) -- which family a config belongs to
  - `estimate_params(cfg)` (line 208) -- total/active param count, dense/MoE/MLA/hybrid
  - `count_attention_layers(cfg)` (line 510) -- how many layers carry a growing KV cache
  - `kv_cache_dim_per_layer(cfg, hidden)` (line 545) -- K+V elements per token per attention layer
  - `UnsupportedModelError` (line 26) -- raised instead of silently guessing when a config doesn't match a verified family or is missing a field that family's math needs
- `opal/llm_inference/opal_model.py`
  - `OpalModel` (line 145) -- wraps the above into one object (`total_params`,
    `active_params`, `key_value_bytes`, etc.)
- `opal/llm_inference/roofline_inference_model.py`
  - `LLMRooflineModel` (line 127) and its subclasses -- turn those static
    quantities into FLOPs/bytes *per batch*, then into a wall-clock time
    estimate via the roofline model (`max(compute time, memory time)`)
- `opal/llm_inference/inference_model_factory.py`
  - `ModelFactory` -- picks the concrete `LLMRooflineModel` subclass for a
    config and threads through the architecture-specific extras
    (`n_attn_layers`, `mla_latent_dim`, `topk`, `d_state`, ...) that
    `OpalModel`'s generic parse doesn't derive on its own

## 2. The four families, at a glance

| Family | `model_type` examples | Total vs. active params | KV cache / token | Extra per-token state |
|---|---|---|---|---|
| **Dense (GQA)** | `llama`, `granite` | equal | `2 * num_key_value_heads * head_dim` | none |
| **MoE** | `qwen3_moe`, (+ MLA: `deepseek_v3`) | total > active (all experts vs. `experts_per_tok`) | same as underlying attention type (GQA or MLA) | none |
| **MLA** | `deepseek_v3` | equal, or > if also MoE | `kv_lora_rank + qk_rope_head_dim` (no `2*`: K/V share one latent) | none |
| **Hybrid Mamba+MoE+attention** | `nemotron_h`, `granitemoehybrid`, `jamba` | equal, or > if MoE-FFN is present | GQA formula, but **only** on pattern-marked attention layers | fixed-size recurrent state on Mamba layers (`d_state * d_inner`), independent of context length |

Notes on reading this table:

- **MoE is an orthogonal axis, not a family of its own.** A model is MoE if
  it has MoE hallmark fields (`num_experts`/`n_routed_experts`, etc.);
  separately, its attention is either plain GQA (`full_attention`) or MLA.
  DeepSeek-V3 is both MoE *and* MLA at once.
- **"Total vs. active" only diverges where routing exists.** Dense and
  hybrid-without-MoE-tokens models have `total == active`; anything with a
  MoE-FFN component does not.
- **KV cache size depends only on the attention mechanism**, not on MoE.
  Switching a model from dense-FFN to MoE-FFN never changes its KV-cache
  formula.
- **Hybrid is the only family where per-layer type varies within one
  model** -- see [section 9](#9-hybrid-mamba-moe-attention) for the pattern
  string that decides this layer by layer.

## 3. Core idea: total vs. active params

Every model family distinguishes:

- **total params**: everything stored in weights (drives memory footprint).
- **active params**: everything touched computing one token (drives FLOPs).

For a dense model these are equal. For MoE they diverge: total counts every
expert, active counts only the `experts_per_tok` that actually fire per
token (plus always-on shared experts). This split is `ParamEstimate.total` /
`ParamEstimate.active` (`config_loader.py:202`), and becomes
`OpalModel.total_params` / `OpalModel.active_params` via
`parse_opal_model_fields` (`opal_model.py:73`).

`estimate_params()` (`config_loader.py:208`) computes both in one pass:
embeddings + per-layer transformer terms + (optionally untied) output head,
calling `detect_architecture(cfg)` once up front as the single source of
truth for which family's per-layer math applies, then looping over
`num_layers` accumulating `total_transformer_params`/
`active_transformer_params` layer by layer.

```
dense model:      total ========================= active   (one bar, same size)

MoE model:        total  [ E0 ][ E1 ][ E2 ]...[E127][shared][router]
                   active           [E_k1]...[E_k8] [shared][router]
                          (only experts_per_tok=8 of 128 experts fire per token)
```

## 4. Strict validation: `UnsupportedModelError`

`detect_architecture(cfg)` (`config_loader.py:142`) checks `cfg["model_type"]`
against per-family allowlists *first*:

```
_FULL_ATTENTION_MODEL_TYPES      = {"llama", "granite", "qwen3_moe"}
_MLA_MODEL_TYPES                 = {"deepseek_v3"}
_HYBRID_MAMBA_MOE_MODEL_TYPES    = {"nemotron_h", "granitemoehybrid", "jamba"}
```

- `model_type` present and in a family's allowlist -> route directly to that
  family.
- `model_type` present but in none of them -> raise `UnsupportedModelError`
  naming the unrecognized type, rather than falling through to a
  field-presence guess.
- `model_type` absent entirely -> fall back to the original field-presence
  heuristic (`has_mamba`/`has_mla`/`has_sparse_attn`) as a secondary signal,
  since some configs (e.g. a bare `text_config` block) legitimately omit it.

This allowlist is deliberately just the `model_type`s this porting effort
has verified end-to-end against a real published param count -- adding a new
model means adding it here *and* to a worked example/test, not just hoping
the field-presence heuristic happens to guess right.

Beyond family detection, every per-family branch in `estimate_params()` and
`ModelFactory` asserts the specific fields its formula depends on, and
raises `UnsupportedModelError` instead of silently defaulting when one is
missing. Concretely, no code path in this pipeline falls back to a made-up
constant for:

| Missing field | Old silent default (removed) | Where |
|---|---|---|
| `state_size`/`ssm_state_size`/`mamba_d_state` | `16` | Mamba branch, `estimate_params` |
| `conv_kernel`/`mamba_d_conv` | `4` | Mamba branch, `estimate_params` |
| `expand`/`mamba_expand` | `2` | Mamba branch, `estimate_params` |
| `num_attention_heads` | `1` | attention branch, `estimate_params` |
| `num_experts_per_tok` | `1` | MoE branch, `estimate_params` |
| `moe_intermediate_size`/`intermediate_size` | `hidden_size * 4` | MoE branch, `estimate_params` |
| `intermediate_size` (dense) | `hidden_size * 4` | dense-MLP branch, `estimate_params` |
| `hidden_size`/`d_model`/`n_embd` | `4096` | `ModelFactory.create_from_config` |
| `num_hidden_layers`/pattern length | `32` | `ModelFactory.create_from_config` |
| mamba/attention layer split | `round(n_layers * 0.1)` | `ModelFactory._build_hybrid_mamba_moe` |
| `mamba_d_state`/`ssm_state_size`/`state_size` | `128` | `ModelFactory._build_hybrid_mamba_moe` |
| `kv_lora_rank`/`mla_latent_dim` | `512` | `ModelFactory._build_mla`/`_build_sparse_mla` |
| `index_topk`/`dsa_topk`/`nsa_topk` | `2048` | `ModelFactory._build_sparse_mla` |
| indexer share period | `4` | `ModelFactory._build_sparse_mla` |
| indexer FLOPs/token | `2 * hidden * 256` | `ModelFactory._build_sparse_mla` |

The one deliberate exception: `HybridMambaMoERooflineModel`'s `scan_const`
(Mamba-2 selective-scan FLOPs-per-element) keeps a real default of `3.0` in
the model class's own constructor (`roofline_inference_model.py:363`) rather
than raising. This isn't a stand-in for a missing *config* value -- no real
Mamba-2 `config.json` carries a scan-cost constant -- it's a genuine
roofline-model constant, only overridable via an explicit `mamba_scan_const`
config field if a future model needs a different one.

`count_attention_layers()` (`config_loader.py:510`) similarly raises
`UnsupportedModelError` if a hybrid `model_type` has no layer-type pattern
field at all, *except* for `jamba`, which has no pattern field in any real
config and instead derives its attention layers from `attn_layer_period`
(every `attn_layer_period`-th layer is attention) -- the one hybrid
`model_type` where "no pattern" is expected, not a sign of a malformed
config.

`tests/test_unsupported_models.py` exercises all of this with synthetic
(non-network) configs: unrecognized `model_type`, a hybrid config missing
Mamba-2 fields, MoE configs missing `num_experts_per_tok`/expert size, dense
configs missing `intermediate_size`/`num_attention_heads`, an MLA config
missing its latent-dim field, a hybrid config with no pattern and no
explicit layer counts, and a Granite-4-style pattern that contradicts the
2-way mamba/attention convention (see [section 9](#9-hybrid-mamba-moe-attention)).

## 5. Dense models (Llama-style GQA)

Architecture tag: `detect_architecture()` returns `"full_attention"` for
`model_type in {"llama", "granite", "qwen3_moe"}`, or (if `model_type` is
absent) when none of the MoE/MLA/sparse/Mamba hallmark fields are present.

### 5.1 FLOPs per token

Per layer, attention/communication params:

```
q_proj  = hidden_size * (num_attention_heads * head_dim)
kv_proj = 2 * (num_key_value_heads * head_dim) * hidden_size   # GQA: fewer KV heads than Q heads
o_proj  = hidden_size * hidden_size
```

FFN params, standard 3-matrix SwiGLU:

```
mlp_params = 3 * hidden_size * intermediate_size   # gate + up + down projections
```

GQA halves (or more) the KV projection's cost relative to full MHA by
sharing each KV head across several Q heads:

```
Q heads:   [h0][h1][h2][h3] [h4][h5][h6][h7] ...   (32 heads, Llama-3.1-8B)
            \  |  |  / \  |  |  /
KV heads:    [ kv0  ]   [ kv1  ]   ...              (8 heads -- each shared by 4 Q heads)
```

`FullAttentionRooflineModel` (`roofline_inference_model.py:235`) turns this
into per-batch FLOPs. Prefill attention FLOPs are quadratic in context
length (`a * n_attn_layers * d_model * sum(n*(n+1) - p*(p+1))` -- the
`n*(n+1)-p*(p+1)` term nets out pairs already covered by a cached prefix,
see `_sum_prefill_pairwise`, `roofline_inference_model.py:102`); decode
attention FLOPs are linear in context length per new token. The dense/FFN
term itself is `2 * active_params * new_tokens` -- the standard "2 FLOPs per
active parameter per token" approximation, i.e. **FLOPs/token (dense-FFN
part) = 2 * active_params**, with the attention term added on top and scaled
by current context length.

### 5.2 Memory requirements

Total params = embeddings + sum over layers of (attention + FFN + 2*hidden_size
for norms) + output head (skipped if `tie_word_embeddings`). Since every
weight fires on every token, `total_params == active_params`.

KV cache per token per attention layer: `2 * num_key_value_heads * head_dim`
elements (the `2*` is because K and V are genuinely separate tensors under
GQA/MHA). `kv_cache_dim_per_layer()` (`config_loader.py:545`) returns this
value; `OpalModel.__init__` (`opal_model.py:155`) turns it into bytes:

```
key_value_bytes = kv_cache_dim_per_layer * attention_layers * kv_cache_bytes_per_elem
```

`count_attention_layers()` (`config_loader.py:510`) is what
`attention_layers` comes from -- for a dense model, every layer. No
additional per-token state beyond the KV cache; there is no SSM/recurrent
component in this family.

### 5.3 Worked example: Llama-3.1-8B-Instruct

Config: [`meta-llama/Llama-3.1-8B-Instruct/config.json`](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json)
(relevant fields only):

```json
{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "vocab_size": 128256,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16"
}
```

```
detect_architecture()  ->  "full_attention"   (model_type "llama" in allowlist)

total_params  = active_params  =  8,030,257,152  (~8.03B)   <- matches the
                                                                8.03B Meta
                                                                model card
                                                                (test tolerance: 2%)

key_value_bytes/token = 2 * 8 kv_heads * 128 head_dim * 2 bytes (bf16) *
                         32 layers = 131,072 bytes (~128 KiB)
```

Since `total_params == active_params`, this is the simplest case: no
routing, no expert selection, every parameter fires on every token. This
matches `OpalModel.key_value_bytes` and
`tests/test_hf_model_configs.py::TestDenseModel::test_kv_cache_per_token`.

## 6. MoE (DeepSeek-V3 / Kimi-K2.x / GLM-5.x style)

Hallmark fields: `num_experts`/`n_routed_experts`/`num_local_experts` (via
`get_config_field`, `config_loader.py:40`). MoE is an orthogonal axis to
attention type -- `qwen3_moe` is MoE + plain GQA; `deepseek_v3` is MoE + MLA
([section 7](#7-mla-multi-head-latent-attention)).

### 6.1 FLOPs per token

Per MoE layer, standard 3-matrix SwiGLU experts, plus always-active shared
experts and a router:

```
per_expert_params    = 3 * hidden_size * expert_size
shared_expert_params = num_shared_experts * per_expert_params
router                = hidden_size * num_experts

layer_total_mlp  = num_experts * per_expert_params + shared_expert_params + router   # all experts (capacity)
layer_active_mlp = experts_per_tok * per_expert_params + shared_expert_params + router  # only the ones that fire
```

`expert_size` is required to be present as either `moe_intermediate_size` or
`intermediate_size` (raises `UnsupportedModelError` otherwise -- see
[section 4](#4-strict-validation-unsupportedmodelerror)); there is no
`hidden_size * 4` fallback.

`first_k_dense_replace`/`num_dense_layers` lets early layers stay dense
(non-MoE) even in an MoE model:

```
layer 0        layer 1        layer 2        layer 3 ... layer 47
[ dense MLP ]  [ dense MLP ]  [ dense MLP ]  [ MoE ]  ... [ MoE ]
   \_______first_k_dense_replace=3_______/    \___the rest are MoE___/
                                     (Qwen3-30B-A3B: 0 dense layers --
                                      DeepSeek-V3: first 3 layers dense)
```

FLOPs/token (MoE-FFN part) = `2 * active_params_for_that_layer` -- same "2
FLOPs per active param" rule as dense, but `active_params` here only counts
`experts_per_tok` experts plus always-on shared experts, not the full expert
pool. This is why total and active FLOPs diverge sharply for MoE even though
the *rule* (2 FLOPs/active-param) is identical to dense.

### 6.2 Memory requirements

Total params counts every expert (the full "capacity" pool); active params
counts only `experts_per_tok` fired experts plus shared experts -- this gap
is `MoEParams` (`opal_model.py:27`).

KV cache formula is unchanged from whichever attention type the model uses
(GQA formula for `qwen3_moe`; MLA formula for `deepseek_v3` -- MoE does not
add or change any per-token cache state on its own).

For weight *bytes moved per batch* (not per single token): because different
tokens in a batch route to different experts, the number of *distinct*
experts whose weights must be read from memory grows with batch size (not
just `experts_per_tok` per token). `_expected_distinct_experts()`
(`roofline_inference_model.py:78`) models this as a coupon-collector union
bound: `num_experts * (1 - (1 - experts_per_tok/num_experts)^num_tokens)`.
`bytes_weights()` combines that expected-distinct-experts count with
`expert_params_per_layer` for the MoE bytes term, plus `shared_active_params`
(always resident) for the rest. No additional recurrent/SSM state.

### 6.3 Worked example: Qwen3-30B-A3B

Config: [`Qwen/Qwen3-30B-A3B/config.json`](https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json)
(relevant fields only):

```json
{
  "model_type": "qwen3_moe",
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "num_attention_heads": 32,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  "torch_dtype": "bfloat16"
}
```

No shared experts, no MLA fields -- plain GQA attention plus MoE.

```
detect_architecture()  ->  "full_attention"   (model_type "qwen3_moe" in
                                                allowlist; MoE fields present,
                                                but MoE and attention type are
                                                independent axes)

total_params   =  30,330,781,696   (~30.3B, all 128 experts x 48 MoE layers)
active_params  =   3,151,691,776   (~3.15B, only 8 of 128 experts/token)

              total_params / active_params  ~=  9.6x

key_value_bytes/token = 2 * 4 kv_heads * 128 head_dim * 2 bytes (bf16) *
                         48 layers = 98,304 bytes (~96 KiB) -- identical
                         formula/shape to the dense case; MoE didn't change it
```

Matches the HF model card's "30.5B total / 3.3B active" within the test's
2%/5% tolerance -- the ~10x gap between total and active is MoE's entire
point: pay the memory cost of 128 experts, the compute cost of only 8.

## 7. MLA (Multi-head Latent Attention)

Covers DeepSeek-V3, Kimi-K2.x, GLM-5.2. Hallmark fields:
`kv_lora_rank`/`q_lora_rank`/`mla_latent_dim`. `model_type` allowlist:
`deepseek_v3`.

### 7.1 FLOPs per token

Timing: `MLAAttentionRooflineModel` (`roofline_inference_model.py:283`)
reuses `FullAttentionRooflineModel`'s FLOP shape unchanged -- MLA still
attends over the full prior context, so the same quadratic-prefill /
linear-decode attention FLOPs formula from [section 5.1](#51-flops-per-token)
applies. Only the KV *byte* footprint differs (below); FLOPs/token math is
identical to dense/GQA, including the `2 * active_params` dense-FFN term
(and the MoE variant of it, if the model is also MoE, per
[section 6.1](#61-flops-per-token)).

LatentMoE variant (e.g. Nemotron-3-Ultra, hallmark field `moe_latent_size`):
experts operate on a projected-down latent dim with a 2-matrix (up+down)
shape instead of standard 3-matrix SwiGLU -- confirmed against real
safetensors tensor names.

### 7.2 Memory requirements

The key difference from GQA is the KV cache: MLA compresses K and V into one
shared low-rank latent vector per token, plus a small decoupled RoPE key
shared across heads -- there is no `2*` factor, because K/V are not separate
tensors here the way GQA's are. `kv_cache_dim_per_layer()`
(`config_loader.py:545`):

```
kv_cache_dim = kv_lora_rank + qk_rope_head_dim
```

```
GQA / MHA:  cache [ K_head0 ][ K_head1 ]...[ K_headN ][ V_head0 ]...[ V_headN ]
                   \_____________ 2 * num_kv_heads * head_dim ______________/

MLA:        cache [   compressed latent (kv_lora_rank)   ][ RoPE key (small) ]
                   \_______________ no "2 *" --  K and V are reconstructed ___/
                                      from ONE shared latent, not stored twice
```

Using the GQA formula for an MLA model overcounts KV-cache size by ~40x
(confirmed against GLM-5.2-NVFP4). `_build_mla`/`_build_sparse_mla` in
`ModelFactory` raise `UnsupportedModelError` if `kv_cache_dim_per_layer`
returns `None` (i.e. neither `kv_lora_rank` nor `mla_latent_dim` is present)
rather than falling back to a guessed `512`. No additional recurrent/SSM
state -- MLA's compression happens entirely inside the cached latent itself.

### 7.3 Worked example: DeepSeek-V3

Config: [`deepseek-ai/DeepSeek-V3/config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json)
(relevant fields only):

```json
{
  "model_type": "deepseek_v3",
  "hidden_size": 7168,
  "num_hidden_layers": 61,
  "num_attention_heads": 128,
  "kv_lora_rank": 512,
  "q_lora_rank": 1536,
  "qk_rope_head_dim": 64,
  "n_routed_experts": 256,
  "num_experts_per_tok": 8,
  "n_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "first_k_dense_replace": 3
}
```

MoE + MLA together -- `detect_architecture()` returns `"mla"` for
`model_type == "deepseek_v3"` (there's no sparse-attention hallmark field
alongside MLA's, so it isn't routed to `"sparse_mla"`).

```
total_params   =  673,492,850,688   (~673B; card says 671B, within 2% tol)
active_params  =   40,018,728,960   (~40.0B; card says 37B, within 10% tol --
                                      wider tolerance because active-param
                                      estimation stacks MLA + MoE + shared
                                      experts + 3 dense layers all at once)
moe.num_experts       = 256
moe.experts_per_tok   =   8
moe.moe_layers        =  58   (61 total - 3 dense via first_k_dense_replace)

key_value_bytes/token =  35,136   (~34.3 KiB) via kv_lora_rank + qk_rope_head_dim,
                          vs. a naive GQA-style guess using 128 attention
                          heads that would be nearly 40x larger
```

## 8. Sparse attention (DSA / IndexShare)

Covers DeepSeek Sparse Attention and GLM IndexShare (DeepSeek-V3.2, GLM-5.2).
Hallmark fields: `index_topk`/`dsa_topk`/`nsa_topk`/`index_n_heads`. Builds
on MLA (same KV cache formula, [section 7.2](#72-memory-requirements)) but
each query attends to only `topk` selected prior tokens instead of the full
context.

Not currently covered by `tests/test_hf_model_configs.py` (no
sparse-attention model is validated there yet -- DeepSeek-V3.2/GLM-5.2 with
DSA/IndexShare would be the natural addition) -- no worked example below for
this reason, but the formulas and strict-validation behavior are implemented
and covered indirectly by `ModelFactory._build_sparse_mla`'s own required
fields.

### 8.1 FLOPs per token

`SparseAttentionRooflineModel` (`roofline_inference_model.py:310`) replaces
the quadratic `_sum_prefill_pairwise` term with `topk * new_tokens`, so
attention FLOPs/bytes become **linear** in context length rather than
quadratic (a real efficiency win over MLA's otherwise-identical formula).
It also adds an indexer cost (`flops_extra`) amortized every
`indexer_share_period` layers -- the indexer decides which `topk` tokens to
attend to and is often shared/reused across nearby layers rather than
recomputed every layer:

```
full attention (MLA):     query at position n attends to ALL n prior tokens
                           [t0][t1][t2][t3][t4][t5]...[t_{n-1}] -> quadratic in n

sparse attention (DSA):    query at position n attends to only `topk` of them,
                           chosen by a (periodically shared) indexer
                           [t0][  ][t2][  ][  ][t5]...[t_{n-1}] -> linear in n
                            ^selected  ^selected          ^selected
```

Indexer FLOPs/token: `2 * hidden * (index_n_heads * index_head_dim)` when
`indexer_flops_per_token` isn't given directly in the config --
`ModelFactory._build_sparse_mla` raises `UnsupportedModelError` if neither is
present, rather than guessing `2 * hidden * 256`.

### 8.2 Memory requirements

Identical KV-cache formula to MLA (`kv_lora_rank + qk_rope_head_dim`,
[section 7.2](#72-memory-requirements)) -- sparsity changes which prior
tokens are *attended to* per query, not what's stored in the cache; the full
context's compressed latents are still resident. `topk` and
`indexer_share_period` are required config fields (raise
`UnsupportedModelError` if absent) rather than defaulted to `2048`/`4`. No
additional recurrent/SSM state.

## 9. Hybrid Mamba (+ MoE, + attention)

Covers Nemotron-H, Nemotron-3-Ultra, Jamba, and IBM's Granite-4.0 "H"
variants. Hallmark fields: `hybrid_override_pattern`/`layers_block_type`/
`layer_types` (an explicit per-layer type string or list, e.g. `"M-E*"` for
mamba/mlp/moe/attention) or a Mamba-specific field like `mamba_d_state`.
`model_type` allowlist: `nemotron_h`, `granitemoehybrid`, `jamba`.

**This is the one place the per-layer pattern, not a single global flag,
decides each layer's shape.** `estimate_params()` walks the pattern token by
token:

```
"m"/"mamba"             -> is_layer_mamba
"*"/"attention"/"attn"  -> is_layer_attention
anything else ("-", "e", "moe", ...) -> FFN-only layer (dense MLP or MoE-FFN)
```

### 9.1 FLOPs per token

Layer-type params/FLOPs:

- **Mamba-2 layer**: input/output projections sized by `d_inner = expand *
  hidden_size`, a depthwise conv over `d_inner` channels, and the SSM
  parameters themselves (`d_inner*d_state + d_inner*dt_rank +
  dt_rank*hidden_size`). Additional per-token scan cost on top of the
  `2 * active_params` dense-FFN rule:
  `flops_extra = n_mamba_layers * scan_const * d_model * d_state * new_tokens`
  (`HybridMambaMoERooflineModel`, `roofline_inference_model.py:363`;
  `scan_const` defaults to `3.0`, see [section 4](#4-strict-validation-unsupportedmodelerror)).
  Only *new* tokens pay the scan cost; a cached prefix's recurrent state
  carries forward unchanged, mirroring how attention prefill only pays for
  pairs not already covered by a cached prefix.
- **Attention layer**: same GQA formula and FLOPs as [section 5.1](#51-flops-per-token).
- **FFN-only layer**: standard dense MLP FLOPs, *unless* the pattern token
  (or, for Granite-4-style configs, the model's own convention -- see 9.3)
  marks it as MoE, in which case the MoE formula from
  [section 6.1](#61-flops-per-token) applies instead.

### 9.2 Memory requirements

**KV cache**: only the pattern-marked attention layers carry a growing
per-token KV cache; Mamba layers instead carry a small, *fixed-size*
recurrent state that does not grow with context length. `count_attention_layers()`
(`config_loader.py:510`) counts only the attention-tagged layers for exactly
this reason -- using `num_hidden_layers` unconditionally would overcount
KV-cache memory (13x for Nemotron-H-8B, 9x for Nemotron-3-Ultra).

**SSM state** (the "additional state needed for prefix caching" on top of
KV cache): each Mamba-2 layer's recurrent state is `d_state * d_inner`
elements, resident for the lifetime of the sequence regardless of context
length (unlike KV cache, which grows with tokens seen so far). Bytes moved
per decode step:

```
bytes_extra = n_mamba_layers * d_state * d_model * bytes_per_elem * decode_requests
```

`d_state` here is the same `state_size`/`ssm_state_size`/`mamba_d_state`
config field used in the per-layer Mamba param formula above -- it is both
a param-count input (bigger state = bigger `ssm_params`) and a per-step
compute/bytes driver (bigger state = more scan FLOPs and more recurrent-state
bytes moved per decode step). This field is required (raises
`UnsupportedModelError` if none of the three aliases is present, per
[section 4](#4-strict-validation-unsupportedmodelerror)) -- there is no
silent `128` default.

### 9.3 Why `is_moe` is gated on the pattern/convention, not on `num_experts` alone

Some `transformers` `AutoConfig` classes (e.g. `NemotronHConfig` in
`transformers==5.8.0`) carry a nonzero `n_routed_experts` *class-level
default* that `AutoConfig.to_dict()` returns even for a `config.json` that
never sets it. Trusting `num_experts > 0` alone once inflated a dense 10.1B
model's `total_params` to 24.17B by treating every FFN layer as an 8-expert
MoE layer (see `model-porting-steps.md`, "`estimate_params()` MoE
misclassification bug").

`estimate_params()` fixes this by calling `detect_architecture(cfg)` once as
the single source of truth, then branching on the *combination* of detected
architecture and `model_type`:

- **Nemotron-H-style convention** (`model_type in {"nemotron_h", "jamba"}`,
  or `granitemoehybrid` as a fallback): the pattern is exclusive 3-way --
  each token is mamba XOR attention XOR FFN-only, and `is_moe` for an
  FFN-only layer is `True` only if that token is itself explicitly an
  `"e"`/`"moe"` token. A pattern with no MoE tokens at all means `is_moe` is
  `False` for the whole model, regardless of what `num_experts` says.
- **Granite-4-Hybrid convention** (`model_type == "granitemoehybrid"`, the
  common case): the pattern is only 2-way (mamba XOR attention for
  sequence-mixing) -- there is no separate FFN-only token, because a
  MoE-FFN block runs *additively* on every layer regardless of whether that
  layer's sequence-mixer is Mamba or attention. `moe_on_every_layer` is
  `True` here whenever the model has MoE fields at all. If a
  `granitemoehybrid` config's pattern *does* carry an explicit FFN/MoE
  token anyway (contradicting this assumption), `estimate_params()` raises
  `UnsupportedModelError` rather than silently picking one convention over
  the other (`tests/test_unsupported_models.py::TestGraniteHybridPatternContradiction`).

This is why Granite-4.0-H-Small (72 experts, MoE on all 40 layers) and
Nemotron-H-8B (0 experts, MoE-free) are both correctly handled by the same
`hybrid_mamba_moe` code path despite using the pattern field differently.

### 9.4 Worked example: Nemotron-H-8B-Base-8K

Config: [`nvidia/Nemotron-H-8B-Base-8K/config.json`](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K/blob/main/config.json)
(relevant fields only):

```json
{
  "model_type": "nemotron_h",
  "hidden_size": 4096,
  "num_hidden_layers": 52,
  "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
  "ssm_state_size": 128,
  "conv_kernel": 4,
  "expand": 2
}
```

`hybrid_override_pattern` is 52 characters: 24 `M` (Mamba), 4 `*` (attention),
24 `-` (FFN-only, dense MLP -- no MoE fields set in this particular config,
so `is_moe` is `False` for the whole model despite `NemotronHConfig`'s
class-level `n_routed_experts` default):

```
pattern:  M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-
          |<-----4 M, 1 - ---->|*  (repeated ~5x, then a tail)

layer type tally over all 52 layers:
  Mamba-2 layers:     24   ->  SSM params (in_proj/out_proj/conv/ssm_params)
  attention layers:    4   ->  GQA params (same formula as dense case)
  FFN-only layers:    24   ->  dense MLP params (no MoE tokens in this pattern)
```

```
detect_architecture()  ->  "hybrid_mamba_moe"   (model_type "nemotron_h" in
                                                  allowlist)

total_params  =  active_params  =  10,101,096,448   (~10.1B -- this is the
                                                       model this repo's own
                                                       MoE-misclassification
                                                       bug hit; see 9.3)

attention_layers = 4   (only the 4 "*" positions -- NOT 52; using
                         num_hidden_layers here would overcount KV-cache
                         memory by 13x for this model)

key_value_bytes/token = 16,384 bytes, from the 4 attention layers'
                         GQA-style KV cache alone -- the 24 Mamba layers
                         instead carry a small fixed-size recurrent state
                         (ssm_state_size=128) that does not grow with
                         context length, and is not counted here at all.
```

### 9.5 Worked example: Granite-4.0-H-Small

Config: [`ibm-granite/granite-4.0-h-small/config.json`](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json)
(relevant fields only, fetched live via `ModelConfigLoader.load(...)`):

```json
{
  "model_type": "granitemoehybrid",
  "hidden_size": 4096,
  "num_hidden_layers": 40,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_local_experts": 72,
  "num_experts_per_tok": 10,
  "shared_intermediate_size": 1536,
  "intermediate_size": 768,
  "mamba_d_state": 128,
  "mamba_d_conv": 4,
  "mamba_expand": 2,
  "tie_word_embeddings": true,
  "vocab_size": 100352,
  "torch_dtype": "bfloat16"
}
```

Unlike Nemotron-H, this config's `layer_types` list is 2-way, not 3-way --
`Counter(layer_types)` over the live config gives `{"mamba": 36,
"attention": 4}`, with **no** `"-"`/FFN-only token anywhere. Per the
Granite-4-Hybrid convention ([section 9.3](#93-why-is_moe-is-gated-on-the-patternconvention-not-on-num_experts-alone)),
every one of the 40 layers additionally carries a MoE-FFN block
(72 experts, 10 fired per token, plus a shared expert of
`shared_intermediate_size=1536`) regardless of whether its sequence-mixer is
Mamba or attention:

```
layer_types tally over all 40 layers:
  Mamba-2 layers:      36   ->  SSM params, PLUS a MoE-FFN block on top
  attention layers:     4   ->  GQA params, PLUS a MoE-FFN block on top
  (no FFN-only layer type exists in this convention -- MoE-FFN is additive,
   not a separate layer type, unlike Nemotron-H's exclusive 3-way pattern)
```

```
detect_architecture()  ->  "hybrid_mamba_moe"   (model_type
                                                  "granitemoehybrid" in
                                                  allowlist)

total_params   =  32,188,252,160   (~32.19B; IBM's model card says "32B total")
active_params  =   8,784,035,840   (~8.78B; IBM's model card says "9B active")

moe.num_experts       =  72
moe.experts_per_tok   =  10
moe.moe_layers        =  40   (every layer -- no first_k_dense_replace here;
                                MoE-FFN is additive per section 9.3, not gated
                                by pattern position)

attention_layers = 4 of 40   (only the 4 "attention"-tagged layer_types
                               entries; the 36 "mamba" entries instead carry
                               fixed-size recurrent state, same principle as
                               Nemotron-H)

key_value_bytes/token = 16,384 bytes  (2 * 8 kv_heads * 128 head_dim *
                         2 bytes bf16 * 4 attention layers -- same GQA
                         formula as Nemotron-H's attention layers, just a
                         different attention-layer count/shape)

kv_cache_dim_per_layer = 2,048   (= 2 * 8 * 128, matches num_key_value_heads
                                   and head_dim above)
```

This is the model whose porting exposed both bugs fixed in this pipeline:
`layer_types` wasn't originally in the pattern-field alias list (silently
falling back to "every layer is full attention" and producing ~4.2B instead
of the real ~32.19B), and the 2-way-pattern-plus-additive-MoE convention
wasn't originally distinguished from Nemotron-H's exclusive 3-way pattern
(silently producing a total with no MoE contribution at all). Both are now
covered by `tests/test_unsupported_models.py::TestGraniteHybridPatternContradiction`
and the allowlist/convention logic in [section 9.3](#93-why-is_moe-is-gated-on-the-patternconvention-not-on-num_experts-alone).

## 10. Dtype / quantization bytes-per-element

`guess_bytes_per_elem()` (`config_loader.py:481`) sniffs the config text for
`fp4`/`fp8`/`bf16`/`fp16` and returns bytes/element accordingly (NVFP4 = 0.5
B, FP8 = 1.0 B, bf16/fp16 = 2.0 B). `guess_kv_cache_bytes_per_elem()`
(`config_loader.py:492`) is separate because KV cache is sometimes quantized
to a *different* precision than the weights (e.g. Nemotron-3-Ultra: NVFP4
weights, FP8 KV cache) -- it prefers `quantization_config.kv_cache_scheme`
when present, falling back to the weight dtype guess otherwise.

## 11. Multimodal (VLM) wrapper configs

Some real configs (e.g. Kimi-K2.x-NVFP4) wrap the LLM backbone under a
`text_config` key. `effective_llm_config()` (`config_loader.py:49`) unwraps
this once, transparently, wherever `hidden_size` isn't found at the top
level -- every function above calls it first, so the rest of the math never
needs to know the config was wrapped.

## 12. Summary: five worked models side by side

```
                        Llama-3.1-8B   Qwen3-30B-A3B   DeepSeek-V3      Nemotron-H-8B    Granite-4.0-H-Small
                        ============   =============   ===========      =============    ===================
model_type              llama          qwen3_moe       deepseek_v3      nemotron_h       granitemoehybrid
architecture            full_attention full_attention  mla              hybrid_mamba_moe hybrid_mamba_moe
total_params            8.03B          30.33B           673.49B          10.10B           32.19B
active_params           8.03B (=total) 3.15B            40.02B           10.10B (=total)  8.78B
total/active ratio      1.0x           9.6x             16.8x            1.0x             3.7x
attention layers        32 of 32       48 of 48         61 of 61         4 of 52          4 of 40
KV cache bytes/token    131,072        98,304           35,136           16,384           16,384
MoE experts (total/tok) --             128 / 8          256 / 8          -- (pattern has  72 / 10 (every
                                                                          no MoE tokens)    layer, additive
                                                                                            per section 9.3)
```
