# LLM Inference & Roofline Migration

`opal.llm_inference` replaces the old `opal.config.llm_model.OpalModelConfig`
+ `opal.infra.gpu_model.GPUModel` pair with a single package that (a) loads
and interprets a model's `config.json` in a quantization/MoE/MLA-aware way,
and (b) estimates per-batch execution time from those dimensions using a
roofline model, instead of the old model's fixed `a`/`b` coefficients alone.

## Why

The old `GPUModel` timing math used two hand-tuned coefficients (`a` for the
attention pairwise-interaction cost, `b` for a fixed-shape dense/FFN term)
calibrated against a small set of dense GQA models. It has no notion of
mixture-of-experts, multi-head latent attention (MLA), sparse attention, or
Mamba/SSM hybrids — architectures increasingly common in current model
releases (DeepSeek-V3, Kimi-K2.x, GLM-5.x, Nemotron-H/Ultra). Extending the
old model per-architecture would have meant more special-case branches inside
`GPUModel` itself. Instead, `opal.llm_inference` models FLOPs and bytes
directly from real config dimensions, with architecture-specific subclasses
only overriding the attention term.

The old `a`/`b` engine isn't removed — it's kept as `legacy_gpu_model.GPUModel`
and remains the default (`worker.inference_params.model: "roofline"`) for
every config that predates this change, so existing simulation results are
unaffected until a config opts in.

## Package layout

```
opal/llm_inference/
  opal_model.py               OpalModel, MoEParams, parse_opal_model_fields
  config_loader.py            ModelConfigLoader, detect_architecture, estimate_params, dtype helpers
  roofline_inference_model.py GPUConfig, LLMRooflineModel + 4 architecture subclasses
  legacy_gpu_model.py         GPUModel (old a/b roofline + synthetic engine), unchanged behavior
  inference_model_factory.py  ModelFactory: config.json -> concrete LLMRooflineModel subclass
  inference_engine.py         build_inference_engine(): picks legacy vs. new, per worker.inference_params.model
```

## OpalModel: config-aware sizing

`OpalModel` (successor to `OpalModelConfig`) loads a model's `config.json`
(from `model-configs/<name>/`, or Hugging Face as a fallback — see
`ModelConfigLoader`) and exposes:

- `get_model_params()` / `active_params` — total vs. active (MoE-aware) parameter count
- `weight_bytes_per_elem` — quantization-aware (NVFP4 → 0.5, FP8 → 1.0, bf16/fp16 → 2.0), not a hardcoded `* 2`
- `key_value_bytes`, `get_kvc_bytes(tokens)` — per-token KV cache footprint, MLA-aware (a compressed latent, not full K/V)
- `num_hidden_layers`, `kv_head_size`, `num_key_value_heads`, `max_position_embeddings`, etc. — same accessors `OpalModelConfig` provided, so `kvc_manager.py` needed no logic changes to switch over.

`config_loader.estimate_params()` derives total/active param counts
layer-by-layer from config dimensions alone (no weight download), handling
dense, MoE (with shared experts), LatentMoE, and Mamba/hybrid per-layer
patterns — validated to within a few percent of published param counts for
DeepSeek-V3, Kimi-K2.x, GLM-5.x, and Nemotron-H/Ultra.

## The roofline model family

`LLMRooflineModel` (abstract base, itself an `OpalModel`) computes
`estimate(batch) -> {"time_s": ..., "flops": ..., "bytes": ...}` as
`max(flops / (compute_efficiency * peak_flops), bytes / (memory_efficiency *
peak_bandwidth))` — a standard compute-vs-memory roofline, unlike the legacy
engine's two independent prefill/decode maxes (see `legacy_gpu_model.GPUModel.estimate()`
for that distinction spelled out).

Dense/FFN FLOPs and MoE weight-read bytes are shared in the base class;
subclasses override only the attention term:

| Class | Attention pattern | KV cache footprint |
|---|---|---|
| `FullAttentionRooflineModel` | dense causal, quadratic in prefill | `2 * num_kv_heads * head_dim` per token per layer (GQA/MHA) |
| `MLAAttentionRooflineModel` | same access pattern as full attention | small latent dim (`kv_lora_rank` + RoPE dim) — ~40x smaller than the GQA formula for GLM-5.2-NVFP4 |
| `SparseAttentionRooflineModel` | top-k (DSA-style): linear in context, not quadratic | only the `topk` selected entries read back per decode step |
| `HybridMambaMoERooflineModel` | few dense attention layers + Mamba-2 selective scan (`flops_extra`/`bytes_extra`) | recurrent state carries forward; only new tokens pay the scan cost |

`ModelFactory` (`inference_model_factory.py`) picks the right subclass via
`config_loader.detect_architecture()`, which checks hallmark config fields
(`kv_lora_rank` → MLA, `index_topk` → sparse attention, `mamba_d_state` /
`layers_block_type` → hybrid Mamba) rather than a model-name allowlist.

## Wiring into the simulation

`build_inference_engine(opal_env, opal_config)` reads
`worker.inference_params.model`: `"llm_inference"` resolves to the new
`ModelFactory`-built engine; anything else (`"roofline"`, `"synthetic"`, the
default) resolves to the legacy `GPUModel`, preserving existing behavior
exactly. `OpalSimulatorEnvironment.initialize()` builds this once per
simulation and stores it as `self.inference_engine`; every worker reads the
same shared instance in its own `__init__` (`self.gpu_model =
self.opalEnv.inference_engine`) instead of each constructing its own
`GPUModel` — rebuilding an identical engine per worker was pure redundant
work once every worker already shares the same `opal_config`.

`vllm_worker.LLMWorkerVLLMScheduler._calculate_batch_time` then calls
`self.gpu_model.estimate(batch)["time_s"]` uniformly — see [[vLLM Worker]]'s
Timing Calculation section.

## Porting status

Ported (semantically) from internal commit `c52fbeb`, merged to `main` as
`5fca50c` (fixes #45, #46):

1. `669c3ce` — add the package, unwired.
2. `f65ab9c` — cut over `environment.py` / `kvc_manager.py` / `vllm_worker.py`.
3. Delete `opal/config/llm_model.py`; replace `opal/infra/gpu_model.py`
   with a re-export shim to `opal.llm_inference.legacy_gpu_model.GPUModel`.

All three steps are complete.

Not carried over: internal repo's `vllm_worker.py` also references an
`opal.apc_policy` module (GPU prefix-caching admission policy) in the same
constructor region touched by this migration. That module doesn't exist in
this repo and wasn't part of `c52fbeb`'s diff — it belongs to separate,
not-yet-ported GPU HBM Automatic Prefix Caching work.
