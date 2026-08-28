# Testing

OpalSim's test suite lives under `tests/` and runs with [pytest](https://docs.pytest.org/). This page is the human-readable index of what's actually covered today. Whenever a new test file (or a new test class within an existing file) is added, this page should be updated in the same commit/PR — see the "Keeping this page current" section below.

## Overview

- Tests are discovered automatically: `pyproject.toml` configures `testpaths = ["tests"]` and `python_files = ["test_*.py"]`, so any `tests/test_*.py` file is picked up without extra registration.
- One custom marker is registered: `network` — for tests that download real data (Hugging Face `config.json` files) at test time. These can be skipped for offline/CI runs.
- The full suite runs in roughly 20-30 seconds.

### Running tests

```shell
# All tests, from the project root
uv run pytest

# Verbose output with prints shown
uv run pytest -s -v

# Skip tests that hit the network
uv run pytest -m "not network"

# Run a single file
uv run pytest -s -v tests/test_configs.py

# Run with debug-level simulator logging
OPAL_LOG_LEVEL=DEBUG uv run pytest -s -v tests/test_configs.py
```

If you don't use `uv run`, plain `pytest` works the same way as long as the project's virtualenv is active (see [[Install Requirements]]).

## Test-by-test breakdown

### `tests/test_configs.py` — every shipped config actually runs

Parametrized over **every** `*.json` file in `configs/` (via `CONFIGS_DIR.glob("*.json")`), so each config file in the repo becomes its own test case (`test_config_loads_and_runs[<filename>.json]`). Adding a new config file to `configs/` automatically adds a new test — nothing else to wire up.

- Loads the config through `OpalConfig().initialize(...)`.
- Builds a simulator via `OpalSimulator.from_config(...)`.
- Runs it for 10 virtual seconds (`opal.run(10)`).
- Passing means the config parses and the simulator can execute at least one scheduling step without crashing — it is a smoke test, not a correctness check of simulation output.

### `tests/test_hf_model_configs.py` — model param/KV-cache estimates vs. real HF configs

Marked `pytest.mark.network` (downloads real `config.json` files from Hugging Face via `OpalModel.from_huggingface`, the same `transformers.AutoConfig`-based path `environment.py`'s `get_model()` uses in production). No fixtures are checked into the repo — each model config is covered by its own upstream license. Ground-truth figures are hand-sourced from each model's official card/technical report, with a tolerance band (not exact equality) since both the "documented" figures and the estimate are approximations.

- **`TestDenseModel`** (Llama-3.1-8B-Instruct): total param count within 2% of 8.03B; `total_params == active_params` for a dense (non-MoE) model; per-token KV-cache byte size matches the GQA formula (8 kv heads, head_dim 128, bf16, ×32 layers).
- **`TestMoEModel`** (Qwen3-30B-A3B): total params ~30.5B and active params ~3.3B within tolerance; active < total; MoE breakdown exposes `num_experts == 128` and `experts_per_tok == 8`.
- **`TestMLAModel`** (DeepSeek-V3): total ~671B / active ~37B within tolerance; MoE breakdown (`num_experts == 256`, `experts_per_tok == 8`); asserts the multi-head latent attention (MLA) KV-cache size equals `(kv_lora_rank + qk_rope_head_dim) * attention_layers * bytes_per_elem` (the compressed-latent formula) rather than falling back silently to a naive GQA calculation — and confirms that compressed size is in fact smaller than what a naive GQA formula would produce for the same head count.
- **`TestHybridMambaModel`** (Nemotron-H-8B-Base-8K, hybrid Mamba-2 + attention + MLP): total params within 5% of ~10.1B, checked via two independent paths — `OpalModel.from_huggingface` (production path) and directly through `ModelConfigLoader`/`estimate_params` (raw-JSON path, no `transformers.AutoConfig`). The second path is a regression test for a fixed bug where MLP params were zeroed for every pattern layer instead of only mamba/attention ones, undercounting the model at ~3.76B instead of ~10.1B.

### `tests/test_unsupported_models.py` — strict validation, negative paths

Synthetic config dicts (no network access, no fixtures) that must be rejected with `UnsupportedModelError` rather than silently guessed at. One test class per assumption enforced by the strict-validation pass in `opal/llm_inference/config_loader.py` and `inference_model_factory.py`:

- **`TestUnrecognizedModelType`** — an unknown `model_type` string is rejected by `detect_architecture`, `estimate_params`, and `ModelFactory().create_from_config` alike.
- **`TestHybridMissingMambaFields`** — a hybrid model (`nemotron_h`) whose layer pattern marks layers as Mamba but has none of the Mamba-2 state-space fields (`state_size`/`ssm_state_size`/`mamba_d_state`) raises rather than silently defaulting; separately, a hybrid `model_type` with neither explicit mamba/attention layer counts nor any layer-type pattern also raises (nothing to derive the split from).
- **`TestMoEMissingFields`** — an MoE config missing `num_experts_per_tok`, or missing both `moe_intermediate_size` and `intermediate_size`, is rejected.
- **`TestDenseMissingIntermediateSize`** — a dense config missing `intermediate_size` is rejected.
- **`TestAttentionMissingHeads`** — a config missing `num_attention_heads` is rejected.
- **`TestMLAMissingLatentField`** — exercises `ModelFactory._build_mla` directly for a `deepseek_v3`-typed config missing `kv_lora_rank`/`mla_latent_dim` (and `num_attention_heads`), confirming the defensive check inside MLA construction raises instead of returning something bogus.
- **`TestHybridNoPatternNoModelTypeHeuristic`** — `count_attention_layers` rejects a `granitemoehybrid` config that has no layer-type pattern at all.
- **`TestGraniteHybridPatternContradiction`** — a `granitemoehybrid` pattern is expected to use a 2-way (mamba/attention-only) convention; a pattern that also carries an explicit FFN/MoE token contradicts that assumption and must raise rather than silently picking a convention.
- **`TestSanityPositivePath`** — confirms the well-formed base fixtures the negative tests are derived from (`DENSE_BASE`, `HYBRID_BASE` + Mamba fields) do *not* raise, i.e. the strict checks don't false-positive on valid configs.

### `tests/test_device.py` — I/O device bandwidth/latency model

Uses a minimal fake SimPy environment (`MockEnv`) to drive `AbstractDevice` / `OpalIORequest` directly, independent of the rest of the simulator.

- **`test_latency_dominated`** — a tiny (1 KiB) request's duration is dominated by the device's fixed latency, not bandwidth (duration is within 1% of the configured latency).
- **`test_bandwidth_bound`** — a large (100 GiB) request's duration equals latency *plus* transfer time (`size / BW`) — latency is additive on top of the transfer, not overlapped with it.
- **`test_measured_bw`** — issuing 1000 concurrent requests of a fixed size, the measured aggregate bandwidth never exceeds the device's configured bandwidth.
- **`test_interrupt_bw`** — regression test for the bandwidth manager's interrupt path: a resident large request is mid-transfer when a second request registers partway through; the second request's actual transfer time can never beat the bandwidth-imposed floor (`size / BW`) no matter what else is in flight, i.e. no "fake credit" is handed out on interruption.

### `tests/test_rag_workload.py` — RAG workload request generation

Uses a fake SimPy environment (`MockEnv`) with a seeded RNG (`np.random.default_rng(42)`) to test `RAGWorkload` in isolation, via a `make_workload()` helper with small defaults (10 documents, 4 docs/request, 5 total requests).

- **`test_rejects_docs_per_request_over_num_documents`** — constructing a workload where `docs_per_request > num_documents` raises `ValueError`.
- **`test_request_hashes_include_system_prompt_and_documents`** — the per-request hash-ID sequence built by `_build_request_hashes` starts with the system-prompt hashes and has total length `system_prompt_size + docs_per_request * document_size` (used for prefix-cache simulation).
- **`test_document_hashes_are_cached_and_reused`** — calling `_get_or_create_doc_hashes` twice for the same document index returns the identical (cached) hash list, and only one entry is added to the internal doc-hash table.
- **`test_document_hashes_are_disjoint_across_documents`** — hashes generated for two different document indices never overlap.
- **`test_generate_requests_stops_after_total_requests`** — running the `generate_requests()` SimPy process against a fake queue emits exactly `total_requests` `LLMRequest` objects, marks the workload finished, and sets `request_id` to the expected count.

### `tests/test_stage_statistics.py` — per-stage statistics collection

- **`TestStageStatisticsKVCHitTokens`**:
  - Recording a finished request appends its prefix-cache hit-token count to `raw_kvc_hit_tokens`, in order across multiple requests.
  - `StageStatistics.to_dict()` / `.from_dict()` round-trip preserves `raw_kvc_hit_tokens`.
  - `from_dict()` defaults `raw_kvc_hit_tokens` to an empty list when the field is missing from the serialized data (backward compatibility with older saved stats).
- **`TestStageStatisticsSummaryLogFile`**:
  - `print_summary_results(log_file=...)` writes the vLLM-style "Serving Benchmark Result" block (including a "Successful requests" line) to a provided file-like object.
  - `print_summary_results()` does not raise when called without a `log_file` argument (prints to the default logger/stdout path instead).

### `tests/test_wall_clock_cap.py` — `simulation.max_wall_time_sec`

Both tests load `configs/defaults.json` directly (rather than being parametrized like `test_configs.py`) so they can mutate the `max_wall_time_sec` field before constructing the simulator.

- **`test_wall_clock_cap_truncates_run`** — setting `max_wall_time_sec` to a tiny value (0.001s) causes the run to stop almost immediately: the simulator ends in a clean "done" state (`opal.sim.are_we_done()`), virtual time reached is far short of what the workload would otherwise run for (>100s), and real wall-clock elapsed stays under 5 seconds.
- **`test_no_wall_clock_cap_runs_to_completion`** — with `max_wall_time_sec` left at its default (`-1`, disabled), behavior is unaffected: `opal.run(10)` runs the full 10 virtual seconds requested and ends in a clean done state.

## Markers and fixtures

- **`pytest.mark.network`** — registered in `pyproject.toml`; currently used only by `test_hf_model_configs.py`. Deselect with `-m "not network"` when offline or in CI environments without Hugging Face access.
- **`tmp_path`** — pytest's built-in temporary-directory fixture, used throughout to give each simulator run (or `ModelConfigLoader`) an isolated output/cache directory instead of writing into the repo.
- **`monkeypatch.chdir(...)`** — used in tests that load a config via a relative path (e.g. `./configs/defaults.json`), to ensure the test runs with the project root as the working directory regardless of where pytest was invoked from. See the pattern in `test_configs.py` and `test_wall_clock_cap.py`.
- **Fake/mock SimPy environments** — `test_device.py` and `test_rag_workload.py` both define a minimal `MockEnv` class (a real `simpy.Environment()` wrapped with just the handful of methods the code under test needs, e.g. `get_config()`, `are_we_done()`, `get_fresh_random_variable()`) rather than spinning up a full `OpalSimulatorEnvironment`. This keeps those unit tests fast and focused on the component in isolation.

## Writing a new test

A minimal end-to-end style test — load a config, build a simulator, run it for a short virtual duration:

```python
import pytest
from pathlib import Path
from opal.core.opal import OpalSimulator
from opal.config.opal_config import OpalConfig

def test_my_feature(tmp_path, monkeypatch):
    """Test that loads a config, runs the sim for a short time."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    config = OpalConfig()
    config.initialize("./configs/defaults.json")
    opal = OpalSimulator.from_config(config=config, output_dir=str(tmp_path))
    opal.run(10)  # run for 10 virtual seconds
```

For a narrower unit test that doesn't need a full simulator (e.g. testing a workload, a device model, or a stats class in isolation), follow the `MockEnv` pattern from `test_device.py` / `test_rag_workload.py` instead — construct only the fake environment methods the class under test actually calls, and drive it with a real `simpy.Environment()`.

If the code under test downloads anything (Hugging Face configs, etc.), mark the test `@pytest.mark.network` so it can be excluded from offline runs.

## Keeping this page current

Contributors and coding agents are expected to update this page's test-by-test breakdown whenever they add a new test file, or a new test class within an existing file, in the same commit/PR that adds the test — not as a follow-up. See `AGENTS.md`'s Testing section for the underlying convention. This keeps this page a reliable index of what's actually covered rather than a stale snapshot.

See also [[Contributing]] for the broader pull-request workflow (branching, formatting, commit conventions).
