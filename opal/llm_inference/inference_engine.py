# SPDX-License-Identifier: Apache-2.0
"""
Selects and builds the inference-timing engine for a simulation, based on
worker.inference_params.model.

Both paths return an object exposing .estimate(batch) -> dict with at least
a "time_s" key, so opal.worker.vllm_worker.py's _calculate_batch_time can
call either uniformly: self.gpu_model.estimate(batch)["time_s"].

Built once per simulation, in opal.core.environment.OpalSimulatorEnvironment
(stored as self.inference_engine), not once per worker: every worker is
constructed against the same shared opal_config, so worker.hw/
worker.inference_params are simulation-wide already -- rebuilding an
identical engine per worker was pure redundant work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from opal.llm_inference.inference_model_factory import ModelFactory
from opal.llm_inference.legacy_gpu_model import GPUModel
from opal.llm_inference.opal_model import OpalModel
from opal.llm_inference.roofline_inference_model import GPUConfig, LLMRooflineModel

if TYPE_CHECKING:
    # type hints only -- see legacy_gpu_model.py for why these stay guarded
    from opal.core.environment import OpalSimulatorEnvironment
    from opal.config.opal_config import ConfigProxy


def build_inference_engine(
    opal_env: "OpalSimulatorEnvironment", opal_config: "ConfigProxy"
) -> Union[GPUModel, LLMRooflineModel]:
    model_name = str(opal_config["worker"]["inference_params"]["model"]).casefold()
    if model_name == "llm_inference":
        return _build_llm_inference_engine(opal_env, opal_config)
    # "roofline" / "synthetic" / anything else -- GPUModel itself validates
    # the value and stops the simulation on an unrecognized one, unchanged
    return GPUModel(opal_env, opal_config)


def _gpu_config_from_hw(hw_cfg, tp: int) -> GPUConfig:
    """Builds the dtype-specific peak-FLOPS fields (see GPUConfig) from a
    worker.hw config block. A bare `tflops` (no suffix) is treated as the
    fp16 figure, matching every config committed so far. Explicit
    tflops_fp32/fp16/fp8/fp4 keys override individually; whichever isn't
    given is derived from fp16: fp8 = fp16*2, fp4 = fp16*2 (not compounding a
    further 2x -- real fp4/fp8 ratios vary too much by GPU generation to
    guess reliably), fp32 = fp16/2.

    Each field is pre-scaled by tp here, matching how peak_bandwidth is
    scaled below -- callers read GPUConfig's fields directly, no further tp
    scaling needed."""
    fp16 = float(hw_cfg.get("tflops_fp16", hw_cfg.get("tflops")))
    fp32 = float(hw_cfg.get("tflops_fp32", fp16 / 2))
    fp8 = float(hw_cfg.get("tflops_fp8", fp16 * 2))
    fp4 = float(hw_cfg.get("tflops_fp4", fp16 * 2))
    return GPUConfig(
        num_gpus=int(hw_cfg.get("num_gpus", tp)),
        tp=tp,
        peak_flops_fp32=fp32 * 1e12 * tp,
        peak_flops_fp16=fp16 * 1e12 * tp,
        peak_flops_fp8=fp8 * 1e12 * tp,
        peak_flops_fp4=fp4 * 1e12 * tp,
    )


def _build_llm_inference_engine(opal_env, opal_config) -> LLMRooflineModel:
    llm_model: OpalModel = opal_env.llm_model  # already loaded once for this simulation
    hw_cfg = opal_config["worker"]["hw"]
    inference_params = opal_config["worker"]["inference_params"]

    tp = int(hw_cfg.get("tp", 1))

    # same worker.inference_params.a/.b keys the legacy engine reads --
    # only overridden when a config explicitly sets them
    a_raw = inference_params.get("a")
    b_raw = inference_params.get("b")

    # Default compute/memory efficiency depends on whether a/b are present.
    # The legacy GPUModel has no separate efficiency factor at all -- its
    # tflops/mem_bw_TBps, combined with a/b, are already calibrated to
    # reproduce real achieved throughput. If a/b carry over from a config
    # already tuned for the legacy engine, applying a <1.0 default
    # efficiency on top would double-discount: confirmed empirically on a
    # real RAG workload, new-vs-old prefill time diverged by ~2.6-3x with
    # naive 0.4/0.6 defaults, dropping to ~6-25% once efficiency was set to
    # 1.0 instead. So: default to 1.0 whenever a/b are present, 0.4/0.6
    # (typical achievable fraction of peak) otherwise -- either is
    # overridden if the config sets its own compute_efficiency/
    # memory_efficiency.
    legacy_calibrated = a_raw is not None or b_raw is not None
    default_efficiency = 1.0 if legacy_calibrated else 0.4
    default_mem_efficiency = 1.0 if legacy_calibrated else 0.6

    hw = _gpu_config_from_hw(hw_cfg, tp)
    # matches legacy_gpu_model.GPUModel's convention: tp scales both
    # effective compute (tflops * tp) and effective aggregate bandwidth
    # (dividing by tp * mem_bw == scaling bandwidth by tp)
    hw.peak_bandwidth = float(hw_cfg["mem_bw_TBps"]) * 1e12 * tp
    hw.compute_efficiency = float(inference_params.get("compute_efficiency", default_efficiency))
    hw.memory_efficiency = float(inference_params.get("memory_efficiency", default_mem_efficiency))

    factory = ModelFactory()  # config_dir unused here: create_from_config() never touches the loader
    model = factory.create_from_config(
        llm_model.config_dict,
        llm_model.model_name,
        hw,
        a_override=float(a_raw) if a_raw is not None else None,
        b_override=float(b_raw) if b_raw is not None else None,
    )
    return model
