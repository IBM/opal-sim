# SPDX-License-Identifier: Apache-2.0
"""
Backward-compatible shim. GPUModel now lives in
opal.llm_inference.legacy_gpu_model, where it's kept as the fallback
inference-timing engine (worker.inference_params.model == "roofline" or
"synthetic") alongside the new opal.llm_inference roofline model family --
see opal.llm_inference.inference_engine for how a worker picks between them.
Re-exported here so `from opal.infra.gpu_model import GPUModel` keeps
working unchanged.
"""

from opal.llm_inference.legacy_gpu_model import GPUModel

__all__ = ["GPUModel"]
