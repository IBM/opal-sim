# SPDX-License-Identifier: Apache-2.0
"""
GPUModel: the pre-existing empirical a/b roofline + synthetic latency
engines, moved here from opal/gpu_model.py (still re-exported there for
backward compatibility) and kept as the fallback inference engine -- selected
via worker.inference_params.model == "roofline" (default) or "synthetic".
opal.llm_inference.inference_engine picks between this and the new
llm_inference roofline model family (model == "llm_inference") once per
simulation, in opal.core.environment.OpalSimulatorEnvironment.

Everything below estimate() is unchanged from the original file. estimate()
is new: it packages the same per-request max prefill time + decode-batch
time computation opal.worker.vllm_worker.py's _calculate_batch_time used to
do inline, behind the same estimate(batch) -> dict interface
LLMRooflineModel uses, so vllm_worker.py can call either engine identically.
"""

import logging
from typing import TYPE_CHECKING

from opal.llm_inference.opal_model import OpalModel
from opal.utils.util import generate_time_with_rate_variation

if TYPE_CHECKING:
    # unguarded import would be circular: vllm_worker imports GPUModel (via
    # gpu_model's re-export) before BatchMetadata is even defined
    from opal.worker.vllm_worker import BatchMetadata


class GPUModel:
    """This class deals with model and GPU computation specific details"""

    def __init__(self, opal_env, opal_config):

        self.log = logging.getLogger(str(self))
        self.opalEnv = opal_env
        self.opalConfig = opal_config
        self.model: OpalModel = self.opalEnv.llm_model
        self.gpu_name = self.opalConfig["worker"]["hw"]
        if "tp" in self.opalConfig["worker"]["hw"]:
            self.tp = int(self.opalConfig["worker"]["hw"]["tp"])
        else:
            self.tp = 1
        self.gpu_tflops = self.opalConfig["worker"]["hw"]["tflops"]
        self.mem_bw_Bps = (self.opalConfig["worker"]["hw"]["mem_bw_TBps"]) * (10**12)
        self._local_config = self.opalConfig["worker"]["inference_params"]

        if self._local_config["model"].casefold() == "roofline":
            self._prefill_engine = self._roofline_model
            self._decode_engine_batch = self._roofline_get_decode_latency_batch
            self.const_a: float = float(self._local_config["a"])
            self.const_b: float = float(self._local_config["b"])
            self.log.info(
                f"{self} using the Roofline (analytical) model with a = {self.const_a}, b = {self.const_b}, "
                f"gpu = {self.gpu_name}, tflops = {self.gpu_tflops}, count = {self.tp}"
            )
        elif self._local_config["model"].casefold() == "synthetic":
            self._prefill_engine = self._synthetic_model
            self._decode_engine_batch = self._synthetic_get_decode_latency_batch
            self.mean_latency_secs = float(self._local_config["mean_latency_secs"])
            self.log.info(
                f"{self} using the synthetic latency model with each step taking {self.mean_latency_secs} secs."
            )
            self.log.warning(f" (Not used) gpu = {self.gpu_name}, tflops = {self.gpu_tflops}, count = {self.tp}")
        else:
            self.log.error(
                f"Uknown GPU latency model : {self._local_config['model']}. Valid choices are: Roofline, or Synthetic"
            )
            self.opalEnv.stop_simulation()

        assert self.model is not None
        self.log.info(f"working with model {self.model}")

    def __str__(self) -> str:
        return "GPUModel"

    def _roofline_model(self, prompt_length: int = 1000, prefix_length: int = 0):
        if prompt_length < prefix_length:
            str = f"prompt length {prompt_length} must be greater or equal to than the prefix length {prefix_length}"
            assert False, str
        a: float = self.const_a
        b: float = self.const_b
        l: int = self.model.num_hidden_layers
        tp = self.tp
        d: int = self.model.hidden_size
        n: int = prompt_length
        p: int = prefix_length
        # in the full self-attention matrix you have N*(N+1)/2 elements; with
        # a P-token prefix you skip the P*(P+1)/2 pairs already computed
        an2d = a * ((n * (n + 1)) - (p * (p + 1))) * d
        bnd2 = b * (n - p) * (d * d)
        tflops = self.gpu_tflops
        flops_needed1 = l * an2d
        flops_needed2 = l * bnd2
        effective_tflops = tflops * tp
        runtime1 = flops_needed1 / (effective_tflops * (10**12))
        runtime2 = flops_needed2 / (effective_tflops * (10**12))
        runtime_sec = runtime1 + runtime2
        # TODO: model the megatron style communication latencies overheads as explained in here:
        # https://docs.vllm.ai/en/stable/serving/parallelism_scaling/
        return runtime_sec

    def get_prefill_latency(self, prompt_length, prefix_length: int):
        return self._prefill_engine(prompt_length, prefix_length)

    def _synthetic_model(self, prompt_length, prefix_length: int):
        return generate_time_with_rate_variation(1.0 / self.mean_latency_secs, 0.0)

    def _synthetic_get_decode_latency_batch(self, decode_batch: list[int]):
        return generate_time_with_rate_variation(1.0 / self.mean_latency_secs, 0.0)

    def get_decode_latency_batch(self, decode_batch: list[int]):
        return self._decode_engine_batch(decode_batch)

    def _roofline_get_decode_latency_batch(self, decode_batch: list[int]):
        """This function executes a batch of decode requests.
        The logic here is that the model is loaded once, and then KVCaches are
        streamed for each request. Only 1 forward pass is supported for now.

        Args:
            decode_batch (list[int]): A list of current requests lengths.

        Returns:
            int: in second the time to execute this batch
        """
        total_time_sec = 0
        # quantization-aware weight size (e.g. 0.5 B/param for NVFP4), not a
        # hardcoded "2 for FP16" assumption
        model_size = self.model.get_model_params() * self.model.weight_bytes_per_elem
        # load the model _once_
        total_time_sec = model_size / (self.tp * self.mem_bw_Bps)
        # stream KVCaches for all requests
        total_time_sec += sum(
            [self.model.get_kvc_bytes(prompt_length) / (self.tp * self.mem_bw_Bps) for prompt_length in decode_batch]
        )
        return total_time_sec

    def get_decode_latency(self, current_prompt_length: int, new_decode_tokens: int = 1):
        total_time = 0
        assert new_decode_tokens > 0
        while new_decode_tokens > 0:
            total_time += self._decode_engine_batch([current_prompt_length + 1])
            new_decode_tokens -= 1
            current_prompt_length += 1

        return total_time

    # ------------------------------------------------------------------
    # Opal-integration entry point (new): matches
    # LLMRooflineModel.estimate(batch) -> dict interface, so vllm_worker.py
    # can call either engine uniformly.
    # ------------------------------------------------------------------

    def estimate(self, batch: "BatchMetadata") -> dict:
        """Reproduces what vllm_worker.py's _calculate_batch_time computed
        inline before this method existed: max(per-prefill-request latency,
        decode-batch latency). Unlike the new engine, this charges prefill
        and decode as two independent maxes rather than one combined
        forward pass -- the old model's existing behavior, preserved as-is."""
        max_prefill_time = 0.0
        for request in batch.prefill_requests:
            tokens_to_process = batch.request_tokens.get(request.request_id, 0)
            prefill_time = self.get_prefill_latency(
                request.prompt_processed + tokens_to_process, request.prompt_processed
            )
            max_prefill_time = max(max_prefill_time, prefill_time)

        decode_batch = [r.prompt_tokens + r.decode_tokens_generated for r in batch.decode_requests]
        decode_time = self.get_decode_latency_batch(decode_batch)

        return {"time_s": max(max_prefill_time, decode_time)}
