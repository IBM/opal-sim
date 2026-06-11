# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import json
import math
import os
import re
import numpy as np

from opal.request import LLMRequest
from opal.util import generate_time_with_rate_variation
from opal.workloads.abstract_workload import AbstractWorkload


class UniformReqRate(AbstractWorkload):
    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        stage_id: int,
        workload_params: dict,
        req_router,
        name: str = "Workload(Uniform Rate)",
    ):
        super().__init__(opal_env, stage_id, workload_params, req_router, name)
        self.request_rate = self.workload_params["workload_params"]["request_rate"]
        self.request_interval = 1.0 / self.request_rate
        self.log = logging.getLogger(self.name)
        self.is_finished = False
        self.default_prefix_length = self.workload_params["workload_params"]["default_prefix_length"]
        self.prompt_size_min = self.workload_params["workload_params"]["prompt_size_min"]
        self.prompt_size_max = self.workload_params["workload_params"]["prompt_size_max"]
        self.output_tokens_min = self.workload_params["workload_params"]["output_tokens_min"]
        self.output_tokens_max = self.workload_params["workload_params"]["output_tokens_max"]

        if "max_outstanding_requests" in self.workload_params["workload_params"]:
            self.max_outstanding = self.workload_params["workload_params"]["max_outstanding_requests"]
        else:
            self.max_outstanding = 32

        self.default_hash_ids = [i for i in range(self.default_prefix_length)]
        self.total_requests = self.workload_params["workload_params"]["total_requests"]
        self.request_id = -1
        self.log.info(f"Initialized {self.name} workload generator")

    def __str__(self):
        return f"{self.name}.{self.stage_id}"

    def _intra_request_delay(self):
        return self.request_interval

    def _request_generation_stops(self):
        # what are the conditions when the request generation stops?
        # Global condition 1: global simulation time is over
        # Local condition:
        #                  local timeout happens or
        #                  total requests have been generated, whatever happens first
        global_timeout_happened = (
            self.simpy_env.now > self.opal_env.simulation_time if self.opal_env.simulation_time > 0 else False
        )
        local_timeout_happened = self.local_timeout
        # check if we are doing request-based generation or not
        local_finished_all_requests = self.request_id >= self.total_requests if self.total_requests > 0 else False

        if not (global_timeout_happened or local_timeout_happened or local_finished_all_requests):
            # while none of these conditions have happened, we continue generation
            return False
        else:
            # if any one of them have happened, we stop
            self.log.info(
                f"Workload generation stops here due to one of the following conditions: (1) global stop: {global_timeout_happened}; (2) local timeout: {local_timeout_happened}; (3) all reqs generated: {local_finished_all_requests}"
            )
            return True

    def _generate_prompt(self):
        # this is where request generation and content matching can be implemented
        # pending: https://github.ibm.com/zrl-cloud-data-platforms/opal-sim/issues/24 (atr)
        # numpy's randint is [low, high) so we add 1 to max to make it inclusive like Python's random.randint
        prompt_size = self.rng.integers(self.prompt_size_min, self.prompt_size_max + 1)
        default_hash_ids = [i for i in range(prompt_size)]
        request = LLMRequest(self.simpy_env, self.stage_id, prompt_size, hash_ids=default_hash_ids)
        request.output_length = self._generate_output_size()
        return request

    def _generate_output_size(self):
        # numpy's randint is [low, high) so we add 1 to max to make it inclusive like Python's random.randint
        output_tokens = self.rng.integers(self.output_tokens_min, self.output_tokens_max + 1)
        return output_tokens

    def generate_requests(self):
        """Generate requests with exponential inter-arrival times."""
        self.request_id = 0
        while not self._request_generation_stops():
            request = self._generate_prompt()
            self.request_id += 1
            # UGLY: this is need for accounting, this is ugly right now
            self.generated_requests += 1
            self.log.debug(f"{request} generated")
            yield self.req_router.input_queue.put(request)
            sleep_time = self._intra_request_delay()
            yield self.simpy_env.timeout(sleep_time)

        self.is_finished = True
        self.log.info(f"Workload generation (stage_id:{self.stage_id} finished with {self.request_id} requests ")


class ExponentialReqRate(UniformReqRate):
    def __init__(self, opal_env, stage_id: int, workload_params: dict, req_router):
        super().__init__(opal_env, stage_id, workload_params, req_router, name="Workload(ExponentialReqRate)")
        self._jitter = float(workload_params["workload_params"]["jitter"])

    def _intra_request_delay(self):
        return generate_time_with_rate_variation(self.request_rate, self._jitter)


class Trace(AbstractWorkload):
    def __init__(self, opal_env: "OpalSimulatorEnvironment", stage_id: int, workload_params: dict, req_router):
        super().__init__(opal_env, stage_id, workload_params, req_router, name="Trace")
        self.trace_file = self.workload_params["workload_params"]["trace_file"]

        if "total_requests" in self.workload_params["workload_params"]:
            self.total_requests = int(self.workload_params["workload_params"]["total_requests"])
        else:
            self.total_requests = -1
        # override the name
        self.name = f"Workload(trace {os.path.basename(self.trace_file)})"
        self.log = logging.getLogger(self.name)
        # Extract chunk_size from workload_params, default to 512 if not specified
        if "chunk_size" in self.workload_params["workload_params"]:
            self.chunk_size = int(self.workload_params["workload_params"]["chunk_size"])
        else:
            self.chunk_size = 512

        # Hash-id layout for _expand_prompt: split int32 into [base | offset].
        # offset_bits must be wide enough to hold (chunk_size - 1); base fills the rest.
        # Layout: [ base : 31 - offset_bits ][ offset : offset_bits ]   ← total 31 bits, signed-int32 safe.
        # Shifting base into the high bits guarantees disjoint id ranges per chunk hash
        # (no collisions between chunk_A's tail and chunk_B's head).
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {self.chunk_size}")
        self._hash_id_offset_bits = max(1, math.ceil(math.log2(max(2, self.chunk_size))))
        self._hash_id_base_bits = 31 - self._hash_id_offset_bits
        if self._hash_id_base_bits < 1:
            raise ValueError(
                f"chunk_size={self.chunk_size} too large for int32 hash_ids "
                f"(needs {self._hash_id_offset_bits} bits for offset, leaving {self._hash_id_base_bits} for base)"
            )
        self._hash_id_base_mask = (1 << self._hash_id_base_bits) - 1

        # how to convert timestamps to seconds
        if "multiplier_to_sec" in self.workload_params["workload_params"]:
            self.multiplier_to_sec = float(self.workload_params["workload_params"]["multiplier_to_sec"])
        else:
            self.multiplier_to_sec: float = 1

        # we keep track of what is the raw prompts we have generated for each individual chunked' hashes
        # in the trace. The idea here is that with the prefix matching a hashed chunked token of
        # "A" should have the same matching prompts of individual chunk_size tokens for the matching to work.
        self._expanded_generated_prompts = {}
        self.is_finished = False
        self.log.info(f"Initialized {self.name} workload generator")

    def __str__(self):
        return self.name

    def _expand_prompt(self, chunked_hash: list[int], target_input_size: int):
        """
        Expand chunked hashes into unique non-overlapping token sequences.

        Performance optimized: Uses deterministic expansion instead of random generation.
        Each chunk hash expands to a unique, reproducible sequence of token IDs.

        Args:
            chunked_hash: List of chunk hash integers
            target_input_size: Total number of tokens to generate

        Returns:
            List of token IDs (first target_input_size tokens)
        """
        raw_tokenized_prompt = []
        for h in chunked_hash:
            # Calculate how many tokens to expand for this chunk
            expanded_size = self.chunk_size if target_input_size >= self.chunk_size else target_input_size

            # Cache key includes size for partial chunks at the end
            key = f"{h}:{expanded_size}"

            if key not in self._expanded_generated_prompts:
                # Deterministic expansion: each chunk hash maps to unique token sequence.
                # Layout the id as [ base : self._hash_id_base_bits ][ offset : self._hash_id_offset_bits ]
                # so (base + i) for i in [0, chunk_size) cannot overflow int32, and two distinct
                # `key` values always produce disjoint id ranges.
                base = (hash(key) & self._hash_id_base_mask) << self._hash_id_offset_bits
                hash_ids = [(base + i) for i in range(expanded_size)]
                self._expanded_generated_prompts[key] = hash_ids

            raw_tokenized_prompt.extend(self._expanded_generated_prompts[key])
            target_input_size -= expanded_size

            if target_input_size <= 0:
                break

        return raw_tokenized_prompt

    def generate_requests(self):
        self.request_id = 0
        try:
            f = open(self.trace_file, "r")
        except Exception as e:
            self.log.error(f"Error opening trace file {self.trace_file}")
            self.log.error(e)
            raise
        if self.total_requests == 0:
            # special case:
            self.is_finished = True
            return

        # total_requests = -1 means run through all the requests
        for line in f:
            if not self.opal_env.are_we_done():
                entry = json.loads(line.strip())
                input_length = entry.get("input_length")
                output_length = entry.get("output_length")
                hash_ids = self._expand_prompt(entry.get("hash_ids", []), input_length)
                assert len(hash_ids) == input_length, f"{len(hash_ids)} != {input_length}"
                timestamp = float(entry.get("timestamp")) * self.multiplier_to_sec
                request = LLMRequest(
                    self.simpy_env, self.stage_id, input_length, hash_ids=hash_ids, output_length=output_length
                )
                self.request_id += 1
                # UGLY
                self.generated_requests += 1
                self.log.debug(f"{request} generated")
                sleep_time = timestamp - self.simpy_env.now
                if sleep_time > 0:
                    yield self.simpy_env.timeout(sleep_time)

                yield self.req_router.input_queue.put(request)
                if self.request_id == self.total_requests:
                    self.log.warning(
                        f"Stopping trace replay after {self.request_id} requests, total_requests = {self.total_requests}"
                    )
                    break

        self.log.info(
            f"Trace replay finished replay after {self.request_id} requests, total_requests = {self.total_requests}"
        )
        self.is_finished = True
