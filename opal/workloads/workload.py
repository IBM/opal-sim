# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import json
import math
import os
import re
import numpy as np

from opal.core.request import LLMRequest
from opal.utils.util import generate_time_with_rate_variation
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


def _load_pretokenized_sessions(trace_file: str) -> list[tuple[list[dict], "datetime"]]:
    """Load a pre-tokenized trace as the same (turns, base_time) shape load_sessions() returns.

    Each turn is a drop-in for a raw span: it carries span_id/start_time/end_time
    under the same names, plus the token fields Trace._turn_fields() reads.
    """
    from opal.workloads.otel_tokenizer import OtelTokenizer

    sessions = []
    with open(trace_file, "r") as f:
        for line in f:
            if not line.strip():
                continue
            session = json.loads(line)
            turns = session["turns"]
            if not turns:
                continue
            # replay reads the session id off the first span; ours is session-level
            for turn in turns:
                turn["trace_id"] = session["trace_id"]
            sessions.append((turns, OtelTokenizer.parse_iso(turns[0]["start_time"])))
    if not sessions:
        raise ValueError(f"No sessions with turns found in {trace_file}")
    return sessions


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
        yield from self._generate_requests_jsonl()

    def _generate_requests_jsonl(self):
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


class Otel(AbstractWorkload):
    """Replay an OpenTelemetry gen_ai trace (config: type "otel").

    Raw OTel traces are tokenized on the fly and need a `tokenizer` param; traces
    pre-tokenized by tools/agentic/tokenizer.py carry their token ids already, so no
    tokenizer is loaded at all. Distinct from the legacy flat-jsonl `Trace` workload,
    which replays pre-expanded (timestamp, hash_ids) rows.
    """

    def __init__(self, opal_env: "OpalSimulatorEnvironment", stage_id: int, workload_params: dict, req_router):
        super().__init__(opal_env, stage_id, workload_params, req_router, name="OTel")
        self.trace_file = self.workload_params["workload_params"]["trace_file"]

        if "total_requests" in self.workload_params["workload_params"]:
            self.total_requests = int(self.workload_params["workload_params"]["total_requests"])
        else:
            self.total_requests = -1
        # override the name
        self.name = f"Workload(otel {os.path.basename(self.trace_file)})"
        self.log = logging.getLogger(self.name)

        # how to convert timestamps to seconds
        if "multiplier_to_sec" in self.workload_params["workload_params"]:
            self.multiplier_to_sec = float(self.workload_params["workload_params"]["multiplier_to_sec"])
        else:
            self.multiplier_to_sec: float = 1

        # scales only the inter-turn gap (time between a turn finishing and the next turn's
        # wall-clock start), independent of multiplier_to_sec which also affects session
        # arrival spacing. Set to 0 to replay turns back-to-back with no inter-turn delay.
        if "inter_turn_multiplier" in self.workload_params["workload_params"]:
            self.inter_turn_multiplier = float(self.workload_params["workload_params"]["inter_turn_multiplier"])
        else:
            self.inter_turn_multiplier: float = 1

        self.is_finished = False
        self._is_multi_session = False

        # Set for traces already tokenized by tools/agentic/tokenizer.py (they carry
        # token ids, so no tokenizer is loaded); raw OTel traces leave it False.
        self._pretokenized = bool(self.workload_params["workload_params"].get("pretokenized", False))
        if self._pretokenized:
            # tools/agentic/tokenizer.py always writes one session per line
            self._is_multi_session = True
            self._otel_sessions = _load_pretokenized_sessions(self.trace_file)
            self.log.info(f"Trace {self.trace_file} is pre-tokenized; skipping tokenizer load")
        else:
            tokenizer_name = self.workload_params["workload_params"].get("tokenizer")
            if not tokenizer_name:
                raise ValueError(
                    f"OTel trace '{self.trace_file}' requires a 'tokenizer' param in workload_params"
                )
            self.log.warning(
                f"Trace '{self.trace_file}' is not pre-tokenized: every turn will be tokenized "
                f"on the fly, once per run. Tokenize it once instead with "
                f"`python tools/agentic/tokenizer.py {self.trace_file} --tokenizer {tokenizer_name}` "
                f"and point trace_file at the resulting '-tokenized.jsonl' (the 'tokenizer' param "
                f"is then unused). Same simulation, considerably faster startup."
            )
            from opal.workloads.otel_tokenizer import OtelTokenizer
            self._otel_tokenizer = OtelTokenizer(tokenizer_name)
            self._is_multi_session = OtelTokenizer.is_multi_session(self.trace_file)
            if self._is_multi_session:
                self._otel_sessions = OtelTokenizer.load_sessions(self.trace_file)
            else:
                self._otel_spans, self._otel_base_time = OtelTokenizer.load_spans(self.trace_file)

        if self._is_multi_session:
            total_available_sessions = len(self._otel_sessions)
            if self.total_requests > 0:
                self._otel_sessions = self._otel_sessions[: self.total_requests]
            self.log.info(
                f"Loaded {total_available_sessions} session(s) from {self.trace_file}; "
                f"will start {len(self._otel_sessions)} (total_requests={self.total_requests})"
            )
            self._max_concurrent_sessions = int(
                self.workload_params["workload_params"].get("max_concurrent_sessions", 10)
            )

        self.log.info(f"Initialized {self.name} workload generator")

    def __str__(self):
        return self.name

    def generate_requests(self):
        self.request_id = 0
        if self.total_requests == 0:
            self.is_finished = True
            return

        if self._is_multi_session:
            yield from self._generate_requests_otel_multi_session()
        else:
            yield from self._generate_requests_otel_single()

    def _generate_requests_otel_single(self):
        from opal.workloads.otel_tokenizer import OtelTokenizer

        self.log.debug(
            f"[trace_replay] single-session mode: {len(self._otel_spans)} spans, "
            f"base_time={self._otel_base_time.isoformat()}, multiplier={self.multiplier_to_sec}"
        )

        for span_idx, span in enumerate(self._otel_spans):
            if self.opal_env.are_we_done():
                break

            span_id = span.get("span_id", "?")
            trace_id = span.get("trace_id", "?")
            parent_span_id = span.get("parent_span_id")
            wall_start = span.get("start_time", "?")
            wall_end = span.get("end_time", "?")

            fields = self._turn_fields(span)
            token_ids = fields["hash_ids"]
            input_length = fields["input_length"]
            output_token_ids = fields["output_token_ids"]
            output_length = fields["output_length"]
            op_name = fields["op_name"]
            model = fields["model"]
            prompt_tokens_attr = fields["prompt_tokens"]

            delta = (OtelTokenizer.parse_iso(wall_start) - self._otel_base_time).total_seconds()
            timestamp = delta * self.multiplier_to_sec
            sleep_time = timestamp - self.simpy_env.now
            if sleep_time > 0:
                yield self.simpy_env.timeout(sleep_time)

            sim_submit_time = self.simpy_env.now

            request = LLMRequest(
                self.simpy_env, self.stage_id, input_length, hash_ids=token_ids, output_length=output_length
            )
            request.output_token_ids = output_token_ids
            request.span_id = span_id
            request.trace_id = trace_id
            self.request_id += 1
            self.generated_requests += 1

            self.log.debug(
                f"[trace_replay] span {span_idx + 1}/{len(self._otel_spans)} submitted"
                f" | req_id={request.id} trace_id={trace_id} span_id={span_id}"
                f" parent_span_id={parent_span_id}"
                f" | wall_start={wall_start} wall_end={wall_end}"
                f" | op={op_name} model={model}"
                f" | delta_from_base={delta:.6f}s sleep_time={sleep_time:.6f}s"
                f" | sim_submit_time={sim_submit_time:.6f}s"
                f" | in={input_length} (attr={prompt_tokens_attr}) out={output_length}"
                f" has_output_toks={output_token_ids is not None}"
            )

            yield self.req_router.input_queue.put(request)

            if 0 < self.total_requests <= self.request_id:
                self.log.warning(f"Stopping OTel trace replay after {self.request_id} requests")
                break

        self.log.info(f"OTel trace replay finished with {self.request_id} requests")
        self.is_finished = True

    def _generate_requests_otel_multi_session(self):
        """Up to `_max_concurrent_sessions` sessions are queued at once; a queued session only
        starts replaying once a currently-active session finishes and frees its slot (FIFO via
        the simpy.Resource semaphore -- see _replay_session)."""
        import simpy
        from opal.utils.util import safe_process

        capacity = (
            len(self._otel_sessions) if self._max_concurrent_sessions < 0
            else self._max_concurrent_sessions
        )
        self.log.info(
            f"[TRACE REPLAY] [START] Total sessions: {len(self._otel_sessions)},"
            f" Max session concurrency: {capacity}."
        )
        semaphore = simpy.Resource(self.simpy_env, capacity=capacity)
        active_processes = []

        for session_idx, (spans, session_base_time) in enumerate(self._otel_sessions):
            if self.opal_env.are_we_done():
                break
            trace_id = spans[0].get("trace_id", "?") if spans else "?"
            self.log.info(
                f"[TRACE REPLAY] [Session {session_idx + 1}/{len(self._otel_sessions)}] queued"
                f" | trace_id: {trace_id} spans: {len(spans)}"
                f" | original time:{session_base_time.isoformat()}"
            )
            proc = safe_process(
                self.simpy_env,
                self._replay_session(spans, session_base_time, semaphore),
            )
            active_processes.append(proc)

        # wait for all in-flight sessions to finish
        for proc in active_processes:
            yield proc

        self.log.info(f"Multi-session OTel replay finished with {self.request_id} requests")
        self.is_finished = True

    def _turn_fields(self, span: dict) -> dict:
        """Content of one turn: the token fields an LLMRequest needs, plus op_name/model/
        prompt_tokens for the logs.

        Pre-tokenized turns already hold exactly these keys, so they are returned as-is
        (read-only); raw OTel spans are tokenized here on the fly.
        """
        if self._pretokenized:
            return span

        attrs = span["attributes"]
        messages = json.loads(attrs["gen_ai.input.messages"])
        raw_tool_defs = attrs.get("gen_ai.tool.definitions")
        tool_defs = json.loads(raw_tool_defs) if isinstance(raw_tool_defs, str) else raw_tool_defs
        system_instructions = attrs.get("gen_ai.system_instructions")
        token_ids = self._otel_tokenizer.tokenize_messages(
            messages, tool_defs=tool_defs, system_instructions=system_instructions
        )

        raw_output = attrs.get("gen_ai.output.messages")
        output_token_ids = None
        if raw_output:
            output_messages = json.loads(raw_output)
            output_token_ids = self._otel_tokenizer.tokenize_output(
                messages, output_messages, system_instructions=system_instructions
            )

        # Prefer the actual re-tokenized output length; fall back to the
        # trace-reported completion_tokens only when we have no output messages.
        if output_token_ids:
            output_length = len(output_token_ids)
        else:
            output_length = int(attrs.get("gen_ai.usage.completion_tokens", 1))

        return {
            "hash_ids": token_ids,
            "input_length": len(token_ids),
            "output_token_ids": output_token_ids,
            "output_length": output_length,
            "op_name": attrs.get("gen_ai.operation.name", span.get("name", "?")),
            "model": attrs.get("gen_ai.request.model", attrs.get("gen_ai.response.model", "?")),
            "prompt_tokens": int(attrs.get("gen_ai.usage.prompt_tokens", 0)),
        }

    def _replay_session(self, spans, session_base_time, semaphore):
        from opal.workloads.otel_tokenizer import OtelTokenizer

        trace_id = spans[0].get("trace_id", "unknown")
        wall_session_start = spans[0].get("start_time", "?")
        wall_session_end = spans[-1].get("end_time", "?")
        with semaphore.request() as req:
            yield req
            session_sim_start = self.simpy_env.now
            self.log.debug(
                f"[trace_replay][session={trace_id}] acquired semaphore at sim_time={session_sim_start:.6f}s"
                f" | spans={len(spans)} wall_session_start={wall_session_start} wall_session_end={wall_session_end}"
                f" | base_time={session_base_time.isoformat()}"
            )
            prev_wall_end_dt = None
            for span_idx, span in enumerate(spans):
                if self.opal_env.are_we_done():
                    break

                span_id = span.get("span_id", "?")
                parent_span_id = span.get("parent_span_id")
                wall_start = span.get("start_time", "?")
                wall_end = span.get("end_time", "?")

                fields = self._turn_fields(span)
                token_ids = fields["hash_ids"]
                input_length = fields["input_length"]
                output_token_ids = fields["output_token_ids"]
                output_length = fields["output_length"]
                op_name = fields["op_name"]
                model = fields["model"]
                prompt_tokens_attr = fields["prompt_tokens"]

                wall_start_dt = OtelTokenizer.parse_iso(wall_start)
                wall_end_dt = OtelTokenizer.parse_iso(wall_end)

                if span_idx == 0:
                    # Sleep until the first span's wall_start relative to session base
                    delta = (wall_start_dt - session_base_time).total_seconds()
                    sleep_time = delta * self.multiplier_to_sec - (self.simpy_env.now - session_sim_start)
                else:
                    # Sleep for inter-turn gap: time between previous span's end and this span's start
                    inter_turn = (wall_start_dt - prev_wall_end_dt).total_seconds()
                    sleep_time = inter_turn * self.multiplier_to_sec * self.inter_turn_multiplier

                if sleep_time > 0:
                    yield self.simpy_env.timeout(sleep_time)

                sim_submit_time = self.simpy_env.now

                request = LLMRequest(
                    self.simpy_env, self.stage_id, input_length, hash_ids=token_ids, output_length=output_length
                )
                request.output_token_ids = output_token_ids
                request.session_id = trace_id
                request.span_id = span_id
                request.trace_id = trace_id

                self.generated_requests += 1
                if span_idx == 0:
                    inter_turn_log = f"delta_from_session_base={(wall_start_dt - session_base_time).total_seconds():.6f}s"
                else:
                    inter_turn_log = f"inter_turn_gap={(wall_start_dt - prev_wall_end_dt).total_seconds():.6f}s"
                self.log.info(
                    f"[TRACE REPLAY] [Session {trace_id}, Span {span_idx + 1}/{len(spans)}] submitted"
                    f" | req_id={request.id} span_id={span_id} parent_span_id={parent_span_id}"
                    f" | wall_start={wall_start} wall_end={wall_end}"
                    f" | op={op_name} model={model}"
                    f" | {inter_turn_log}"
                    f" sleep_time={sleep_time:.6f}s sim_submit_time={sim_submit_time:.6f}s"
                    f" | in={input_length} (attr={prompt_tokens_attr}) out={output_length}"
                    f" has_output_toks={output_token_ids is not None}"
                )
                self.request_id += 1
                yield self.req_router.input_queue.put(request)

                # Wait for span i to complete before sleeping inter-turn and submitting span i+1
                yield request.has_completed

                prev_wall_end_dt = wall_end_dt
            self.log.info(
                f"[TRACE REPLAY] [Session {trace_id}] finished"
            )
