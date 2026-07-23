# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import itertools


class LLMRequestStats:
    def __init__(self):
        """Here is the description of the steps and timestamps
        1. creation_time: the request itself is created at the workload generator
        2. arrival_time: request arrives at (a) router and (b) vLLM worker
        3. start_processing_time: request starts processing at a worker, i.e., the scheduler looked at it
        4. request_ready_time: The request is made ready (i) either KVCache was fetched; (ii) no KVCache match
        5. prefill_done_time: prefill done time at the worker
        6. decode_done_time: decode done time at the worker

        __scheduler_timestamps: raw scheduler timestamps
        __num_prefill_sched_steps: number of scheduler prefill steps (the last step is about end of prefill, hence TTFT)
        __num_kvc_hit_tokens: number of tokens that hit the KVCache
        """
        self._1_creation_time = 0
        self._2a_router_arrival_time = 0
        self._2b_worker_time = 0
        self._3_start_processing_time = 0
        self._4_request_ready_time = 0
        self._5_gpu_start_time = 0
        self._6_prefill_done_time = 0
        self._7_decode_done_time = 0
        self._8_done_at_router = 0

        self.__scheduler_timestamps = []
        self.__kvc_hit_tokens = 0
        # min what is needed is one
        self.__num_prefill_sched_steps = 0

    def set_router_arrival_time(self, time):
        self._2a_router_arrival_time = time

    def set_worker_arrival_time(self, time):
        self._2b_worker_time = time

    def set_completion_time(self, time):
        self._8_done_at_router = time

    def get_router_arrival_time(self):
        return self._2a_router_arrival_time

    def get_completion_time(self):
        return self._8_done_at_router

    def get_queue_time(self):
        """Queue time is the time between a request creation and it gets to the GPU. In between we are in
        some sort of queue always"""
        return self._5_gpu_start_time - self._1_creation_time

    def set_prefix_hit_tokens(self, tokens: int):
        self.__kvc_hit_tokens = tokens

    def get_prefix_hit_tokens(self):
        return self.__kvc_hit_tokens

    def get_scheduler_steps(self):
        return len(self.__scheduler_timestamps)

    def add_scheduler_timestamp(self, timestamp):
        self.__scheduler_timestamps.append(timestamp)

    def mark_prefill_done(self):
        self.__num_prefill_sched_steps = len(self.__scheduler_timestamps)
        assert self.__num_prefill_sched_steps > 0, "Prefill done time should be set after at least one scheduler step"
        self._5_prefill_done_time = self.__scheduler_timestamps[self.__num_prefill_sched_steps - 1]

    def get_ttft(self):
        return self._5_prefill_done_time - self._1_creation_time

    def get_kvc_fetch_time(self):
        return self._4_request_ready_time - self._3_start_processing_time

    def get_decode_times_including_ttft(self) -> list[float]:
        # these are timestamps, and we need to convert them into deltas

        # Start with TTFT as the first element
        result = [self.get_ttft()]

        # Calculate inter-token latencies (ITLs) for decode tokens
        # Start from the prefill completion timestamp and calculate deltas
        decode_timestamps = self.__scheduler_timestamps[self.__num_prefill_sched_steps :]

        for i in range(len(decode_timestamps)):
            if i == 0:
                # First decode token: delta from prefill completion
                prev_timestamp = self.__scheduler_timestamps[self.__num_prefill_sched_steps - 1]
            else:
                # Subsequent decode tokens: delta from previous decode token
                prev_timestamp = decode_timestamps[i - 1]

            itl = decode_timestamps[i] - prev_timestamp
            result.append(itl)

        return result

    def get_total_worker_time(self):
        return self._7_decode_done_time - self._2b_worker_time

    def get_gpu_time(self):
        # start end difference is the GPU time (it includes the scheduling overheads)
        # TODO: may be we can do better
        return self.__scheduler_timestamps[-1] - self.__scheduler_timestamps[0]


class LLMRequest:
    _id_counter = itertools.count()
    __slots__ = ("id", "env", "stage_id", "worker_id", "input_length", "output_length", "hash_ids", "output_token_ids", "stats", "session_id", "span_id", "trace_id", "has_completed")

    """Represents a request with an ID and timing information."""

    def __init__(self, env, stage_id: int, input_length: int, hash_ids: list = None, output_length=1):
        self.id = next(LLMRequest._id_counter)
        self.env = env
        self.stage_id = stage_id
        self.worker_id: int | None = None
        self.input_length = input_length
        self.output_length = output_length
        self.hash_ids = hash_ids
        self.output_token_ids: list[int] | None = None
        self.session_id: str | None = None
        self.span_id: str | None = None
        self.trace_id: str | None = None
        self.stats = LLMRequestStats()
        self.stats._1_creation_time = self.env.now
        self.has_completed = self.env.event()

    def mark_completed(self):
        # Why this is required: a producer that replays turns in order (e.g. an OTel
        # session) submits turn i, then does `yield request.has_completed` and blocks,
        # because it must not send turn i+1 until turn i is actually done. Nothing else
        # tells it the turn finished. `has_completed` is the simpy event it waits on;
        # firing it here is the signal that unblocks the producer. Without this call the
        # event is never triggered and that `yield` hangs forever, stalling the session.
        # The guard keeps it idempotent (simpy raises if an event is succeeded twice), and
        # the event is owned by this request, so its firing rule lives here, not in callers.
        if not self.has_completed.triggered:
            self.has_completed.succeed(self)

    def __str__(self):
        return (
            f"LLMReq (id={self.id}, stage_id={self.stage_id}, input={self.input_length}, output={self.output_length})"
        )


class IORequest:
    _id_counter = itertools.count()

    class _Stats:
        def __init__(self):
            self.arrival_time = 0
            self.submission_time = 0
            self.completion_time = 0
            self.start_kvc_time = 0
            self.io_time = 0

    """Represents a request with an ID and timing information."""

    def __init__(self, env, io_type: int = 0, io_size_mb: float = 1, source=None):
        self.id = next(IORequest._id_counter)
        self.env = env
        self.io_type = io_type
        self.source = source
        self.io_size_mb = io_size_mb
        self.remaining_size = io_size_mb
        self.iostats = self._Stats()
        self.iostats.arrival_time = self.env.now
        self.has_completed = self.env.event()  # simpy event / condition variable

    def __str__(self):
        return f"IORequest {self.id} (size={self.io_size_mb})"
