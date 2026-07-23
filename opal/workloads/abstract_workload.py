# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Generator
import abc
import logging
import simpy
import numpy as np
from opal.core.request import LLMRequest
from opal.utils.util import get_with_timeout, safe_process


class AbstractWorkload:

    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        stage_id: int,
        workload_params: dict,
        req_router: "Router",
        name: str = "AbstractWorkload",
        process_completion: bool = True,
    ):
        self.log = logging.getLogger(f"AbstractWorkload{stage_id}")
        self.stage_id = stage_id
        self.opal_env = opal_env
        self.workload_params = workload_params
        self.req_router = req_router
        self.rng: np.random.Generator = self.opal_env.get_fresh_random_variable()
        self.name = name
        self.is_finished = False
        self.process_completion = process_completion
        # we initialize it with false
        self.local_timeout = False
        self.generated_requests = 0
        self.received_responses = 0
        self.simpy_env = self.opal_env.simpy_env
        self._router_response_queue = simpy.Store(self.simpy_env)

    @abc.abstractmethod
    def generate_requests(self):
        raise NotImplementedError

    def get_completed_requests(self):
        completed = yield self._router_response_queue.get()
        return completed

    def _queue_response_from_router(self, request: LLMRequest, delay: int = 0):
        self.log.debug(
            f"queuing to the result queue {request}, current queue length {len(self._router_response_queue.items)}"
        )
        # This runs only from the router's completion collection, so reaching here means
        # the turn is actually finished. A producer that submitted this turn may be parked
        # on `yield request.has_completed`, waiting to learn the turn is over before sending
        # the next one; mark_completed() is what tells it, so it doesn't stay stuck. Firing
        # it here (the one point that knows the turn ended) is what makes that signal correct.
        request.mark_completed()
        yield self._router_response_queue.put(request)

    def _process_responses(self):
        # this function gets responses from the router to process
        while (not (self.is_finished and (self.generated_requests == self.received_responses))) and (
            not self.opal_env.are_we_done()
        ):
            ret_value = yield self.simpy_env.process(get_with_timeout(self.simpy_env, self._router_response_queue, 1.0))
            if ret_value == None:
                # self.log.debug(
                #     f" Gone none {self.is_finished} and ({self.generated_requests} == {self.received_responses})"
                # )
                continue
            req: LLMRequest = ret_value
            self.received_responses += 1
            self.log.debug(
                f"processed {req}, self.is_finished = {self.is_finished} self.received_responses={self.received_responses}, self.generated_requests={self.generated_requests}"
            )

        self.log.debug(f"Breaking the response process loop!")

    def _mark_local_timeout_true(self):
        # only schedule if we have > 0 timeout, i.e., we expect to stop in a certain duration
        time_sec = self.workload_params["workload_params"].get("time_duration_sec", -1)
        if time_sec > 0:
            self.log.debug(f"scheduling a timeout {time_sec}")
            yield self.simpy_env.timeout(time_sec)
            self.local_timeout = True
            self.log.info(f"TIMEOUT triggered to finish this stage")
        else:
            self.log.info(f"No timeout for this stage")

    def _run(self) -> Generator[Any, Any, None]:
        # schedule a timeout, if needed
        safe_process(self.simpy_env, self._mark_local_timeout_true())
        if self.process_completion == True:
            # Start processes to generate and collect responses
            process_response = safe_process(self.simpy_env, self._process_responses())
        else:

            def _dummy_process():
                yield self.simpy_env.timeout(0)

            process_response = safe_process(self.simpy_env, _dummy_process())
            self.log.warning(
                "No response processing is configured for this workload. Workload should call get_completed_requests()"
            )

        # start foreground workload generation
        process_generation = safe_process(self.simpy_env, self.generate_requests())
        yield process_generation & process_response
        self.log.debug(f"stage{self.stage_id} done")
