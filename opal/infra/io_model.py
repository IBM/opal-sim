# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import itertools
import logging
import simpy


class OpalIORequest:
    __id = itertools.count(0)

    def __init__(self, size: int):
        self.id = next(OpalIORequest.__id)
        self.size = size
        self.arrival_time = -1
        self.finish_time = -1


class AbstractDevice:
    """This is an abstract device that is designed to
    have certain capacity, have bandwidth of certain unit per unit,
    a base latency of latency_unit with concurrency for parallel
    requests. After which there will be queuing happening. You
    can pass -1, in that case there is no limit.

    Uses a centralized bandwidth manager (similar to DFSBackend) that:
    - Sleeps indefinitely when no requests are active
    - Gets interrupted when new requests arrive
    - Dynamically calculates time to next completion
    - Handles interrupts by recalculating progress based on elapsed time
    """

    # Minimum sleep time to avoid floating point errors (10 microseconds)
    MIN_SLEEP_TIME = 100e-7

    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        name: str = "abstract",
        capacity_bytes: int = 1,
        bandwidth_bytes_per_sec: float = 10.0,
        latency_per_request_sec: float = 0.0,
        concurrency: int = 5,
    ):
        self.opal_env = opal_env
        self.opal_config = self.opal_env.get_config()
        self.simpy_env = self.opal_env.simpy_env
        self.capacity_bytes = capacity_bytes
        self.capacity_remaining_bytes = self.capacity_bytes
        self.bytes_per_sec = bandwidth_bytes_per_sec
        self.latency_per_request_sec = latency_per_request_sec
        self.name = name
        self.max_concurrency = concurrency
        self.concurrency = simpy.Resource(self.simpy_env, capacity=concurrency)

        # Track active requests with their remaining sizes
        self.active_requests = {}  # {request_id: {'request': OpalIORequest, 'remaining': bytes}}
        # Requests queued since the most recent IO completion
        # Invariant: len(self.active_requests) + len(interrupt_queue) < self.concurrency
        self.interrupt_queue = []

        self.log = logging.getLogger(self.name)
        self.log.debug(
            f" {self.name} initialized with capacity: {self.capacity_bytes/2**20} Mbytes, max_bw: {self.bytes_per_sec} bytes/sec, latency: {self.latency_per_request_sec}, concurrency: {concurrency}"
        )

        # Start the centralized bandwidth manager
        self.bw_manager_process = self.simpy_env.process(self._bandwidth_manager())

    def _bandwidth_manager(self):
        """Centralized bandwidth manager that handles all active requests.

        Behavior (per requirements in bob-prompts/io-dynamic-ticks.md):
        - Sleeps indefinitely when no requests are active
        - Gets interrupted when new requests arrive or complete
        - Calculates time to next completion based on current bandwidth sharing
        - Updates progress for all active requests based on elapsed time
        - Minimum sleep time is 10 microseconds
        """
        last_time = self.simpy_env.now

        while not self.opal_env.are_we_done():
            assert len(self.active_requests) + len(self.interrupt_queue) <= self.max_concurrency

            # Examine the interrupt queue and add any requests that were submitted on interrupt.
            for request in self.interrupt_queue:
                self.active_requests[request.id] = {"request": request, "remaining": request.size}
                # self.log.debug(f"{self.simpy_env.now} IORequest[{request.id}] begins transfer via bandwidth manager")
            self.interrupt_queue.clear()

            # If no active requests, sleep indefinitely until interrupted
            if not self.active_requests:
                try:
                    yield self.simpy_env.timeout(1e9)  # Sleep indefinitely
                except simpy.Interrupt:
                    last_time = self.simpy_env.now
                    continue

            # Calculate bandwidth share (equally divided among active requests)
            num_active = len(self.active_requests)
            if num_active == 0:
                continue
            bw_per_req = self.bytes_per_sec / num_active

            # Find the time until the next request completes
            # TODO: Consider using a minheap (e.g. heapq) data structure to avoid O(N^2) request insertion complexity
            time_to_next_completion = float("inf")
            for req_data in self.active_requests.values():
                time_needed = req_data["remaining"] / bw_per_req
                time_to_next_completion = min(time_to_next_completion, time_needed)

            # Enforce minimum sleep time
            time_to_next_completion = max(self.MIN_SLEEP_TIME, time_to_next_completion)

            # Sleep until next completion OR until interrupted
            try:
                yield self.simpy_env.timeout(time_to_next_completion)
                elapsed = self.simpy_env.now - last_time
            except simpy.Interrupt:
                # New request arrived or request completed - recalculate based on elapsed time
                elapsed = self.simpy_env.now - last_time

            # Update progress for all active requests based on elapsed time
            bytes_processed_per_req = bw_per_req * elapsed
            completed_requests = []

            for req_id, req_data in self.active_requests.items():
                req_data["remaining"] -= bytes_processed_per_req

                # Check if request completed
                if req_data["remaining"] <= 0:
                    request = req_data["request"]
                    request.finish_time = self.simpy_env.now
                    request.event.succeed(request)
                    completed_requests.append(req_id)
                    # self.log.debug(f"{self.simpy_env.now} IORequest[{req_id}] completed")

            # Remove completed requests from active set
            for req_id in completed_requests:
                del self.active_requests[req_id]

            last_time = self.simpy_env.now

    def _process_request_io(self, request: OpalIORequest):
        """Process one IO request with latency, then hand off to bandwidth manager.

        This process:
        1. Waits for a concurrency slot
        2. Applies minimum latency
        3. Checks if request can complete within latency period (optimization)
        4. If not, registers with the bandwidth manager
        5. Waits for completion event from bandwidth manager
        """
        request.arrival_time = self.simpy_env.now
        # self.log.debug(f"{self.simpy_env.now} IORequest.{request.id} arrives (size={request.size})")

        # Wait for a concurrency slot
        with self.concurrency.request() as req_slot:
            # self.log.debug(f"{self.simpy_env.now} IORequest[{request.id}] queued")
            yield req_slot  # WAIT IN QUEUE

            # Enforce minimum latency before the transfer begins
            if self.latency_per_request_sec > 0:
                # self.log.debug(f"{self.simpy_env.now} IORequest[{request.id}] starts latency {self.latency_per_request_sec}")
                yield self.simpy_env.timeout(self.latency_per_request_sec)

            # Interrupt the bandwidth manager to recalculate
            try:
                self.interrupt_queue.append(request)
                self.bw_manager_process.interrupt()
            except RuntimeError as e:
                self.log.warning(
                    f"Failed to interrupt bandwidth manager for request {request.id}: {e}. This should not happen in normal operation."
                )

            # Wait for bandwidth manager to complete this request
            yield request.event

            # self.log.debug(f"[\"{self.name}\"] {self.simpy_env.now} IORequest[{request.id}] completed in {format_decimal_nonzero(request.finish_time - request.arrival_time)} time")

    def process_one_request(self, request: OpalIORequest):
        """This is an async interface to process one request

        Args:
            request (OpalIORequest): _description_
        """
        request.event = self.simpy_env.event()
        self.simpy_env.process(self._process_request_io(request))

    def process_requests(self, requests: list[OpalIORequest]):
        """This is an async interface to process a list of requests

        Args:
            requests (list[OpalIORequest]): _description_
        """
        for r in requests:
            r.event = self.simpy_env.event()
            self.simpy_env.process(self._process_request_io(r))


class AbstractKeyDevice(AbstractDevice):
    def __init__(self, opal_env: "OpalSimulatorEnvironment", key: str):
        config = opal_env.get_config()
        self.capacity_GB = config["kvc"][key]["capacity_GB"]
        self.bw_GBps = config["kvc"][key]["bandwidth_GBps"]
        self.latency_nsec = config["kvc"][key]["latency_nsec"]
        self.concurrency = config["kvc"][key]["concurrency"]
        super().__init__(
            opal_env,
            key,
            self.capacity_GB * 2**30,
            self.bw_GBps * 10**9,
            self.latency_nsec / 10**9,
            self.concurrency,
        )


class CPUMemory(AbstractKeyDevice):
    def __init__(self, opal_env: "OpalSimulatorEnvironment"):

        super().__init__(opal_env, "CPUMemory")


class LocalNVMe(AbstractKeyDevice):
    def __init__(self, opal_env: "OpalSimulatorEnvironment"):

        super().__init__(opal_env, "LocalNVMe")


class DistributedFS(AbstractKeyDevice):
    """Singleton implementation of DistributedFS device.

    This ensures only one DistributedFS instance exists throughout the application.
    Multiple calls to DistributedFS(opal_env) will return the same instance.
    """

    # Class variable to hold the single instance
    _instance = None

    def __new__(cls, opal_env: "OpalSimulatorEnvironment"):
        """Control instance creation to enforce singleton pattern.

        __new__ is called before __init__ and is responsible for creating
        the actual object instance. By overriding it, we can control whether
        a new instance is created or an existing one is returned.

        Args:
            opal_env: The OPAL simulator environment

        Returns:
            The single DistributedFS instance (creates it on first call, returns
            existing instance on subsequent calls)
        """
        if cls._instance is None:
            # First time: create the instance using parent's __new__
            cls._instance = super(DistributedFS, cls).__new__(cls)
        # Always return the same instance
        return cls._instance

    def __init__(self, opal_env: "OpalSimulatorEnvironment"):
        """Initialize the Scale instance only once.

        Since __init__ is called every time Scale() is instantiated (even when
        returning an existing instance), we use the _initialized flag to ensure
        the parent class initialization only happens once.

        Args:
            opal_env: The OPAL simulator environment
        """
        # Only initialize once - check if we've already initialized this instance
        if not hasattr(self, "_initialized"):
            super().__init__(opal_env, "DistributedFS")
            self._initialized = True
