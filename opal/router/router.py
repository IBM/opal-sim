# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from time import sleep
from typing import Any
import simpy
import logging
import numpy as np
from opal.core.events import OpalInfraEvent, KVCEvent, SystemEvent
from opal.kvcache.kvbm import KVBM
from opal.core.request import LLMRequest
from opal.stats.metrics import MetricsSnapshot
from opal.worker.vllm_worker import LLMWorkerVLLMScheduler
from opal.utils.util import parse_bool, safe_process


class Router:
    """Handles incoming requests and distributes them to workers."""

    def _get_policy_func(self):
        policy: str = self.opalConfig["router"]["router_params"]["policy"]
        self.log.debug(f"Initializing with {policy} routing policy")
        policy_lower = policy.lower()

        if policy_lower == "roundrobin":
            # define this extra variable to keep track of assignments
            self._last_selected = -1
            return self._policy_roundrobin
        elif policy_lower == "leastloaded":
            return self._policy_leastloaded
        elif policy_lower == "random":
            return self._policy_random
        elif policy_lower == "maxprefix":
            return self._policy_maxprefix
        elif policy_lower == "balanced":
            return self._policy_balanced
        else:
            # If an exact match is not confirmed, this last case will be used if provided
            raise Exception(
                f"{policy} no such routing policy. Supported policies are: RoundRobin, LeastLoaded, Random, MaxPrefix, Balanced"
            )

    def __init__(self, opal_env, opal_config):
        self.opal_env = opal_env
        self.opalConfig = opal_config
        self.name = "Router"
        self.rng: np.random.Generator = self.opal_env.get_fresh_random_variable()
        # Setup logger
        self.log = logging.getLogger(str(self))
        self.sim_env: simpy.Environment = self.opal_env.simpy_env
        self.input_queue = simpy.Store(self.sim_env)  # , capacity=127)
        self.results_queue = simpy.Store(self.sim_env)
        # leave infinite capacity for this
        self._event_queue = simpy.Store(self.sim_env)
        self.periodic_infra_update_collection_time = self.opalConfig["router"]["router_params"][
            "periodic_infra_update_collection_time"
        ]
        self._kvbm = KVBM(self.opal_env)
        self._worker_stats = None
        self._policy_func = self._get_policy_func()

        # Last SystemEvent reported by each worker and the latest
        # MetricsSnapshot built from them.
        self._latest_worker_metrics: dict[int, SystemEvent] = {}
        self._latest_metrics: MetricsSnapshot | None = None

        self.num_workers = self.opalConfig["simulation"]["num_workers"]
        self._worker_cls = LLMWorkerVLLMScheduler

        # important, put this empty arrays before calling the add workers
        self._active_workers: dict[int, Any] = {}
        # why do we need this separate set? Because we need to quickly index into
        # the values() or keys() of the dictionary above. But these are not _scriptable_
        # we cannot do dict.values()[2]. That we can in the set.
        self._active_worker_ids: list[int] = []
        # Live queue depth per worker (worker_id -> in-flight request count).
        # Incremented on route_request, decremented on completion; read by
        # LeastLoaded routing and by the autoscaler in _per_second_scaling.
        self._outstanding_requests_per_worker: dict[int, int] = {}
        self._stats_request_allocated_per_worker: dict[int, int] = {}
        self.safe_add_workers(add_new_workers=self.num_workers)

        # Link workers to results queue
        # FIXME: very ugly
        for w in self._active_workers.values():
            w.simpy_env.results_queue = self.results_queue

        self.log.info(f"Created {self.num_workers} workers")
        # Start simulation processes and accounting
        # This is accounting for tracking per-second requests finished
        self.cumulative_finished = 0
        self.show_progress = parse_bool(self.opalConfig["simulation"]["show_progress"])
        if parse_bool(self.opalConfig["router"]["router_params"]["enable_scaling"]):
            # get scaling parameters
            self.max_workers = int(self.opalConfig["router"]["router_params"]["max_workers"])
            self.scale_latency = int(self.opalConfig["router"]["router_params"]["scale_latency"])
            # this threshold is used to trigger starting a new worker when any worker's queue hits
            # this threshold.
            self.max_queue_threshold = int(self.opalConfig["router"]["router_params"]["max_queue_threshold"])
        self._run()

    def __del__(self):
        for w in self._active_workers.values():
            del w  # This will trigger the worker's __del__ method

    def _run(self):
        """Start gateway processes."""
        self.sim_env.process(self._accept_requests())
        self.sim_env.process(self._collect_completion())
        self.sim_env.process(self._per_second_stats())
        self.sim_env.process(self.process_events())
        if self.show_progress:
            import threading

            t = threading.Thread(target=self.print_user_time_updates)
            t.start()
        if parse_bool(self.opalConfig["router"]["router_params"]["enable_scaling"]):
            # enable scaling logic
            self.sim_env.process(self._per_second_scaling())

    def __str__(self):
        return self.name

    def print_user_time_updates(self):
        last = 0
        while not self.opal_env.are_we_done():
            sleep(1.0)
            print(f"Processed {self.cumulative_finished - last} request/sec. Done so far {self.cumulative_finished}")
            last = self.cumulative_finished
        print(f"Finished the user printing thread, total processed are: {self.cumulative_finished}")

    def _per_second_scaling(self):
        while not self.opal_env.are_we_done():
            yield self.sim_env.timeout(1)
            max_elements = max(self._outstanding_requests_per_worker.values())
            if max_elements > self.max_queue_threshold and len(self._active_workers) < self.max_workers:
                yield self.sim_env.timeout(self.scale_latency)
                self.safe_add_workers(add_new_workers=1)
        self.log.debug(f"Breaking the per second scaling loop at {self.sim_env.now}")

    def _per_second_stats(self):
        from opal.stats.stage_statistics import StageStatistics

        last = 0
        while not self.opal_env.are_we_done():
            yield self.sim_env.timeout(1)
            stats: StageStatistics = self.opal_env.workload_orchestrator.get_active_stage_stats()
            stats.add_per_unit_workdone(self.cumulative_finished - last)
            last = self.cumulative_finished
            stats.add_per_unit_workers(len(self._active_workers))
            gpu_time_avg = np.mean([w.gpu_busy_time for w in self._active_workers.values()])
            # when all GPU are 100% utilized all the time, then this number to the close to the time
            # so far.
            utilization = gpu_time_avg * 100 / self.sim_env.now
            stats.add_per_unit_gpu_utilization(utilization)
            self._pool_metrics_snapshot(stats)
        self.log.debug(f"Breaking the per second stats loop at {self.sim_env.now}")

    def _pool_metrics_snapshot(self, stats):
        """Create a new MetricsSnapshot object containing all of the most recent
        telemetry from workers and newly computed latency percentiles.

        A new snapshot is created every second, but workers only push new telemetry
        every worker.periodic_infra_update_time seconds.
        """
        p95_ttft, p95_itl = stats.recent_p95_ttft_itl()
        snapshot = MetricsSnapshot(
            timestamp=self.sim_env.now,
            queue_depth_per_worker={wid: ev.queue_depth for wid, ev in self._latest_worker_metrics.items()},
            kvc_util_per_worker={wid: ev.kvc_utilization for wid, ev in self._latest_worker_metrics.items()},
            gpu_util_per_worker={wid: ev.gpu_utilization for wid, ev in self._latest_worker_metrics.items()},
            p95_ttft_secs=p95_ttft,
            p95_itl_secs=p95_itl,
        )
        self._latest_metrics = snapshot
        stats.add_metrics_snapshot(snapshot.to_dict())

    def _policy_leastloaded(self, req: LLMRequest):
        queue_size = min(self._outstanding_requests_per_worker.values())
        worker = next((k for k, v in self._outstanding_requests_per_worker.items() if v == queue_size), None)
        return worker

    def _policy_random(self, req: LLMRequest | None = None):
        worker = self._active_workers[self.rng.choice(self._active_worker_ids)]
        return worker

    def _policy_roundrobin(self, req: LLMRequest | None = None):
        self._last_selected += 1
        if self._last_selected == len(self._active_worker_ids):
            self._last_selected = 0
        worker = self._active_workers[self._active_worker_ids[self._last_selected]]
        return worker

    def _policy_balanced(self, req: LLMRequest | None = None):
        raise Exception()

    def _policy_maxprefix(self, req: LLMRequest | None = None):
        scores = self._kvbm.scorer(req)
        if len(scores) == 0 or (max(list(scores.values())) == 0):
            # if all is empty, no updates so far, do:
            # also in the case when there is actually no worthy match
            # in that case, always the first worker gets the work.
            # FIXME: random assignment or sticky?
            worker = self._policy_random()
        else:
            max_score = max(scores.values())
            # there can be more than one worker with the same max prefix
            workers = [k for k, v in scores.items() if v == max_score]
            # then pick one randomly from them
            worker = self._active_workers[self.rng.choice(workers)]

        return worker

    def __allocate_register_single_worker(self):
        w = self._worker_cls(self.opal_env, self.opalConfig, self.results_queue, self._event_queue)
        self.opal_env.registry.add_worker(w)
        return w

    def safe_add_workers(self, add_new_workers: int = 1):
        new_workers: list[LLMWorker | LLMWorkerSingleStage] = [
            self.__allocate_register_single_worker() for _ in range(add_new_workers)
        ]
        for w in new_workers:
            self._active_workers[w.id] = w
            self._active_worker_ids.append(w.id)
            # expand the array where we keep track of outstanding requests
            # FIXME: this should be removed
            self._outstanding_requests_per_worker[w.id] = 0
            self._stats_request_allocated_per_worker[w.id] = 0
        logging.info(f"** adding {add_new_workers} workers, so far now {len(self._active_workers)}")

    def _accept_requests(self):
        """Distribute requests to the worker with shortest incoming request queue."""
        while not self.opal_env.are_we_done():
            request: LLMRequest = yield self.input_queue.get()
            request.stats.set_router_arrival_time(self.sim_env.now)
            self.opal_env.workload_orchestrator.get_active_stage_stats().queued_requests += 1
            worker = self._policy_func(request)
            self.log.debug(
                f" req={request} mapped to worker.{worker.id}, queue_status: {str(self._outstanding_requests_per_worker[worker.id])}"
            )
            request.worker_id = worker.id
            self._outstanding_requests_per_worker[worker.id] += 1
            self._stats_request_allocated_per_worker[worker.id] += 1

            yield safe_process(self.sim_env, worker.queue_work(request))
            self.log.debug(f"{request} routed to {worker}")

    def _collect_completion(self):
        """Collect and log completed requests."""
        while not self.opal_env.are_we_done():
            request: LLMRequest = yield self.results_queue.get()
            request.stats.set_completion_time(self.sim_env.now)
            total_time = request.stats.get_completion_time() - request.stats.get_router_arrival_time()
            assert request.worker_id is not None
            assert self._outstanding_requests_per_worker[request.worker_id] > 0
            self._outstanding_requests_per_worker[request.worker_id] -= 1
            self.log.debug(f"{request} completed (E2E time: {total_time:.2f})")
            safe_process(
                self.opal_env.simpy_env, self.opal_env.workload_orchestrator.queue_response_from_router(request)
            )
            self.opal_env.workload_orchestrator.get_active_stage_stats().add_finished_request(request)
            self.cumulative_finished += 1

        self.log.info(f"Finishing the request collection loop at the router {self.cumulative_finished}")

    def queue_events(self, elist: list[OpalInfraEvent], delay: float = 0):
        if delay > 0:
            yield self.sim_env.timeout(delay)

        for elem in elist:
            yield self._event_queue.put(elem)

    def process_events(self):
        max_event_batch_size = self.opalConfig["router"]["router_params"]["max_event_batch_size"]

        while not self.opal_env.are_we_done():
            kvbm_events = []
            systems_events = []

            # Collect events up to batch size
            while len(self._event_queue.items) > 0 and not self.opal_env.are_we_done():
                # Stop if either batch is full
                if len(kvbm_events) >= max_event_batch_size or len(systems_events) >= max_event_batch_size:
                    break

                e = yield self._event_queue.get()
                if isinstance(e, KVCEvent):
                    kvbm_events.append(e)
                elif isinstance(e, SystemEvent):
                    systems_events.append(e)
                    # Store the newest telemetry for this worker so we can record
                    # it in a new MetricsSnapshot.
                    self._latest_worker_metrics[e.worker_id] = e
                else:
                    raise Exception(f"Unknown event type {type(e)}")

            # Process batches if we have any events
            if kvbm_events or systems_events:
                self._kvbm.process_kvc_events(kvbm_events)
                self._kvbm.process_system_events(systems_events)

            # If queue still has items, continue immediately
            if len(self._event_queue.items) > 0:
                continue

            # Otherwise sleep until next periodic check
            yield self.sim_env.timeout(self.periodic_infra_update_collection_time)

    def shutdown(self):
        self._stats_request_allocated_per_worker = dict(sorted(self._stats_request_allocated_per_worker.items()))
        self.log.debug(f"Summary of the worker <-> request assignments")
        for i, v in self._stats_request_allocated_per_worker.items():
            self.log.debug(f"\t worker[{i}] processed : {v} requests")
