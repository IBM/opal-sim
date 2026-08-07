# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import threading
import simpy
import datetime
import sys, os
import logging
import time
import numpy as np
from opal.config.opal_config import OpalConfig
from opal.config.llm_model import get_model
from opal.core.opal_registery import OpalRegistry
from opal.router.router import Router
from opal.utils.util import check_and_create_directory
from opal.utils.opal_logging import reset_logging_formatter, setup_logging
from opal.workloads.workload_orchestrator import WorkloadOrchestrator

# make sure printing is done properly
sys.stdout.reconfigure(line_buffering=True)


class OpalSimulatorEnvironment:
    """Entry class to create and run the llm-platform simulation.

    This class initializes and manages a llm-platform simulation with configurable parameters
    such as random seed, number of workers, workload type, and simulation duration.
    """

    def __init__(self, config: OpalConfig, output_dir: str = None):
        # find and open the config file and pass that around
        self.opalConfig = config
        self.output_dir = output_dir
        self.simulation_time = -1
        self.simpy_env = None
        self.log = None
        self.llm_model = None
        self.router = None
        self.workload_done = False
        self.initialize()

    def get_config(self):
        return self.opalConfig

    def get_fresh_random_variable(self):
        """Return a numpy random generator with the configured seed."""
        seed = self.opalConfig("simulation/seed")
        return np.random.Generator(np.random.PCG64(seed))

    def initialize(self):
        self.simulation_time = self.opalConfig["simulation"]["simulation_time"]
        # Optional real-time (wall-clock) cap in seconds. -1 / absent = disabled.
        self.max_wall_time_sec = self.opalConfig["simulation"].get("max_wall_time_sec", -1)
        # Set up environment
        from simpy import Environment

        self.simpy_env: Environment = simpy.Environment()
        self._lock = threading.Lock()
        self.registry = OpalRegistry()
        # Setup logging
        if self.output_dir is not None:
            timestamp = datetime.datetime.now().strftime("%y-%m-%d_%H_%M_%S")
            self.output_dir = os.path.join(self.output_dir, f"sim-{timestamp}")
            check_and_create_directory(self.output_dir, create_parents=True, fail_if_exists=True)
            log_level = os.getenv("OPAL_LOG_LEVEL", "INFO").upper()
            setup_logging(self.simpy_env, log_level=log_level, log_file=f"{self.output_dir}/simulation.log")
            self.log = logging.getLogger("OpalEnv")
            self.log.info(f"Simulation results will be stored in: {self.output_dir}")
            self.opalConfig.save(f"{self.output_dir}/sim_config.json")
        else:
            log_level = os.getenv("OPAL_LOG_LEVEL", "INFO").upper()
            setup_logging(self.simpy_env, log_level=log_level, log_file=None)
            self.log = logging.getLogger("OpalEnv")
            self.log.warning(f"No output directory set")

        self.llm_model = get_model(**self.opalConfig["model"]["model_params"])
        if not "name" in self.opalConfig["model"]["model_params"]:
            # the "name" is not in the "model_param", then init it
            self.opalConfig["model"]["model_params"]["name"] = self.llm_model.get_model_name()

        # Step-1: Initialize gateway (which initializes workers)
        self.router = Router(self, self.opalConfig)
        # put the router in the global registry
        self.registry.put_router(self.router)
        # Step-2 Initialize the Workload orchestrator
        self.workload_orchestrator = WorkloadOrchestrator(self)
        # setup the directory structure
        for i in range(self.workload_orchestrator.get_num_stages()):
            check_and_create_directory(
                os.path.join(self.output_dir, self.workload_orchestrator.get_stage_directory_name(i))
            )

        self.workload_done = False
        self.finish_time = -0.1
        self.simpy_env.process(self._check_if_workload_done())
        self.simpy_env.process(self.workload_orchestrator.run())

    def __del__(self):
        del self.router
        del self.workload_orchestrator

    # these functions are needed to replace the while(True) loops around
    def mark_done(self):
        with self._lock:
            self.workload_done = True
            self.finish_time = self.simpy_env.now
            self.log.info(f"Simulation is finished now!")
            reset_logging_formatter()

    def are_we_done(self):
        with self._lock:
            return self.workload_done

    def _check_if_workload_done(self):
        while not self.workload_orchestrator.are_we_done():
            # check per-second
            yield self.simpy_env.timeout(1.0)
        # when the loop finishes mark ourself done.
        self.workload_orchestrator.get_active_stage_stats().simulation_end = self.simpy_env.now
        self.log.info(
            f"Simulation is marked down now at {self.workload_orchestrator.get_active_stage_stats().simulation_end} seconds"
        )
        self.mark_done()
        self.shutdown()

    def shutdown(self):
        # the idea of this function is to orderly shutdown the whole system,
        # Thus also leading to printing and processing statistics
        self.router.shutdown()
        self.workload_orchestrator.shutdown()

    def run(self, simulation_time):
        if simulation_time is None:
            simulation_time = self.simulation_time

        # Start wall clock timer
        self.wall_clock_start = time.perf_counter()

        if not (self.max_wall_time_sec and self.max_wall_time_sec > 0):
            # No wall-clock cap: run until the end of the simulation
            if simulation_time == -1:
                self.log.debug(f"Running the simulation until the end")
                # run until all events elapsed
                self.simpy_env.run()
            else:
                # No wall-clock cap: run for a finite number of virtual seconds
                self.log.debug(f"Running the simulation for {simulation_time} virtual seconds")
                self.simpy_env.run(until=simulation_time)
                self.mark_done()
                self.shutdown()
        else:
            # run with wall clock cap
            self.log.debug(f"Running the simulation with a {self.max_wall_time_sec}s wall-clock cap")
            self._run_with_wall_cap(simulation_time)

        # Calculate and log wall clock time
        wall_clock_elapsed = time.perf_counter() - self.wall_clock_start
        self.log.info(
            f"Simulation completed in {wall_clock_elapsed:.2f} seconds (wall clock time) for {self.finish_time:.2f} virtual seconds | speed up {self.finish_time/wall_clock_elapsed:.2f}x"
        )
        return wall_clock_elapsed, self.finish_time

    def _run_with_wall_cap(self, simulation_time):
        """Drive the SimPy event loop manually so we can enforce a real-time cap.

        Replicates ``env.run(until=...)`` while checking the wall clock every
        ``CHECK_EVERY_N_STEPS`` events, adding zero extra events to the queue.
        """
        CHECK_EVERY_N_STEPS = 1000
        until = None if simulation_time == -1 else simulation_time
        n = 0
        truncated = False
        try:
            while until is None or self.simpy_env.peek() < until:
                self.simpy_env.step()
                n += 1
                if n % CHECK_EVERY_N_STEPS == 0: # check every N simulated steps if we have exceeded the wall clock cap
                    if time.perf_counter() - self.wall_clock_start >= self.max_wall_time_sec:
                        truncated = True
                        break
        except simpy.core.EmptySchedule:
            # no more events: the workload finished naturally
            pass

        if truncated:
            elapsed = time.perf_counter() - self.wall_clock_start
            self.log.warning(
                f"WALL-TIME CAP: run truncated after {elapsed:.2f}s real time "
                f"(limit {self.max_wall_time_sec}s) at virtual time {self.simpy_env.now:.2f}s"
            )

        # Graceful stop, unless a workload-completion / until path already marked us done.
        if not self.are_we_done():
            stats = self.workload_orchestrator.get_active_stage_stats()
            if stats is not None:
                stats.simulation_end = self.simpy_env.now
            self.mark_done()
            self.shutdown()

    def write_simulation_data(self):
        # simulation is now finished, ask workload orchestrator to give our results
        if self.output_dir is not None:
            for i, s in enumerate(self.workload_orchestrator.stage_stats):
                stats_file = os.path.join(self.output_dir, f"stage_{i}", "opal_stats.json")
                s.write_to_json(stats_file)
        self.log.info(f"Simulation data is written to {self.output_dir}")

    @classmethod
    def stop_simulation(cls):
        raise simpy.core.StopSimulation()
