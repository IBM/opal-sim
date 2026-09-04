# SPDX-License-Identifier: Apache-2.0
import argparse
import os

# Add project root to PYTHONPATH to enable direct execution from the current directory
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from opal import DEFAULT_CONFIG_FILE, PROJECT_ROOT
from opal.stats.stage_statistics import StageStatistics
from opal.stats.plot import simend_plot
from opal.utils.util import check_and_create_directory
from opal.core.environment import OpalSimulatorEnvironment
from opal.config.opal_config import OpalConfig
import gc

import faulthandler
import signal

faulthandler.register(signal.SIGUSR1)  # will print stacks on SIGUSR1
"""
with the above hook, you can dump the stack traces of all threads by sending SIGUSR1 to the process.
# Find the right python PID for the simulator 
$ kill -USR1 `pidof python | tail -1`
"""


DEFAULT_MODELLING_OUTPUT = os.path.join(PROJECT_ROOT, "simulation-runs")


class OpalSimulator:

    def __init__(self, config: OpalConfig, output_dir: str | None = None, plot_graphs: bool = False):
        """
        Build the simulator around a config. See from_config() and from_cmd_args()
        for the alternative entry points.

        FIXME: perhaps move the output_dir and plot_graphs also in the config file

        Args:
            config (OpalConfig): the simulation config.
            output_dir (str, optional): where to put the run's files. Defaults to None,
                i.e. DEFAULT_MODELLING_OUTPUT, as the config does not carry the -o flag.
            plot_graphs (bool, optional): generate graphs or not. Defaults to False.
        """
        # Enable debugging (optional)
        # TODO(atr) - when is this useful?
        # gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE)

        # keep __del__ safe if any of the initialization below raises
        self.sim: OpalSimulatorEnvironment | None = None

        self.config = config
        self.plot_graphs = plot_graphs
        self.default_modelling_output = DEFAULT_MODELLING_OUTPUT
        self.parser = self._build_parser()

        output_dir = output_dir if output_dir is not None else DEFAULT_MODELLING_OUTPUT
        check_and_create_directory(output_dir, create_parents=True, fail_if_exists=True)
        self.sim = OpalSimulatorEnvironment(config, output_dir=output_dir)

    def __del__(self):
        # dynamically enable it to track which config are never used
        if False:
            report = self.config.report_unused_config(log_warnings=True)
            print(f"Total keys: {report['total_keys']}")
            print(f"Accessed: {report['accessed_keys']}")
            print(f"Unused: {report['unused_count']}")
            print(f"Unused keys: {report['unused_keys']}")

        del self.sim
        # gc.set_debug(0)
        # print at the end of the program, it takes a bit of time.
        print("Python Garbage collector stats:")
        print(gc.get_stats())
        # for i in range(3):  # three generations: 0,1,2
        #     print(f"Generation {i}: {gc.get_count()[i]} objects")
        print("=========")

    @staticmethod
    def _build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Welcome to OpalSim, the ultimate GenAI simulator",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            help="directory to put files in",
            default=DEFAULT_MODELLING_OUTPUT,
            required=False,
        )
        parser.add_argument(
            "-c",
            "--config",
            help="Simulation configuration file",
            default=DEFAULT_CONFIG_FILE,
            required=False,
        )
        parser.add_argument(
            "-g",
            "--graphs",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Generate graphs or not",
        )
        parser.add_argument(
            "--max-wall-time",
            type=float,
            default=None,
            help="Cap the run after this many real (wall-clock) seconds. Overrides simulation.max_wall_time_sec.",
            required=False,
        )
        return parser

    def get_parser(self):
        return self.parser

    @classmethod
    def from_cmd_args(cls, argv: list[str] | None = None) -> "OpalSimulator":
        """
        Build a simulator from command line arguments.

        Args:
            argv (list[str], optional): argument list to parse. Defaults to None,
                which makes argparse read sys.argv.

        Returns:
            OpalSimulator: a simulator initialized from the parsed arguments.
        """
        args = vars(cls._build_parser().parse_args(argv))
        config = OpalConfig()
        config.initialize(args["config"])
        if args["max_wall_time"] is not None:
            config._config["simulation"]["max_wall_time_sec"] = args["max_wall_time"]
        return cls(config, output_dir=args["output_dir"], plot_graphs=args["graphs"])

    @classmethod
    def from_config(
        cls, config: OpalConfig, output_dir: str | None = None, plot_graphs: bool = False
    ) -> "OpalSimulator":
        """
        Build a simulator using a specific config.

        Args:
            config (OpalConfig): _description_
            output_dir (str, optional): _description_. Defaults to None.
            plot_graphs (bool, optional): _description_. Defaults to False.

        Returns:
            OpalSimulator: a simulator initialized from the given config.
        """
        return cls(config, output_dir=output_dir, plot_graphs=plot_graphs)

    def run(self, simulation_time: int | None = None):
        runtime, virtual_time = self.sim.run(simulation_time=simulation_time)
        self.process_sim_results()
        if self.config["simulation"]["save_simulation_data"]:
            self.sim.write_simulation_data()
        return runtime, virtual_time

    def _process_per_stage(self):
        stats = self.sim.workload_orchestrator.stage_stats
        log_path = os.path.join(self.sim.output_dir, "simulation.log") if self.sim.output_dir else None
        log_file = open(log_path, "a") if log_path else None
        try:
            for i, s in enumerate(stats):
                header = f"===== stage_{i} ====="
                print(header)
                if log_file is not None:
                    print(header, file=log_file)
                # s.print_simend_stats()
                s.print_summary_results(log_file=log_file)
                if self.plot_graphs:
                    working_dir = os.path.join(
                        self.sim.output_dir,
                        self.sim.workload_orchestrator.get_stage_directory_name(i),
                    )
                    simend_plot(s, self.config, working_dir)
                else:
                    print(f"Not plotting graphs as --no-graphs was set.")
                    print(f"If you want the final graphs, please specify -g / --graphs flag.")
        finally:
            if log_file is not None:
                log_file.close()

    def _process_global_stats(self):
        # here we collect per-stage number and plot a global trend
        # we support generating these three graphs for now
        # QPS-TTFT(mean), QPS-TPOT (mean), and QPS-tokens/sec

        # check how many stages we have where we can plot this stuff
        stages = self.config.get_workflow_stages()
        # check how many of them have target QPS
        valid_stages = []
        for c in stages:
            if ("request_rate" in c["workload_params"]) and (c["workload_params"]["request_rate"] > 0):
                valid_stages.append(c)
        print(valid_stages)
        global_results = []
        for vs in valid_stages:
            # what was the stage's QPS
            qps = c["workload_params"]["request_rate"]
            # what was the stage's TTFT (mean), TPOT (mean), tokens/sec (mean)

    def process_sim_results(self, process_global: bool = False):
        self._process_per_stage()
        if process_global:
            self._process_global_stats()

        print("Opal: Good bye!")
        print("-------------")

    def get_sim_stats(self) -> list[StageStatistics]:
        return self.sim.workload_orchestrator.get_all_stages_statistics()
