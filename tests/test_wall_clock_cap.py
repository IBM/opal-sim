# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from opal.core.opal import OpalSimulator
from opal.config.opal_config import OpalConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_config():
    config = OpalConfig()
    config.initialize(str(CONFIGS_DIR / "defaults.json"))
    return config


def test_wall_clock_cap_truncates_run(tmp_path, monkeypatch):
    """A tiny max_wall_time_sec should stop the run almost immediately,
    well before the workload would finish naturally, and still leave the
    simulator in a clean, marked-done state."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    config = _load_config()
    config._config["simulation"]["max_wall_time_sec"] = 0.001

    opal = OpalSimulator.from_config(config=config, output_dir=str(tmp_path))
    wall_clock_elapsed, virtual_time = opal.run(None)

    assert opal.sim.are_we_done()
    # the workload's own stopping condition (simulation_time == -1) would run
    # for >100 virtual seconds; the wall-clock cap must cut that off early
    assert virtual_time < 100
    assert wall_clock_elapsed < 5.0


def test_no_wall_clock_cap_runs_to_completion(tmp_path, monkeypatch):
    """With max_wall_time_sec left at its default (-1, disabled), behavior
    must be unaffected: the run proceeds until the workload finishes."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    config = _load_config()
    assert config._config["simulation"]["max_wall_time_sec"] == -1.0

    opal = OpalSimulator.from_config(config=config, output_dir=str(tmp_path))
    _, virtual_time = opal.run(10)

    assert opal.sim.are_we_done()
    assert virtual_time == 10
