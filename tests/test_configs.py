# SPDX-License-Identifier: Apache-2.0
from pathlib import Path  # noqa: I001

import pytest

from opal.core.opal import OpalSimulator
from opal.config.opal_config import OpalConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def get_config_files():
    return sorted(CONFIGS_DIR.glob("*.json"))


@pytest.fixture(params=get_config_files(), ids=lambda p: p.name)
def config_file(request):
    return request.param


def test_config_loads_and_runs(config_file, tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    config = OpalConfig()
    config.initialize(str(config_file))
    opal = OpalSimulator.from_config(config=config, output_dir=str(tmp_path))
    opal.run(10)
