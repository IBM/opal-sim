# SPDX-License-Identifier: Apache-2.0
import os

# Repository root: the directory containing the `opal` package. Anchored here so
# modules can move between subpackages without recounting ".." levels.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG_FILE = os.path.join(PROJECT_ROOT, "configs", "defaults.json")

# Re-exported below the constants above: opal.core.opal and opal.config.opal_config
# both import PROJECT_ROOT / DEFAULT_CONFIG_FILE from this module, so the constants
# must be bound before these imports run.
from opal.config.opal_config import OpalConfig
from opal.core.opal import OpalSimulator

__all__ = [
    "DEFAULT_CONFIG_FILE",
    "PROJECT_ROOT",
    "OpalConfig",
    "OpalSimulator",
]
