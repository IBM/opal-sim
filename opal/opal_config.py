# SPDX-License-Identifier: Apache-2.0
from dataclasses import asdict, dataclass
import json
import logging
import os

"""
from collections import defaultdict

def to_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

def nested_dict():
    return defaultdict(nested_dict)

def print_all_defaults(data): 
    import json 
    print(json.dumps(to_dict(data), indent=2))


opal_defaults = nested_dict()

# this should match the config file keys and there could be some validation setup 
opal_defaults["simulation"]["simulation_duration"] = 10 
opal_defaults["simulation"]["seed"] = 42 
"""


@dataclass
class OpalDefaults:
    save_simulation_data: bool = True


class ConfigProxy:
    """
    Proxy for nested config access with default fallback.

    This class enables:
    1. Nested dictionary access: config["simulation"]["save_simulation_data"]
    2. Automatic fallback to defaults when keys are missing
    3. Support for ** unpacking: get_model(**config["model"]["model_params"])

    The proxy implements the mapping protocol (keys, values, items, __iter__, __len__)
    which allows Python's ** operator to unpack it as keyword arguments.
    """

    def __init__(self, config_dict, defaults, log, accessed_keys=None, path_prefix=""):
        self._config = config_dict
        self._defaults = defaults
        self._log = log
        self._accessed_keys = accessed_keys if accessed_keys is not None else set()
        self._path_prefix = path_prefix

    def __getitem__(self, key):
        """
        Access config values with automatic fallback to defaults.

        When accessing nested dicts (e.g., config["simulation"]["save_simulation_data"]):
        1. config["simulation"] returns a ConfigProxy wrapping the "simulation" dict
        2. ["save_simulation_data"] calls this method on the proxy
        3. If key not found in config, falls back to _defaults
        """
        # Track access
        full_path = f"{self._path_prefix}.{key}" if self._path_prefix else key
        self._accessed_keys.add(full_path)

        # Try to get from config first
        if key in self._config:
            value = self._config[key]
            # If it's a dict, return another proxy for further nesting
            if isinstance(value, dict):
                return ConfigProxy(value, self._defaults, self._log, self._accessed_keys, full_path)
            return value
        # Fall back to defaults
        if key in self._defaults:
            self._log.warning(f"Using default for '{key}': {self._defaults[key]}")
            return self._defaults[key]
        # Not found anywhere
        raise KeyError(f"Key '{key}' not found in config or defaults")

    # Mapping protocol methods - required for ** unpacking
    # Python's ** operator requires these methods to treat ConfigProxy as a mapping

    def keys(self):
        """Return keys from the underlying config dict. Required for ** unpacking."""
        return self._config.keys()

    def values(self):
        """Return values from the underlying config dict. Required for ** unpacking."""
        return self._config.values()

    def items(self):
        """Return items from the underlying config dict. Required for ** unpacking."""
        return self._config.items()

    def __iter__(self):
        """Make ConfigProxy iterable. Required for ** unpacking."""
        return iter(self._config)

    def __len__(self):
        """Return length of underlying config dict. Required for ** unpacking."""
        return len(self._config)

    def __setitem__(self, key, value):
        self._config[key] = value

    def get(self, key, default=None):
        """
        Get a value with optional default, like dict.get().

        Args:
            key: The key to look up
            default: Value to return if key not found (default: None)

        Returns:
            The value if found, otherwise the default
        """
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key):
        """Support 'in' operator: if key in config."""
        return key in self._config or key in self._defaults


class OpalConfig:
    def __init__(self):
        self.log = logging.getLogger("OpalConfig")
        self.default_config_file = os.path.normpath(os.path.join(os.path.dirname(__file__), "../configs/defaults.json"))
        self._config = None
        self._defaults = asdict(OpalDefaults())
        self._accessed_keys = set()  # Track which config keys are accessed

    def sanitize_workload_stage(self, stage):
        # check both time and num_requests can not be -1
        if "total_requests" not in stage["workload_params"]:
            stage["workload_params"]["total_requests"] = -1

        if "time_duration_sec" not in stage["workload_params"]:
            stage["workload_params"]["time_duration_sec"] = -1

        if not (
            self._config["simulation"]["simulation_time"] > 0
            or stage["workload_params"]["total_requests"] > 0
            or stage["workload_params"]["time_duration_sec"] > 0
        ):
            # this happens, then this must be moonshot type workload
            workload = stage["type"].lower()
            # FIXME: put this as the workload parameter that each workload defines for itself
            if not (workload == "trace".casefold() or workload == "SC25Workload".casefold()):
                str = (
                    f"total_request = -1 and simulation_time = -1, are invalid combination for this {workload} workload"
                )
                return False, str
        # FIXME: add other conditions one by one
        return True, ""

    def get_workflow_stages(self):
        # backward compatible parsing
        return (
            self._config["workload"]["stages"] if "stages" in self._config["workload"] else [self._config["workload"]]
        )

    def sanitize_config(self):
        stages = self.get_workflow_stages()
        for s in stages:
            passed, reason = self.sanitize_workload_stage(s)
            if not passed:
                self.log.error(f"Failed stage sanity check for {s}")
            assert passed, reason

    def initialize(self, config_file: str | None = None):
        """
        Load a simulation config JSON configuration file and return it as a Python dictionary.
        """
        if config_file == None:
            self.log.warning(f"Using the default config file at {self.default_config_file}")
            config_file = self.default_config_file

        with open(config_file, "r") as f:
            self._config = json.load(f)
        # after init - check and sanitize config
        self.sanitize_config()

    def print_config(self):
        print(f"{self._config}")

    def __call__(self, key: str, sep: str = "/"):
        val = self.get_value(key, sep)
        return val

    def __getitem__(self, key):
        # Track access
        self._accessed_keys.add(key)

        # Try to get from config first
        if key in self._config:
            value = self._config[key]
            # If it's a dict, return a proxy for further nesting
            if isinstance(value, dict):
                return ConfigProxy(value, self._defaults, self.log, self._accessed_keys, key)
            return value
        # Fall back to defaults
        if key in self._defaults:
            self.log.debug(f"Using default for '{key}': {self._defaults[key]}")
            return self._defaults[key]
        # Not found anywhere
        raise KeyError(f"Key '{key}' not found in config or defaults")

    def __str__(self):
        return json.dumps(self._config, indent=4)

    def get_value(self, key: str, sep: str = "/"):
        def get_nested(d, keys):
            for k in keys:
                d = d[k]
            return d

        # Track access with dot notation
        dot_key = key.replace(sep, ".")
        self._accessed_keys.add(dot_key)

        split_key = key.split(sep)
        return get_nested(self._config, split_key)

    def save(self, file: str):
        f = open(file, "w")
        f.write(str(self))
        f.close()

    def _get_all_config_keys(self, config_dict=None, prefix=""):
        """
        Recursively get all keys in the config with their full paths.

        Args:
            config_dict: Dictionary to traverse (defaults to self._config)
            prefix: Current path prefix for nested keys

        Returns:
            Set of all config key paths (e.g., {"simulation.time", "workload.type"})
        """
        if config_dict is None:
            config_dict = self._config

        all_keys = set()

        for key, value in config_dict.items():
            full_path = f"{prefix}.{key}" if prefix else key
            all_keys.add(full_path)

            # Recursively process nested dicts
            if isinstance(value, dict):
                all_keys.update(self._get_all_config_keys(value, full_path))
            # Handle lists that might contain dicts
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, dict):
                        list_path = f"{full_path}.{idx}"
                        all_keys.update(self._get_all_config_keys(item, list_path))

        return all_keys

    def report_unused_config(self, log_warnings=True):
        """
        Report config values that were loaded but never accessed.

        Args:
            log_warnings: If True, log warnings for each unused key

        Returns:
            Dictionary with:
                - 'unused_keys': List of unused config key paths
                - 'total_keys': Total number of config keys
                - 'accessed_keys': Number of accessed keys
                - 'unused_count': Number of unused keys
        """
        if self._config is None:
            self.log.warning("Config not initialized. Call initialize() first.")
            return {"unused_keys": [], "total_keys": 0, "accessed_keys": 0, "unused_count": 0}

        # Get all available keys
        all_keys = self._get_all_config_keys()

        # Find unused keys
        unused_keys = all_keys - self._accessed_keys

        # Sort for consistent output
        unused_keys_list = sorted(unused_keys)

        # Log warnings if requested
        if log_warnings and unused_keys_list:
            self.log.warning(f"Found {len(unused_keys_list)} unused config values:")
            for key in unused_keys_list:
                self.log.warning(f"  - {key}")

        return {
            "unused_keys": unused_keys_list,
            "total_keys": len(all_keys),
            "accessed_keys": len(self._accessed_keys),
            "unused_count": len(unused_keys_list),
        }

    def get_accessed_keys(self):
        """Return the set of config keys that have been accessed."""
        return self._accessed_keys.copy()

    def reset_access_tracking(self):
        """Reset the access tracking. Useful for testing or multi-stage simulations."""
        self._accessed_keys.clear()


if __name__ == "__main__":
    config = OpalConfig()
    config.initialize("sim_config/defaults.json")
    print(config)  # pretty print

    # Example: access specific parameters
    duration = config["simulation"]["simulation_time"]
    print(f"Simulation will run for {duration} time units.")

    # Demonstrate usage tracking
    print("\n" + "=" * 60)
    print("Config Usage Tracking Report")
    print("=" * 60)
    report = config.report_unused_config(log_warnings=True)
    print(f"\nSummary:")
    print(f"  Total config keys: {report['total_keys']}")
    print(f"  Accessed keys: {report['accessed_keys']}")
    print(f"  Unused keys: {report['unused_count']}")
