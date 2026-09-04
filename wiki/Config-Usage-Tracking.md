# Config Usage Tracking

## Overview

The `OpalConfig` class now includes functionality to track which configuration values are accessed during a simulation run and report any unused values. This helps identify:

- Dead/deprecated configuration parameters
- Configuration bloat
- Potential configuration errors (values that should be used but aren't)

## How It Works

The tracking system monitors all config access through:
1. `config["key"]` - Direct dictionary access
2. `config["nested"]["key"]` - Nested access via ConfigProxy
3. `config.get_value("path/to/key")` - Path-based access
4. `**config["section"]` - Dictionary unpacking

Each access is recorded with its full path (e.g., `"simulation.seed"`, `"workload.stages.0.type"`).

## Usage

### Basic Usage

```python
from opal_config import OpalConfig

# Initialize config
config = OpalConfig()
config.initialize("path/to/config.json")

# Use config values as normal
duration = config["simulation"]["simulation_time"]
seed = config.get_value("simulation/seed")

# At the end of your simulation, generate report
report = config.report_unused_config(log_warnings=True)

print(f"Total keys: {report['total_keys']}")
print(f"Accessed: {report['accessed_keys']}")
print(f"Unused: {report['unused_count']}")
print(f"Unused keys: {report['unused_keys']}")
```

### Report Structure

The `report_unused_config()` method returns a dictionary with:

```python
{
    'unused_keys': ['list', 'of', 'unused.key.paths'],
    'total_keys': 100,           # Total config keys
    'accessed_keys': 75,         # Number accessed
    'unused_count': 25           # Number unused
}
```

### Additional Methods

```python
# Get set of accessed keys
accessed = config.get_accessed_keys()

# Reset tracking (useful for multi-stage simulations)
config.reset_access_tracking()
```

## Integration Example

Add to your main simulation script:

```python
def main():
    config = OpalConfig()
    config.initialize(args.config_file)
    
    # Run simulation
    run_simulation(config)
    
    # Report unused config at the end
    print("\n" + "="*60)
    print("Configuration Usage Report")
    print("="*60)
    report = config.report_unused_config(log_warnings=True)
    
    if report['unused_count'] > 0:
        print(f"\nWarning: {report['unused_count']} config values were never used")
        print("Consider removing them or checking if they should be used.")
```

## Example Output

Illustrative only — `kvc.Scale` and `storage.DFSBackend` below are sample
names, not real sections in [[Configuration Simulation]]; the real tiers are
`CPUMemory`/`LocalNVMe`/`DistributedFS` under `kvc`.

```
============================================================
Configuration Usage Report
============================================================
WARNING:OpalConfig:Found 15 unused config values:
WARNING:OpalConfig:  - kvc.LocalNVMe.bandwidth_GBps
WARNING:OpalConfig:  - kvc.LocalNVMe.capacity_GB
WARNING:OpalConfig:  - kvc.LocalNVMe.concurrency
WARNING:OpalConfig:  - kvc.LocalNVMe.latency_nsec
WARNING:OpalConfig:  - kvc.Scale.bandwidth_GBps
WARNING:OpalConfig:  - kvc.Scale.capacity_GB
WARNING:OpalConfig:  - kvc.Scale.concurrency
WARNING:OpalConfig:  - kvc.Scale.latency_nsec
WARNING:OpalConfig:  - router.router_params.max_event_batch_size
WARNING:OpalConfig:  - storage.DFSBackend.max_bandwidth
WARNING:OpalConfig:  - storage.DFSBackend.min_latency
WARNING:OpalConfig:  - workload.stages.0.workload_params.jitter
WARNING:OpalConfig:  - workload.stages.1.workload_params.chunk_size
WARNING:OpalConfig:  - workload.stages.1.workload_params.multiplier_to_sec
WARNING:OpalConfig:  - worker.inference_params.mean_latency_secs

Summary:
  Total config keys: 75
  Accessed keys: 60
  Unused keys: 15
```

## Implementation Details

### Tracking Mechanism

- **ConfigProxy**: Tracks nested access by maintaining a path prefix and shared `_accessed_keys` set
- **OpalConfig**: Tracks top-level access and provides the reporting interface
- **Path Format**: Uses dot notation (e.g., `"simulation.seed"`) for consistency

### Performance Impact

- Minimal overhead: Only set operations for tracking
- Memory: O(n) where n is number of unique accessed keys
- No impact on config access speed

### Limitations

1. **Direct `_config` access**: If code directly accesses `config._config["key"]`, it won't be tracked
2. **Dynamic keys**: Keys constructed at runtime may not be tracked accurately
3. **Conditional usage**: A key might be intentionally unused based on other config values

## Best Practices

1. **Run at simulation end**: Call `report_unused_config()` after all config access is complete
2. **Review regularly**: Use reports to clean up deprecated config parameters
3. **Document intentional unused keys**: Some keys may be optional or conditional
4. **Multi-stage simulations**: Use `reset_access_tracking()` between stages if needed

## Testing

Run the test script to see the feature in action:

```bash
python test_config_tracking.py
```

Or test directly:

```bash
python opal/opal_config.py