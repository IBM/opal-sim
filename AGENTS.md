# AGENTS.md - Guide for AI Agents Working on OPAL

**Last Updated:** 2026-05-11
**Purpose:** Essential information for AI agents (like Claude, GPT, etc.) working on the OPAL simulator codebase

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Overview](#project-overview)
3. [Code Structure](#code-structure)
4. [Running the Simulator](#running-the-simulator)
5. [Configuration System](#configuration-system)
6. [Key Components](#key-components)
7. [Development Workflow](#development-workflow)
8. [Testing](#testing)
9. [Important Patterns](#important-patterns)
10. [Common Tasks](#common-tasks)
11. [Debugging Tips](#debugging-tips)

---

## Quick Start

### Environment Setup
```bash
# Clone repository
git clone git@github.com:IBM/opal-sim.git
cd opal-sim

# setup or run from uv directly 
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# (Optional) enable the repo's pre-commit hook (Black formatter)
git config core.hooksPath .githooks
```

### Run Basic Simulation
```bash
# From project root (main.py self-extends sys.path; PYTHONPATH not required)
python ./opal/main.py

# With custom config and graphs
python ./opal/main.py -c ./configs/defaults.json -g

# With debug logging
OPAL_LOG_LEVEL=DEBUG python ./opal/main.py
```

### Run Tests
```bash
# All tests
pytest

# Specific test with verbose output
OPAL_LOG_LEVEL=DEBUG pytest -s -v ./tests/test_configs.py

# With output shown
pytest -s -v
```

---

## Project Overview

**OPAL** (Open simulator Platform for distributed AI and LLM workflows) is a discrete-event simulator for LLM inference platforms written in Python using SimPy.

### Key Features
- **Discrete-event simulation** using SimPy framework
- **vLLM worker modeling** with batching and scheduling
- **Distributed KV cache** management with tiering
- **Storage backends** (DFS, fixed latency, custom)
- **Workload generation** (uniform, exponential, trace replay)
- **Router with autoscaling** support
- **GPU modeling** with utilization tracking

### Why Simulation?
- **Cost:** Avoid expensive GPU infrastructure for exploration
- **Speed:** Fast iteration on design decisions
- **Complexity:** Explore policy space without full implementation

---

## Code Structure

```
opal-sim/
├── opal/                          # Main package
│   ├── __init__.py
│   ├── main.py                    # Entry point
│   ├── opal.py                    # OpalSimulator class
│   ├── opal_base.py               # Base classes (NoDynamicAttributes)
│   ├── environment.py             # OpalSimulatorEnvironment (SimPy env wrapper)
│   ├── opal_config.py             # Configuration system (ConfigProxy)
│   ├── opal_logging.py            # Logging utilities
│   ├── opal_profile.py            # Performance profiling decorator
│   ├── opal_registery.py          # OpalRegistry (worker singleton registry)
│   ├── defaults.py                # Reserved for future default values
│   │
│   ├── vllm_worker.py             # ⭐ LLMWorkerVLLMScheduler (1700+ lines)
│   ├── router.py                  # Router with policies
│   ├── autoscaling.py             # Autoscaling logic
│   ├── workload_orchestrator.py   # WorkloadOrchestrator (multi-stage)
│   ├── kvc_manager.py             # KV cache: OpalStorageBackend, OpalStorageManager
│   ├── kvbm.py                    # KV block manager
│   ├── gpu_model.py               # GPUModel (roofline/synthetic inference)
│   ├── io_model.py                # I/O: CPUMemory, LocalNVMe, DistributedFS devices
│   ├── llm_model.py               # OpalModelConfig (HF or local config loading)
│   │
│   ├── request.py                 # LLMRequest, LLMRequestStats, IORequest
│   ├── datatypes.py               # STR_DTYPE_TO_BYTES (dtype → byte-size map)
│   ├── events.py                  # KVCEvent, SystemEvent, OpalInfraEvent
│   ├── stage_statistics.py        # StageStatistics collection
│   ├── plot.py                    # Plotting utilities
│   ├── util.py                    # safe_process(), check_and_create_directory(), etc.
│   │
│   ├── workloads/                 # Workload generators
│   │   ├── abstract_workload.py   # AbstractWorkload base class
│   │   ├── workload.py            # UniformReqRate, ExponentialReqRate, Trace
│   │   └── sc25_blog.py           # SC25Workload
│   │
│   └── regression-fitting/        # Model calibration (a, b parameters)
│       ├── offline_calc_a_b.py
│       ├── online_calibration.py
│       └── README.md
│
├── configs/                       # Configuration files
│   ├── defaults.json              # Default configuration (local model dir)
│   └── hf.json                    # Variant that pulls model config from Hugging Face
│
├── model-configs/                 # Local model configurations
│   └── granite-3.3-8b-instruct/
│       └── config.json
│
├── traces/                        # Trace files for replay workloads
│   └── hello.jsonl                # Example trace
│
├── tests/                         # Unit tests
│   └── test_configs.py            # Parametrized config loading + run tests
│
├── wiki/                          # Documentation (cloned from GitHub wiki)
│   ├── Configuration-Simulation.md
│   ├── vllm-worker.md
│   ├── vLLM-modeling.md
│   ├── KVCache-manager.md
│   ├── Router.md
│   ├── Worker.md
│   ├── Workload-generation.md
│   ├── Running.md
│   ├── Running-Workloads.md
│   ├── Install-Requirements.md
│   ├── Config-Usage-Tracking.md
│   ├── Examples.md
│   ├── Home.md
│   ├── Overview.md
│   ├── README.md
│   ├── _Sidebar.md
│   └── opal-overview.png
│
├── simulation-runs/               # Output directory (gitignored)
│   └── sim-YYYY-MM-DD_HH_MM_SS/
│
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata & pytest config
├── sh-black-formatter.sh          # Code formatter script
├── AGENTS.md                      # This file
└── README.md                      # Main documentation
```

### Key Files to Understand

1. **opal/vllm_worker.py** (1700+ lines)
   - `LLMWorkerVLLMScheduler` class - core vLLM worker implementation
   - Interrupt-driven scheduling loop (`_vllm_scheduling_loop`)
   - Batch building and GPU memory management
   - KV cache integration

2. **opal/router.py** (~280 lines)
   - `Router` class with 5 policies: RoundRobin, LeastLoaded, Random, MaxPrefix, Balanced
   - Worker autoscaling triggers

3. **opal/kvc_manager.py** (~1070 lines)
   - `OpalStorageBackend` - per-tier storage simulation
   - `OpalStorageManager` - tiering logic across CPUMemory, LocalNVMe, DistributedFS
   - `OpalTokenDatabase` - prefix-based KV cache lookup (chunked)

4. **opal/gpu_model.py**
   - `GPUModel` class with two inference engines: `roofline` (analytical) and `synthetic` (fixed latency)
   - Roofline uses model params (a, b) for prefill/decode timing

5. **opal/util.py**
   - `safe_process()` - SimPy process wrapper with error handling
   - `check_and_create_directory()`, `get_with_timeout()`, `parse_bandwidth()`

6. **opal/opal_config.py**
   - `OpalConfig` - loads JSON config
   - `ConfigProxy` - nested access with default fallback and `**` unpacking support

---

## Running the Simulator

### Basic Execution
```bash
# Default config (./configs/defaults.json)
python ./opal/main.py

# Custom config with graphs
python ./opal/main.py -c ./configs/defaults.json -g

# Specify output directory
python ./opal/main.py -o ./my-results/
```

### Command Line Arguments
```bash
python ./opal/main.py --help

Options:
  -c, --config PATH                    Configuration file (default: configs/defaults.json)
  -o, --output-dir PATH                Output directory (default: ./simulation-runs/)
  -g, --graphs / --no-graphs           Generate graphs (default: False)
```

### Environment Variables
```bash
# Logging level: DEBUG, INFO, WARN, ERROR
OPAL_LOG_LEVEL=DEBUG

# Log format: 0 (minimal), 1 (standard), 2 (verbose)
OPAL_LOG_FORMAT=1

# Disable colored output
OPAL_NO_COLOR=1

# Example with all options
OPAL_LOG_LEVEL=DEBUG OPAL_LOG_FORMAT=2 python ./opal/main.py
```

### Output Structure
```
simulation-runs/sim-2026-04-02_14_30_45/
├── sim_config.json              # Config used for this run
├── simulation.log               # Full simulation log
├── stage_0/                     # First workload stage
│   ├── opal_stats.json         # Statistics
│   ├── cdf-latencies.pdf       # CDF plots
│   ├── gpu-utilization-per-sec.pdf
│   ├── histo-latencies.pdf
│   ├── thrp-request-sec.pdf
│   └── thrp-workers-sec.pdf
└── stage_1/                     # Second workload stage
    └── ...
```

---

## Configuration System

### Configuration File Structure
```jsonc
{
  "simulation": {
    "simulation_time": -1.0,       // -1 = run until workload completes
    "max_wall_time_sec": -1.0,     // -1 = no real-time cap; else stop after N wall-clock seconds
    "seed": 42,
    "num_workers": 1,
    "save_simulation_data": true,
    "show_progress": false
  },
  "model": {
    "model_params": {
      "name": "granite-3.3-8b-instruct",
      "config_dir": "./model-configs/"  // directory containing model subfolders
    }
  },
  "router": {
    "router_params": {
      "enable_scaling": false,
      "max_queue_threshold": 4,
      "scale_latency": 40,
      "max_workers": 50,
      "policy": "MaxPrefix",       // RoundRobin, LeastLoaded, Random, MaxPrefix, Balanced
      "periodic_infra_update_collection_time": 30,
      "max_event_batch_size": 64
    }
  },
  "workload": {
    "stages": [
      {
        "type": "UniformReqRate",  // or "ExponentialReqRate", "Trace", "SC25Workload"
        "workload_params": {
          "request_rate": 2.0,
          "total_requests": 100,
          "prompt_size_min": 32,
          "prompt_size_max": 16384,
          "default_prefix_length": 1024,
          "jitter": 0.0,
          "output_tokens_min": 32,
          "output_tokens_max": 128
        }
      },
      {
        "type": "trace",
        "workload_params": {
          "total_requests": 10,
          "chunk_size": 1,
          "multiplier_to_sec": 0.001,
          "trace_file": "traces/hello.jsonl"
        }
      }
    ]
  },
  "worker": {
    "worker_params": {
      "worker_local_queue_capacity": 1,
      "periodic_infra_update_time": 30,
      "kvcevent_coalesce_time": 30
    },
    "hw": {
      "gpu": "H100",
      "memory_gb": 80,
      "tflops": 989.5,
      "mem_bw_TBps": 3.3,
      "tp": 1
    },
    "vllm_params": {
      "max_num_seqs": 256,
      "max_num_batched_tokens": 8192,
      "max_kvc_ready_requests": 8,
      "chunked_prefill": true,
      "block_size": 16
    },
    "inference_params": {
      "model": "roofline",         // or "synthetic"
      "mean_latency_secs": 1.0,   // only for synthetic
      "a": "4",                    // roofline params
      "b": "24"
    }
  },
  "kvc": {
    "kvc_tiers": ["CPUMemory"],    // tier names: "CPUMemory", "LocalNVMe", "DistributedFS"
    "chunk_size": 256,
    "save_unfull_chunk": true,
    "CPUMemory": {
      "bandwidth_GBps": 50,
      "latency_nsec": 200,
      "concurrency": 1000000,
      "capacity_GB": 8
    },
    "LocalNVMe": {
      "bandwidth_GBps": 10,
      "latency_nsec": 10000,
      "concurrency": 1000,
      "capacity_GB": 1024
    },
    "DistributedFS": {
      "bandwidth_GBps": 100,
      "latency_nsec": 100000,
      "concurrency": 1000,
      "capacity_GB": 1048576
    }
  }
}
```

### Key Configuration Parameters

**Simulation:**
- `simulation_time`: Virtual seconds to run (-1 = until workload done)
- `max_wall_time_sec`: Real (wall-clock) seconds cap (-1 = disabled); overridable via `--max-wall-time` CLI flag
- `seed`: Random seed for reproducibility
- `num_workers`: Initial worker count

**Workload Types:**
- `UniformReqRate`: Uniform request arrival rate
- `ExponentialReqRate`: Poisson arrival (exponential inter-arrival)
- `Trace`: Replay from JSONL trace file
- `SC25Workload`: SC25 blog workload pattern

**Inference Models (worker.inference_params.model):**
- `roofline`: Analytical model using GPU FLOPS, memory bandwidth, and model params (a, b)
- `synthetic`: Fixed latency per step (for testing/debugging)

**KVC Tiers (kvc.kvc_tiers):**
- `CPUMemory`: CPU DRAM tier
- `LocalNVMe`: Local NVMe storage tier
- `DistributedFS`: Distributed file system tier (slowest)

---

## Key Components

### 1. LLMWorkerVLLMScheduler (opal/vllm_worker.py)

**Main Process:** `_vllm_scheduling_loop()`
- Interrupt-driven architecture (no polling loops)
- Sleeps indefinitely when idle: `yield self.simpy_env.timeout(float("inf"))`
- Woken up by interrupts from `queue_work()` or `_check_new_requests()`

**Key Methods:**
```python
def queue_work(request):
    """Entry point for new requests from router"""
    # Adds to worker_local_queue
    # Interrupts _check_new_request_process to wake it up

def _check_new_requests():
    """Moves requests from worker_local_queue to waiting_requests"""
    # Sleeps indefinitely when queue empty
    # Interrupts scheduler to wake it up

def _vllm_scheduling_loop():
    """Main scheduler loop"""
    # Calls _scheduler_step() inline with yield from
    # Catches interrupts during active work
    # Sleeps indefinitely when no work

def _scheduler_step():
    """One scheduling iteration"""
    # Builds batch with yield from _build_batch()
    # Processes batch
    # Handles completions

def _build_batch():
    """Creates batch from waiting/running requests"""
    # GPU memory accounting
    # Batch size limits
    # Returns VLLMBatch
```

**Important Patterns:**
- `yield from self._scheduler_step()` - Inline execution, interrupt control
- `yield safe_process(env, coroutine())` - Separate process, isolation

### 2. Router (opal/router.py)

**Routing Policies (case-insensitive in config):**
- `Random`: Random worker selection
- `RoundRobin`: Round-robin distribution
- `LeastLoaded`: Route to worker with smallest queue
- `MaxPrefix`: Route to worker with max prefix match (KV cache aware)
- `Balanced`: Balanced distribution

**Autoscaling:**
- Monitors queue lengths (`max_queue_threshold`)
- Adds workers when threshold exceeded
- Scale-up latency configurable (`scale_latency`)

### 3. KVC Manager (opal/kvc_manager.py)

**Key Classes:**
- `OpalStorageBackend`: Per-tier storage simulation (bandwidth, latency, concurrency, capacity)
- `OpalStorageManager`: Multi-tier orchestration (fills tiers in order, spills to next)
- `OpalTokenDatabase`: Abstract prefix-based token lookup (chunked token database logic)

**Operations:**
- `batched_get(keys)`: Retrieve KV cache entries from tiered storage
- `batched_put(keys, values)`: Store entries, filling tiers in order

**Tiering (configured via `kvc.kvc_tiers` list):**
1. `CPUMemory` - CPU DRAM (fastest)
2. `LocalNVMe` - Local NVMe storage
3. `DistributedFS` - Distributed file system (slowest)

Each tier is configured with: `bandwidth_GBps`, `latency_nsec`, `concurrency`, `capacity_GB`

### 4. I/O Model (opal/io_model.py)

**Device Classes (inherit from `AbstractDevice`):**
- `CPUMemory`: CPU DRAM device simulation
- `LocalNVMe`: Local NVMe device simulation
- `DistributedFS`: Distributed file system simulation

Each device models bandwidth sharing among concurrent requests and minimum latency.

---

## Development Workflow

### Code Formatting
```bash
# Install black
pip install black

# Format code (required before PR)
./sh-black-formatter.sh

# Or manually
black --line-length 120 opal/ tests/
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/my-feature

# Make changes, format code
./sh-black-formatter.sh

# Commit with co-author
git add .
git commit -m "Add feature X

- Detail 1
- Detail 2

Co-Authored-By: <put an appropriate sign-off with email>"

# Push and create PR
git push origin feature/my-feature
```

### Pre-commit Hook
The repository ships a pre-commit hook at `.githooks/pre-commit` that runs Black
automatically. It is **not** installed by default — enable it once per clone with:

```bash
git config core.hooksPath .githooks
```

---

## Testing

### Test Structure
```
tests/
└── test_configs.py            # Parametrized config loading + run tests
```

The test is parametrized over every `*.json` file in `configs/`, so adding a new
config file automatically adds a new test case.

### Running Tests
```bash
# All tests (auto-discovered via pyproject.toml: python_files = ["test_*.py"])
pytest

# Run the config test explicitly
pytest -s -v tests/test_configs.py

# With debug logging
OPAL_LOG_LEVEL=DEBUG pytest -s -v tests/test_configs.py
```

### Writing Tests
```python
import pytest
from pathlib import Path
from opal.opal import OpalSimulator
from opal.opal_config import OpalConfig

def test_my_feature(tmp_path, monkeypatch):
    """Test that loads a config, runs the sim for a short time."""
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    config = OpalConfig()
    config.initialize("./configs/defaults.json")
    opal = OpalSimulator()
    opal.init_from_config(config=config, output_dir=str(tmp_path))
    opal.run(10)  # run for 10 virtual seconds
```

---

## Important Patterns

### 1. SimPy Discrete-Event Simulation

**Process Definition:**
```python
def my_process(env):
    """A SimPy process"""
    print(f"Starting at {env.now}")
    yield env.timeout(5.0)  # Wait 5 virtual seconds
    print(f"Finished at {env.now}")

# Start process
env.process(my_process(env))
```

**Interrupt-Driven Pattern:**
```python
def worker_loop(env):
    """Interrupt-driven worker"""
    while True:
        try:
            # Sleep indefinitely until interrupted
            yield env.timeout(float("inf"))
        except simpy.Interrupt:
            # Woken up, do work
            process_work()

# Interrupt to wake up
worker_process.interrupt()
```

### 2. yield vs yield from

**Use `yield safe_process()` for:**
- Cross-component calls (Router → Worker)
- I/O operations (Worker → KVC Manager)
- When you want process isolation

```python
# Separate process, waits for completion
result = yield safe_process(env, other_component.method())
```

**Use `yield from` for:**
- Internal method calls within same component
- When you need interrupt control
- Performance-critical paths

```python
# Inline execution, interrupts propagate
try:
    yield from self._internal_method()
except simpy.Interrupt:
    # Can catch interrupts from inside method
    pass
```

**Refer to `opal/util.py` for the `safe_process` implementation and inline documentation.**

### 3. Configuration Access

```python
# Access nested config with automatic defaults
value = self.config["simulation"]["simulation_time"]

# Unpack config section as kwargs
model = LLMModel(**self.config["model"]["model_params"])

# Check if key exists
if "optional_param" in self.config["section"]:
    value = self.config["section"]["optional_param"]
```

### 4. Logging

```python
from opal.opal_logging import get_logger

class MyClass:
    def __init__(self):
        self.log = get_logger(self.__class__.__name__)
    
    def my_method(self):
        self.log.debug("Debug message")
        self.log.info("Info message")
        self.log.warning("Warning message")
        self.log.error("Error message")
```

### 5. Statistics Collection

```python
# Request statistics
request.stats.arrival_time = env.now
request.stats.ttft = env.now - request.stats.arrival_time
request.stats.completion_time = env.now

# Stage statistics
stage_stats = self.opal_env.workload_orchestrator.get_active_stage_stats()
stage_stats.queued_requests += 1
stage_stats.completed_requests += 1
```

---

## Common Tasks

### Adding a New Workload Type

1. Create class in `opal/workloads/` inheriting from `AbstractWorkload`:
```python
from opal.workloads.abstract_workload import AbstractWorkload

class MyWorkload(AbstractWorkload):
    def __init__(self, opal_env, stage_index, stage_config, router):
        super().__init__(opal_env, stage_index, stage_config, router)
    
    def run(self):
        """SimPy generator — yield events to generate requests"""
        for i in range(self.total_requests):
            request = self._create_request(...)
            yield self.simpy_env.process(self.router.route_request(request))
            yield self.simpy_env.timeout(inter_arrival_time)
```

2. Place in `opal/workloads/` folder — `WorkloadOrchestrator` discovers classes by name
   via `load_class_from_folder("workloads", stage_type)` (case-insensitive match).

3. Use in config:
```json
{
  "workload": {
    "stages": [{
      "type": "MyWorkload",
      "workload_params": { ... }
    }]
  }
}
```

### Adding a New Routing Policy

1. Add method to `Router` class in `opal/router.py`:
```python
def _policy_my_policy(self, req: LLMRequest | None = None):
    """My custom routing policy — returns a worker"""
    # Select and return a worker
    return self._select_best_worker(req)
```

2. Register in `Router._get_policy_func()`:
```python
elif policy_lower == "mypolicy":
    return self._policy_my_policy
```

3. Use in config:
```json
{
  "router": {
    "router_params": {
      "policy": "MyPolicy"
    }
  }
}
```

### Debugging a Simulation

1. **Enable debug logging:**
```bash
OPAL_LOG_LEVEL=DEBUG python ./opal/main.py
```

2. **Add breakpoints:**
```python
import pdb; pdb.set_trace()
```

3. **Check simulation log:**
```bash
tail -f simulation-runs/sim-*/simulation.log
```

4. **Inspect statistics:**
```bash
# After run
cat simulation-runs/sim-*/stage_0/opal_stats.json | jq
```

5. **Use synthetic inference model for deterministic behavior:**
```json
{
  "worker": {
    "inference_params": {
      "model": "synthetic",
      "mean_latency_secs": 1.0
    }
  }
}
```

---

## Debugging Tips

### Common Issues

**1. Import Errors**
```bash
# main.py self-extends sys.path, so this is normally not needed.
# Set PYTHONPATH only if you invoke modules directly with `python -m ...`
PYTHONPATH=`pwd`:$PYTHONPATH python ./opal/main.py
```

**2. SimPy Interrupt Errors**
- See the "yield vs yield from" pattern below for correct usage
- Ensure interrupts are caught in try-except blocks
- Don't interrupt during critical sections

**3. Configuration Errors**
```python
# Check config loading
sim = OpalSimulator()
sim.init_from_config(config)
print(json.dumps(sim.config._config, indent=2))
```

**4. Race Conditions**
- Use `yield from` for inline execution when order matters
- Use `yield safe_process()` for independent operations
- Check interrupt handling in `vllm_worker.py`

**5. Memory Issues**
- Check GPU memory accounting in `_build_batch()`
- Verify KV cache eviction logic
- Monitor `allocated_blocks` and `free_blocks`

### Useful Debug Commands

```bash
# Find all TODOs
grep -r "TODO" opal/

# Find all FIXMEs
grep -r "FIXME" opal/

# Check for print statements (should use logging)
grep -r "print(" opal/ --exclude-dir=__pycache__

# Find all yield statements in the worker
grep -r "yield " opal/vllm_worker.py

# Run tests
pytest -s -v tests/test_configs.py
```

### Performance Profiling

Use the built-in `opal/opal_profile.py` decorator:
```python
from opal.opal_profile import profile_function

# Wrap any callable — prints top 20 cumulative-time functions
profile_function(lambda: sim.run())
```

---

## Additional Resources

### Documentation (wiki/ directory)
- **Configuration:** `wiki/Configuration-Simulation.md`
- **vLLM Worker:** `wiki/vllm-worker.md`
- **vLLM Modeling:** `wiki/vLLM-modeling.md`
- **KV Cache Manager:** `wiki/KVCache-manager.md`
- **Router:** `wiki/Router.md`
- **Workloads:** `wiki/Workload-generation.md`
- **Running:** `wiki/Running.md`, `wiki/Running-Workloads.md`
- **Config Tracking:** `wiki/Config-Usage-Tracking.md`
- **Examples:** `wiki/Examples.md`

### External References
- **Wiki (online):** https://github.com/IBM/opal-sim/wiki
- **SimPy Documentation:** https://simpy.readthedocs.io/
- **vLLM:** https://github.com/vllm-project/vllm
- **LMCache:** https://github.com/LMCache/LMCache

---

## Quick Reference

### File Locations
```
Entry point:           opal/main.py
Main simulator:        opal/opal.py
vLLM worker:           opal/vllm_worker.py
Router:                opal/router.py
Autoscaling:           opal/autoscaling.py
Workload orchestrator: opal/workload_orchestrator.py
KVC manager:           opal/kvc_manager.py
KV block manager:      opal/kvbm.py
GPU model:             opal/gpu_model.py
I/O model:             opal/io_model.py
LLM model:             opal/llm_model.py
Config system:         opal/opal_config.py
Utilities:             opal/util.py
Default config:        configs/defaults.json
Model configs:         model-configs/
Traces:                traces/
Tests:                 tests/
Documentation:         wiki/
```

### Key Classes
```
OpalSimulator              - Main simulator (CLI args, run loop)
OpalSimulatorEnvironment   - SimPy environment wrapper, component init
LLMWorkerVLLMScheduler     - vLLM worker (scheduling, batching, KVC)
Router                     - Request routing (5 policies)
WorkloadOrchestrator       - Multi-stage workload management
OpalStorageManager         - KVC tiering orchestration
OpalStorageBackend         - Per-tier storage simulation
OpalTokenDatabase          - Prefix-based KV cache lookup
GPUModel                   - GPU inference timing (roofline/synthetic)
OpalModelConfig            - LLM model configuration (HF or local)
OpalConfig / ConfigProxy   - Config loading and nested access
LLMRequest                 - Request data structure
LLMRequestStats            - Per-request statistics
AbstractWorkload           - Workload base class
StageStatistics            - Per-stage statistics collection
OpalRegistry               - Worker singleton registry
```

### Environment Variables
```
OPAL_LOG_LEVEL       - DEBUG, INFO, WARN, ERROR
OPAL_LOG_FORMAT      - 0 (minimal), 1 (standard), 2 (verbose)
OPAL_NO_COLOR        - Disable colored output
PYTHONPATH           - Must include project root
```

---

**For AI Agents:** This document provides the essential context needed to understand and work with the OPAL simulator codebase. When making changes:

1. Read relevant documentation in `wiki/` first
2. Run tests after changes: `pytest -s -v tests/test_configs.py`
3. Format code with Black: `./sh-black-formatter.sh`
4. Add co-author to commits
5. Update this file if adding new patterns or components

**Last Updated:** 2026-05-11
**Maintainer:** IBM Research