# SimPy Basics

Opal is built on [SimPy](https://simpy.readthedocs.io/), a Python discrete-event simulation
library. This page is not a general SimPy tutorial — it explains the small set of SimPy
mechanisms Opal actually uses, with real excerpts from this codebase, so you can read
`opal/worker/vllm_worker.py`, `opal/router/router.py`, and the workload/orchestrator code
without having to reverse-engineer SimPy's control flow first. For the higher-level
architecture these mechanisms implement, see [[vLLM Worker]] and [[Router]].

## The four primitives Opal builds on

- **`simpy.Environment`** — the virtual clock and event scheduler. Opal creates exactly one,
  in `OpalSimulatorEnvironment.initialize()` (`opal/core/environment.py`), and every other
  component (router, workers, workloads) holds a reference to it as `self.simpy_env`.
- **Processes** — a SimPy process is a Python generator function registered with
  `env.process(fn())`. Whenever the generator hits a `yield`, its process suspends until the
  yielded event fires, then resumes. Opal's router, each worker, and each workload stage all
  run as one or more of these long-lived generator loops.
- **`env.timeout(delay)`** — an event that fires after `delay` simulated seconds. This is how
  Opal advances simulated time: batch execution time, inter-arrival gaps, periodic polling
  intervals, and "sleep forever" idling are all `env.timeout(...)` under the hood.
- **Events** (`env.event()`, and events returned by `Store.get()`/`.put()`, `Resource` requests,
  `Process.interrupt()`, etc.) — the generic thing a process can `yield` on. Two or more events
  can be combined with `|` (any) or `&` (all) to build wait conditions, which Opal uses in a
  few specific places below.

## `env.process()` — starting concurrent processes

Every long-running loop in Opal (router accept loop, per-worker scheduler, per-stage workload
generator) is started once, at setup time, with `env.process(...)`. The router starts four of
them together in `_run()`:

```python
def _run(self):
    """Start gateway processes."""
    self.sim_env.process(self._accept_requests())
    self.sim_env.process(self._collect_completion())
    self.sim_env.process(self._per_second_stats())
    self.sim_env.process(self.process_events())
    ...
```

*(`opal/router/router.py`)*

These four processes run concurrently for the whole simulation: `_accept_requests` dispatches
incoming requests to workers, `_collect_completion` drains finished requests from the shared
results queue, `_per_second_stats` samples throughput/utilization once per simulated second,
and `process_events` batches KV-cache/system events for the prefix-aware routing policy.
None of them call each other directly — they communicate only through `simpy.Store` queues
(see below), which is the idiomatic SimPy way to decouple concurrent processes. See
[[Router]] for what each loop actually does.

## `env.timeout()` — simulated delays

`env.timeout(delay)` is how Opal represents "this operation takes `delay` simulated seconds."
It shows up in three distinct roles:

**1. Modeling real work taking time** — a vLLM scheduler step yields for however long the
GPU model says the batch takes to run:

```python
yield self.simpy_env.timeout(batch_time)
```

**2. Periodic polling on a real interval** — the router's auto-scaler checks queue depth once
a (simulated) second:

```python
def _per_second_scaling(self):
    while not self.opal_env.are_we_done():
        yield self.sim_env.timeout(1)
        max_elements = max(self._outstanding_requests_per_worker.values())
        if max_elements > self.max_queue_threshold and len(self._active_workers) < self.max_workers:
            yield self.sim_env.timeout(self.scale_latency)
            self.safe_add_workers(add_new_workers=1)
```

*(`opal/router/router.py`, `_per_second_scaling`)* — note the second `timeout(self.scale_latency)`
models the time it takes to actually provision a new worker, distinct from the polling interval.

**3. Rate-based workload generation** — `UniformReqRate` (and other workload generators)
compute an inter-arrival delay from a target request rate and `yield` on it between requests:

```python
self.request_rate = self.workload_params["workload_params"]["request_rate"]
self.request_interval = 1.0 / self.request_rate
...
def _intra_request_delay(self):
    return self.request_interval
```

*(`opal/workloads/workload.py`)* — the workload's request-generation loop yields
`env.timeout(self._intra_request_delay())` between submissions, which is what turns a
"requests per second" config value into a concrete arrival process on the simulated
timeline. See [[Running Workloads]] for the generator itself.

`env.timeout(float("inf"))` is a special case Opal uses deliberately as an idle sleep — see
the interrupt section below.

## Interrupt-driven coroutines — the vLLM worker pattern

This is the most distinctive SimPy pattern in Opal, and it exists to avoid polling. A naive
worker implementation would loop `yield env.timeout(0.001); check_for_work()` forever, wasting
event-queue churn even when there is nothing to do. Instead, `LLMWorkerVLLMScheduler`
(`opal/worker/vllm_worker.py`) runs **two cooperating SimPy processes** that sleep indefinitely
when idle and wake each other only via `simpy.Interrupt`:

```python
def _run(self):
    """Start worker processes."""
    # Start scheduling loop first so _scheduling_loop_process is available
    self._scheduling_loop_process = self.simpy_env.process(self._vllm_scheduling_loop())
    # Then start request checker which may interrupt the scheduler
    self._check_new_request_process = self.simpy_env.process(self._check_new_requests())
    self.simpy_env.process(self._periodic_kvc_updates())
```

**The intake loop (`_check_new_requests`)** drains a `simpy.Store` and, when it finds nothing,
parks on an infinite timeout until interrupted:

```python
else:
    # Queue is empty - sleep indefinitely until woken by queue_work().
    self._check_new_requests_idle = True
    try:
        yield self.simpy_env.timeout(float("inf"))
    except simpy.Interrupt:
        # Woken up by queue_work(), continue to check queue
        self.log.debug("_check_new_requests interrupted by new work arrival")
        continue
    finally:
        self._check_new_requests_idle = False
```

**The wake-up call** comes from `queue_work()`, the entry point the router calls to hand a
request to this worker. It puts the request in the store and then interrupts the intake
process directly, but only if intake is actually parked in its idle sleep:

```python
def queue_work(self, request: LLMRequest) -> Generator[None, None, None]:
    yield self._worker_local_queue.put(request)

    # Wake up _check_new_requests only if it is currently in the idle sleep.
    if self._check_new_requests_idle:
        try:
            self._check_new_request_process.interrupt()
        except RuntimeError:
            # Process may have already finished or not yet started
            pass
```

`Process.interrupt()` raises `simpy.Interrupt` inside the target process at whatever `yield`
it is currently suspended on — here, always the `env.timeout(float("inf"))` — which is what
turns an indefinite sleep into an immediate wake-up without any polling interval at all.

**The scheduler loop (`_vllm_scheduling_loop`)** is woken the same way, but only by the intake
loop, and only when it has no work:

```python
if self.waiting_requests or self.running_requests:
    self._scheduler_busy = True
    try:
        yield from self._scheduler_step()
    except simpy.Interrupt:
        # unexpected while busy; new work should only interrupt while idle
        continue
else:
    self._scheduler_busy = False
    try:
        yield self.simpy_env.timeout(float("inf"))
    except simpy.Interrupt:
        continue
    finally:
        self._scheduler_busy = True
```

Two booleans — `_check_new_requests_idle` and `_scheduler_busy` — gate *when* an interrupt is
sent, so a process is only ever interrupted while it is actually parked on the infinite
timeout, never mid-computation. That guard is what makes the pattern safe: interrupting a
process that is mid-`yield` on something other than the idle sleep would abort whatever it was
waiting on instead.

**Why `yield from self._scheduler_step()` instead of `yield self.simpy_env.process(self._scheduler_step())`.**
This choice matters because of a subtlety in how SimPy delivers interrupts. `yield from`
executes the sub-generator *inline*, in the same process, so a `try/except simpy.Interrupt`
around the `yield from` call catches interrupts raised anywhere inside `_scheduler_step()`.
`env.process(...)` instead spawns a *separate* child process — an interrupt sent to the parent
would not reach the child at all, and if the parent reacted by starting a second
`_scheduler_step()` while the first was still running, that would race on shared state like
`waiting_requests`/`running_requests`. Using `yield from` keeps `_scheduler_step()` atomic with
respect to interrupts, which is required here because it mutates request lists across multiple
internal `yield` points. This is documented at length directly in the source
(`opal/worker/vllm_worker.py`, docstring of `_vllm_scheduling_loop`, and `opal/utils/util.py`,
docstring of `safe_process`) if you want the full before/after trace of what goes wrong with
the alternative.

For the request-state machine this interrupt plumbing drives (WAITING → FETCH_KVC → READY →
PREFILL_CHUNKED → DECODE → COMPLETED), see [[vLLM Worker]].

## `simpy.Store` — producer/consumer queues

Wherever two SimPy processes need to hand off data without calling each other directly, Opal
uses a `simpy.Store`: `store.put(item)` and `store.get()` are both SimPy events, so a process
can `yield` on either one and be suspended until the corresponding counterpart is available.

The router creates three stores in its constructor:

```python
self.input_queue = simpy.Store(self.sim_env)  # , capacity=127)
self.results_queue = simpy.Store(self.sim_env)
# leave infinite capacity for this
self._event_queue = simpy.Store(self.sim_env)
```

*(`opal/router/router.py`)* — `input_queue` feeds `_accept_requests`, `results_queue` is where
every worker deposits finished requests for `_collect_completion` to pick up, and
`_event_queue` carries KV-cache/system events into `process_events`. Consumers block on
`yield store.get()`:

```python
def _accept_requests(self):
    """Distribute requests to the worker with shortest incoming request queue."""
    while not self.opal_env.are_we_done():
        request: LLMRequest = yield self.input_queue.get()
        ...
```

The vLLM worker uses the same pattern for its own inbound queue (`_worker_local_queue`,
drained by `_check_new_requests` above), and each workload stage
(`opal/workloads/abstract_workload.py`) has a `_router_response_queue` that the router's
completion collector fills and the workload drains:

```python
self._router_response_queue = simpy.Store(self.simpy_env)
...
def get_completed_requests(self):
    completed = yield self._router_response_queue.get()
    return completed
```

**Bounded wait on a Store — `get_with_timeout`.** Sometimes a consumer must not block forever
on an empty store (e.g. a workload's response-processing loop needs to periodically re-check
whether the stage has finished). Opal implements this by racing a store-get against a timeout
using SimPy's `|` event-condition operator:

```python
def get_with_timeout(env: simpy.Environment, store: simpy.Store, timeout: float):
    with store.get() as get_req:
        timeout_event = env.timeout(timeout)
        # Wait until either the get request triggers or the timeout triggers
        result = yield get_req | timeout_event
        if get_req in result:
            return get_req.value
        else:
            return None
```

*(`opal/utils/util.py`)* — `get_req | timeout_event` yields a condition that fires as soon as
*either* underlying event fires; using `store.get()` as a context manager ensures the pending
get request is cancelled if the timeout wins the race, so it doesn't silently consume an item
that arrives later. `AbstractWorkload._process_responses` uses exactly this to poll its
response queue once a second while also checking the stage's completion flags.

## Event conditions — `AnyOf` and `has_completed`

Beyond the `|` operator above, Opal uses `simpy.AnyOf` explicitly to wait on a *set* of events
whose size isn't known until runtime. The RAG workload caps concurrent in-flight requests and
needs to wake up as soon as *any one* of them finishes, not all of them:

```python
while self.max_concurrent_requests > 0 and len(outstanding) >= self.max_concurrent_requests:
    events = [r.has_completed for r in self._inflight_requests if r.id in outstanding]
    if not events:
        break
    yield simpy.AnyOf(self.simpy_env, events)
    # Drop any completed ids from the outstanding set.
    for r in list(self._inflight_requests):
        if r.has_completed.triggered and r.id in outstanding:
            outstanding.discard(r.id)
```

*(`opal/workloads/rag_workload.py`, `generate_requests`)*

`r.has_completed` here is a plain `env.event()`, used as a one-shot condition variable rather
than a queue item — each `LLMRequest` creates its own in `__init__`:

```python
self.has_completed = self.env.event()
```

and the router's completion path fires it exactly once when the request's turn is done:

```python
def mark_completed(self):
    # ... a producer that replays turns in order does `yield request.has_completed`
    # and blocks, because it must not send turn i+1 until turn i is actually done.
    if not self.has_completed.triggered:
        self.has_completed.succeed(self)
```

*(`opal/core/request.py`)* — the `if not triggered` guard exists because SimPy raises if you
call `.succeed()` on an event twice; wrapping it makes `mark_completed()` idempotent so it can
safely be called from a single call site regardless of ordering. Any process holding a
reference to that specific request can `yield request.has_completed` and resume exactly when
that request completes, without polling — this is how OTel trace replay serializes turns
within a session (submit turn *i*, `yield request.has_completed`, then submit turn *i+1*).

## Sequential stage orchestration

Opal workloads run in ordered stages (see [[Running Workloads]]), and the orchestrator
advances from one stage to the next by yielding on each stage's `_run()` process rather than
by polling a "done" flag:

```python
def run(self):
    for i, s in enumerate(self.stages):
        self.log.info(f"Stating the execution of stage {i} with type: {str(s)}")
        self.active_stage = i
        self.stage_stats[i].stage_time_start = self.opal_env.simpy_env.now
        # execute each stage in turn and wait until it is done
        yield safe_process(self.opal_env.simpy_env, s._run())
        self.stage_stats[i].stage_time_end = self.opal_env.simpy_env.now
        self.log.info(f"Stage {i} with type: {str(s)} finished")
    self.log.info(f"Workload orchestration finished")
    self.workload_orchestration_done = True
```

*(`opal/workloads/workload_orchestrator.py`)* — `safe_process(env, coro)` (see
`opal/utils/util.py`) just wraps `env.process(coro)` with exception logging and lets
`simpy.Interrupt` propagate as normal control flow; `yield`ing the resulting `Process` object
blocks the orchestrator loop until that stage's `_run()` generator returns, at which point the
`for` loop moves to the next stage. This is the same "yield a process to wait for it" mechanic
used throughout Opal, here used at the coarsest granularity (whole workload stages) rather than
individual requests or batches.

Within a single stage, `AbstractWorkload._run()` composes two of its own child processes with
the `&` (all-of) event-condition operator so the stage isn't considered finished until *both*
request generation and response collection have completed:

```python
process_response = safe_process(self.simpy_env, self._process_responses())
process_generation = safe_process(self.simpy_env, self.generate_requests())
yield process_generation & process_response
```

*(`opal/workloads/abstract_workload.py`, `_run`)*

## See also

- [[vLLM Worker]] — the request state machine and batching algorithm the interrupt-driven
  loops above actually drive.
- [[Router]] — what each of the router's concurrent processes does and how routing policies
  use worker state.
- [[Running Workloads]] — how stages and request-rate generators are configured.
- [Official SimPy documentation](https://simpy.readthedocs.io/) — for the general-purpose
  semantics of `Environment`, `Process`, `Event`, `Store`, `Resource`, and `Interrupt` this
  page assumes.
