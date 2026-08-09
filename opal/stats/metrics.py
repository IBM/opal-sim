# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MetricsSnapshot:
    """A point-in-time view of cluster telemetry as seen by the router.

    Tthe values are built from the last telemetry each worker pushed to the router
    (see SystemEvent) plus latency percentiles.

    Units:
    - queue_depth_per_worker: absolute in-flight request count per worker.
    - kvc_util_per_worker / gpu_util_per_worker: fractions in [0, 1].
    - p95_ttft_secs / p95_itl_secs: seconds; -1 when there are not yet enough
      completed requests to compute a meaningful percentile.
    """

    timestamp: float = 0.0
    queue_depth_per_worker: dict[int, int] = field(default_factory=dict)
    kvc_util_per_worker: dict[int, float] = field(default_factory=dict)
    gpu_util_per_worker: dict[int, float] = field(default_factory=dict)
    p95_ttft_secs: float = -1.0
    p95_itl_secs: float = -1.0

    @property
    def max_queue_depth(self) -> int:
        return max(self.queue_depth_per_worker.values(), default=0)

    @property
    def max_kvc_util(self) -> float:
        return max(self.kvc_util_per_worker.values(), default=0.0)

    @property
    def max_gpu_util(self) -> float:
        return max(self.gpu_util_per_worker.values(), default=0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "queue_depth_per_worker": dict(self.queue_depth_per_worker),
            "kvc_util_per_worker": dict(self.kvc_util_per_worker),
            "gpu_util_per_worker": dict(self.gpu_util_per_worker),
            "p95_ttft_secs": self.p95_ttft_secs,
            "p95_itl_secs": self.p95_itl_secs,
            "max_queue_depth": self.max_queue_depth,
            "max_kvc_util": self.max_kvc_util,
            "max_gpu_util": self.max_gpu_util,
        }
