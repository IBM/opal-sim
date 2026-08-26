# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MetricsSnapshot:
    """A point-in-time view of cluster telemetry as seen by the router.

    Built when workers push SystemEvent telemetry.
    Worker fields are the last report from each worker (delayed by
    periodic_infra_update_time). TTFT/ITL are the configured percentile over a
    sliding window of recent completed requests.

    Units:
    - queue_depth_per_worker: absolute in-flight request count per worker.
    - kvc_util_per_worker: fraction in [0, 1].
    - ttft_secs / itl_secs: seconds; -1 until `window` requests have completed.
    """

    timestamp: float = 0.0
    queue_depth_per_worker: dict[int, int] = field(default_factory=dict)
    kvc_util_per_worker: dict[int, float] = field(default_factory=dict)
    percentile: int = 95  # 90, 95, or 99
    window: int = 50  # last N completed requests used for ttft/itl
    ttft_secs: float = -1.0
    itl_secs: float = -1.0

    @property
    def max_queue_depth(self) -> int:
        return max(self.queue_depth_per_worker.values(), default=0)

    @property
    def max_kvc_util(self) -> float:
        return max(self.kvc_util_per_worker.values(), default=0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "queue_depth_per_worker": dict(self.queue_depth_per_worker),
            "kvc_util_per_worker": dict(self.kvc_util_per_worker),
            "percentile": self.percentile,
            "window": self.window,
            "ttft_secs": self.ttft_secs,
            "itl_secs": self.itl_secs,
            "max_queue_depth": self.max_queue_depth,
            "max_kvc_util": self.max_kvc_util,
        }
