# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum


class KVCEventType(IntEnum):
    INSERT = 1
    DELETE = 2
    MOVE = 3
    COPY = 4


@dataclass
class OpalInfraEvent(ABC):
    worker_id: int


@dataclass
class KVCEvent(OpalInfraEvent):
    # This will happen whenever there is a new event
    # It can also be aggregated to decrease the load on the system
    chunk_hash: int
    src_tier: int
    dst_tier: int
    eventType: KVCEventType


@dataclass
class SystemEvent(OpalInfraEvent):
    # This will be updated periodically like every 5 seconds
    # all these values are normalized between [0, 1]
    # 0 = min, 1 = max
    load: float
    ingress_queue_occupancy: float
    gpu_utilization: float
    # kvc_utilization: fraction of GPU KV-cache blocks in use (1 - free/total), in [0, 1].
    kvc_utilization: float = 0.0
    # queue_depth: absolute in-flight request count on the worker (waiting + running).
    queue_depth: int = 0
