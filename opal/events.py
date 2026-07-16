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
    tier_name: str = ""      # destination tier name (INSERT, MOVE, COPY)
    src_tier_name: str = ""  # source tier name (DELETE, MOVE)


@dataclass
class SystemEvent(OpalInfraEvent):
    # This will be updated periodically like every 5 seconds
    # all these values are normalized between [0, 1]
    # 0 = min, 1 = max
    load: float
    ingress_queue_occupancy: float
    mem_used: float
    gpu_utilization: float
