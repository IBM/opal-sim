# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import logging
import re
import traceback
from typing import Dict, Generator, List, Optional, Sequence, Tuple
import numpy as np

from simpy import Environment, Event
import simpy
from opal.events import KVCEvent, KVCEventType
from opal.io_model import AbstractDevice, CPUMemory, OpalIORequest, LocalNVMe, DistributedFS
from opal.llm_model import OpalModelConfig
from collections import OrderedDict

import abc
import hashlib
import logging
import pickle
from collections.abc import Callable
from typing import Any, Iterable, List, Optional, Tuple, Union
from concurrent.futures import Future


@dataclass
class OpalMemoryObj:
    size: int
    tokens: int
    location_tier: str

    def get_size(self):
        return self.size


# Code taken from `config.py`
@dataclass
class OpalEngineMetadata:
    """name of the LLM model"""

    model_name: str
    """ world size when running under a distributed setting """
    world_size: int
    """ worker id when running under a distributed setting """
    worker_id: int
    """ the format of kv tensors """
    fmt: str
    """ the data type of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    kv_dtype: str
    """ the shape of kv tensors """
    # (Deprecated) Will be replaced by kv_layer_groups_manager in the future
    """ (num_layer, 2, chunk_size, num_kv_head, head_size) """
    kv_shape: tuple[int, int, int, int, int]
    """ whether use MLA"""
    use_mla: bool = False
    """ the role of the current instance (e.g., 'scheduler', 'worker') """
    role: Optional[str] = None
    """ the first rank of the distributed setting """
    first_rank = 0
    served_model_name: Optional[str] = None


# Optimized hash function for tuples/lists of integers
# Original code from https://github.com/vllm-project/vllm/blob/main/vllm/utils/hashing.py
# Modified to avoid pickle overhead since we only hash tuples/lists of integers

# Cache for hash results to avoid recomputation
_hash_cache: Dict[int, bytes] = {}
_cache_hits = 0
_cache_misses = 0


def sha256_fast(input: Any) -> bytes:
    """Fast hash for tuples/lists of integers using SHA-256 with caching.

    Optimized version that:
    1. Caches hash results to avoid recomputation (major speedup)
    2. Uses direct byte conversion instead of pickle for integers

    Args:
        input: Tuple or list of integers, or nested tuple of (hash, tuple).

    Returns:
        Bytes representing the SHA-256 hash.
    """
    global _cache_hits, _cache_misses

    # Try cache first using Python's built-in hash for the key
    # This works because tuples/lists of ints are hashable (after conversion)
    try:
        if isinstance(input, tuple):
            cache_key = hash(input)
        elif isinstance(input, list):
            cache_key = hash(tuple(input))
        else:
            cache_key = None

        if cache_key is not None and cache_key in _hash_cache:
            _cache_hits += 1
            return _hash_cache[cache_key]

        _cache_misses += 1
    except TypeError:
        # Input not hashable (e.g., nested mutable structures)
        cache_key = None

    # Compute hash
    # Handle nested tuple case: (prefix_hash, tokens_tuple)
    if isinstance(input, tuple) and len(input) == 2:
        first, second = input
        if isinstance(first, bytes) and isinstance(second, (list, tuple)):
            # This is (prefix_hash, tokens_tuple) - concatenate the hash and tokens
            try:
                tokens_bytes = np.array(second, dtype=np.int32).tobytes()
                result = hashlib.sha256(first + tokens_bytes).digest()
            except (ValueError, TypeError):
                # Fallback if conversion fails
                input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
                result = hashlib.sha256(input_bytes).digest()
        else:
            # Not the expected pattern, use pickle
            input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
            result = hashlib.sha256(input_bytes).digest()
    # Handle simple tuple/list of integers
    elif isinstance(input, (list, tuple)):
        try:
            # Convert to numpy array for efficient byte conversion
            input_bytes = np.array(input, dtype=np.int32).tobytes()
            result = hashlib.sha256(input_bytes).digest()
        except (ValueError, TypeError):
            # Fallback if conversion fails (e.g., nested structures)
            input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
            result = hashlib.sha256(input_bytes).digest()
    else:
        # Fallback to pickle for other types
        input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
        result = hashlib.sha256(input_bytes).digest()

    # Cache the result
    if cache_key is not None:
        _hash_cache[cache_key] = result

    return result


def get_hash_cache_stats() -> Dict[str, int]:
    """Get hash cache statistics for monitoring."""
    return {
        "hits": _cache_hits,
        "misses": _cache_misses,
        "size": len(_hash_cache),
        "hit_rate": _cache_hits / max(1, _cache_hits + _cache_misses),
    }


def clear_hash_cache():
    """Clear the hash cache. Useful for testing or memory management."""
    global _hash_cache, _cache_hits, _cache_misses
    _hash_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def sha256(input: Any) -> bytes:
    """Hash any picklable Python object using SHA-256.

    Legacy function kept for compatibility. New code should use sha256_fast.

    Args:
        input: Any picklable Python object.

    Returns:
        Bytes representing the SHA-256 hash of the serialized input.
    """
    input_bytes = pickle.dumps(input, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(input_bytes).digest()


def get_hash_fn_by_name(hash_fn_name: str) -> Callable[[Any], bytes]:
    """Get a hash function by name, or raise an error if the function is not found.

    Args:
        hash_fn_name: Name of the hash function.

    Returns:
        A hash function.
    """
    if hash_fn_name == "sha256":
        return sha256_fast  # Use optimized version by default
    elif hash_fn_name == "sha256_legacy":
        return sha256  # Legacy pickle-based version

    raise ValueError(f"Unsupported hash function: {hash_fn_name}")


@dataclass(slots=True)
class OpalCacheEngineKey:
    """
    atr - copied as it is
    """

    fmt: str
    model_name: str
    world_size: int
    worker_id: int
    chunk_hash: int
    dtype_str: str

    def __post_init__(self):
        pass

    def __hash__(self):
        return hash(
            (
                self.fmt,
                self.model_name,
                self.world_size,
                self.worker_id,
                self.chunk_hash,
                self.dtype_str,
            )
        )

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.fmt == other.fmt
                and self.model_name == other.model_name
                and self.world_size == other.world_size
                and self.worker_id == other.worker_id
                and self.chunk_hash == other.chunk_hash
                and self.dtype_str == other.dtype_str
            )

        return False

    def to_string(self):
        s = f"{self.fmt}@{self.model_name}@{self.world_size}" f"@{self.worker_id}@{self.chunk_hash:x}@{self.dtype_str}"
        return s

    @staticmethod
    def from_string(s):
        parts = s.split("@")
        if len(parts) < 6:
            raise ValueError(f"Invalid key string: {s}")
        return OpalCacheEngineKey(parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4], 16), str(parts[5]))

    def to_dict(self):
        msg = {
            "__type__": "CacheEngineKey",
            "fmt": self.fmt,
            "model_name": self.model_name,
            "world_size": self.world_size,
            "worker_id": self.worker_id,
            "chunk_hash": self.chunk_hash,
            "dtype": self.dtype_str,
        }
        return msg

    @staticmethod
    def from_dict(d):
        request_configs = None
        if request_configs_list := d.get("request_configs"):
            request_configs = {}
            for kv in request_configs_list:
                kvs = kv.split("%", 1)
                if len(kvs) != 2:
                    raise ValueError(f"Invalid key dict: {d}")
                request_configs[kvs[0]] = kvs[1]
        return OpalCacheEngineKey(
            fmt=d["fmt"],
            model_name=d["model_name"],
            world_size=d["world_size"],
            worker_id=d["worker_id"],
            chunk_hash=d["chunk_hash"],
            dtype_str=d["dtype"],
        )

    def with_new_worker_id(self, new_worker_id: int) -> "OpalCacheEngineKey":
        # Reconstruct the cache engine key with new worker id
        return OpalCacheEngineKey(
            self.fmt, self.model_name, self.world_size, new_worker_id, self.chunk_hash, self.dtype_str
        )


# Type alias for process_tokens return value
# (start_index, end_index, cache_engine_key｜hash)
ProcessTokensResult = Tuple[int, int, Union[OpalCacheEngineKey, int]]

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[OpalCacheEngineKey, OpalMemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]

# see the `token_database.py` file and init() of abstract TokenDatabase class to see
# some complications on how how NONE_HASH is initialized.
NONE_HASH: int = 0


class OpalTokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert
    each chunk into a cache engine key using prefix hash.

    - SegmentTokenDatabase: It processes tokens into segments based on
    special separators and convert each segment into a cache engine key.

    atr: adapted from the class, and using just ChunkedTokenDatabase logic.
    """

    def __init__(self, chunk_size: int, metadata: OpalEngineMetadata):
        global NONE_HASH
        self.chunk_size = chunk_size
        self.metadata = metadata
        self.hash_algorithm = "sha256"
        self.hash_func = get_hash_fn_by_name(self.hash_algorithm)
        self.log = logging.getLogger(str(self))
        # self.log.debug(f"Using hash algorithm: {self.hash_algorithm}")
        # hidden set variables
        self.save_unfull_chunk = True
        self.save_only_first_rank = False
        self._count_process_tokens = 0

    def __str__(self):
        return f"OpalTokenDatabase{self.chunk_size}"

    def _chunk_tokens(
        self,
        tokens: List[int],
    ) -> Iterable[List[int]]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with
                shape [chunk_size]
        """
        # Simple chunking without caching - tuple conversion was too expensive
        end = len(tokens) if self.save_unfull_chunk else (len(tokens) - len(tokens) % self.chunk_size)
        for i in range(0, end, self.chunk_size):
            yield tokens[i : i + self.chunk_size]

    def _get_init_hash(self) -> int:
        return NONE_HASH

    def _prefix_hash(
        self,
        token_chunks: Iterable[List[int]],
    ) -> Iterable[int]:
        """
        Compute prefix hashes for token chunks.
        """
        # Simple prefix hashing without caching - tuple conversion was too expensive
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: Optional[List[int]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        make_key: bool = False,
        start_idx: int = 0,
        prefix_hash: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Iterable[ProcessTokensResult]:
        """Process the tokens and return the corresponding cache engine keys.

        :param List[int] tokens: The tokens to process.

        :param Optional[List[int]] hashes: The hashes to process. If provided,
            it will be used instead of tokens to generate cache engine keys.

        :param Optional[List[int]] offsets: The number of tokens in each chunk.

        :param bool make_key: Whether to make the cache engine key or not.
            If False, the hash value will be returned instead.

        :param Optional[dict] request_configs: The configs of the request.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key (or hash) for the tokens.
        """
        self._count_process_tokens += 1
        if tokens is not None:
            if start_idx == 0 and prefix_hash is None and end_idx is None:
                # Full pass from token 0 (original behaviour, unchanged).
                total_len = len(tokens)
                token_chunks = self._chunk_tokens(tokens)
                prefix_hashes = self._prefix_hash(token_chunks)
                for chunk_id, hash_val in enumerate(prefix_hashes):
                    s = chunk_id * self.chunk_size
                    e = min(s + self.chunk_size, total_len)
                    yield (s, e, self._make_key_by_hash(hash_val)) if make_key else (s, e, hash_val)
            else:
                # Resume the chain from start_idx (was process_tokens_from).
                assert start_idx % self.chunk_size == 0, f"start_idx {start_idx} not chunk-aligned"
                if end_idx is None:
                    end_idx = len(tokens)
                if prefix_hash is None:
                    prefix_hash = self._get_init_hash()
                chunk_id = start_idx // self.chunk_size
                for token_chunk in self._chunk_tokens(tokens[start_idx:end_idx]):
                    prefix_hash = self._hash_tokens(token_chunk, prefix_hash)
                    s = chunk_id * self.chunk_size
                    e = min(s + self.chunk_size, end_idx)
                    yield (s, e, self._make_key_by_hash(prefix_hash)) if make_key else (s, e, prefix_hash)
                    chunk_id += 1
        elif hashes is not None:
            assert offsets is not None, "If hashes are provided, offsets must also be provided."
            start = 0
            for hash_val, offset in zip(hashes, offsets, strict=False):
                end = start + offset
                yield (start, end, self._make_key_by_hash(hash_val)) if make_key else (start, end, hash_val)
                start = end
        else:
            raise ValueError("Either tokens or hashes must be provided.")

        # name = inspect.currentframe().f_back.f_code.co_name
        # #traceback.print_stack()
        # print(name, get_hash_cache_stats(), self._count_process_tokens)

    def _make_key_by_hash(self, chunk_hash: int):
        return OpalCacheEngineKey(
            self.metadata.fmt,
            self.metadata.model_name,
            self.metadata.world_size if not self.save_only_first_rank else 1,
            self.metadata.worker_id,
            chunk_hash,
            self.metadata.kv_dtype,
        )

    def _hash_tokens(self, tokens: List[int], prefix_hash: Optional[int] = None) -> int:
        tokens_tuple = tuple(tokens)
        # Ignore extra keys for now
        # Extra keys are for multi-modal inputs and
        # request specific metadata (e.g., LoRA ID).
        return self.hash_func((prefix_hash, tokens_tuple))


class OpalStorageBackend:
    """
    class logic copied from RemoteBackend which implements StorageBackendInterface
    Here we maintain per-backend specific logic
    """

    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        opal_config: "OpalConfig",
        device: AbstractDevice,
        worker_id: int,
        tier_index: int,
        tier_name: str,
    ):
        self.opal_env = opal_env
        self.opal_config = opal_config
        self.worker_id = worker_id
        # index and name: is the index and name in the priority configuration as described in the configuration
        self.tier_name = tier_name
        self.tier_index = tier_index
        self.log = logging.getLogger(str(self))
        # this is the hash table index that is being maintained.
        self._index: "OrderedDict[OpalCacheEngineKey, OpalMemoryObj]" = OrderedDict()
        # Running total of obj.tokens over self._index, kept in sync by
        self._tokens_stored: int = 0
        # Keys that must not be evicted: items being fetched (pinned by get_blocking)
        # or being migrated to next_backend (pinned by evict_async/_migrate_to_next).
        self._pinned: set["OpalCacheEngineKey"] = set()
        # Pointer to the next (lower/slower) storage tier; None for the last tier.
        # Wired up by OpalStorageManager.__init__ after all backends are created.
        self.next_backend: Optional["OpalStorageBackend"] = None
        # Fraction of capacity at which the background monitor triggers async eviction.
        self.eviction_threshold: float = (
            opal_config["kvc"].get(tier_name, {}).get("eviction_threshold", 1.0)
        )
        # this is the performance model device
        self.backend_device = device

    def _index_insert(self, key: "OpalCacheEngineKey", memory_obj: "OpalMemoryObj") -> None:
        """Insert/overwrite an entry in self._index, keeping _tokens_stored in sync."""
        old = self._index.get(key)
        if old is not None:
            self._tokens_stored -= old.tokens
            # The caller unconditionally reserved capacity for `memory_obj`
            # (capacity_remaining_bytes -= memory_obj.size) before this insert.
            # On an overwrite the old object's bytes would otherwise be orphaned
            # (charged twice, freed once), leaking capacity. Restore them here so
            # the tier's capacity_remaining_bytes stays consistent with the index.
            self.backend_device.capacity_remaining_bytes += old.get_size()
        self._index[key] = memory_obj
        self._tokens_stored += memory_obj.tokens

    def _index_pop(
        self, key: "OpalCacheEngineKey", default: Optional["OpalMemoryObj"] = None
    ) -> Optional["OpalMemoryObj"]:
        """Pop an entry from self._index, keeping _tokens_stored in sync."""
        obj = self._index.pop(key, default)
        if obj is not None:
            self._tokens_stored -= obj.tokens
        return obj

    def __str__(self):
        return f"{__class__.__name__}.{self.worker_id}.{self.tier_name}"

    def contains(self, key: OpalCacheEngineKey, pin: bool = False) -> Generator[simpy.Event, None, bool]:
        # each lookup has a cost, just modeling it with 8 byte I/O
        r = OpalIORequest(8)
        self.backend_device.process_one_request(r)
        yield r.event
        return key in self._index

    def batched_contains(
        self,
        keys: List[OpalCacheEngineKey],
        pin: bool = False,
    ) -> Generator[simpy.Event, Any, int]:
        """
        Check whether the keys are in the storage backend.

        :param List[CacheEngineKey] keys: The keys of the MemoryObj.

        :param bool pin: Whether to pin the key.
            If True, the corresponding KV cache will be
            pinned in the storage backend.

        :return: Return hit chunks by prefix match.

        atr: code taken from StorageBackendInterface (abstract_backend.py)
        this is used when the backend does not support the batched_contain native call
        """
        hit_chunks = 0
        for key in keys:
            x = yield from self.contains(key, pin)
            if not x:
                break
            hit_chunks += 1
        return hit_chunks

    def support_batched_contains(self) -> bool:
        """
        Is supported batched_contains
        """
        return False

    def exists_in_put_tasks(self, key: OpalCacheEngineKey) -> bool:
        raise NotImplementedError

    def put_callback(self, future: Future, key: OpalCacheEngineKey):
        """
        Callback function for put tasks.
        """
        raise NotImplementedError

    def submit_put_task(
        self,
        key: OpalCacheEngineKey,
        memory_obj: OpalMemoryObj,
    ) -> Generator[simpy.Event, None, KVCEvent]:
        # reflect the capacity decrease up-front
        self.backend_device.capacity_remaining_bytes -= memory_obj.get_size()
        # issue the request
        r = OpalIORequest(memory_obj.get_size())
        self.backend_device.process_one_request(r)
        # wait for completion
        yield r.event
        # insert the object in the index. Use _index_insert (not a raw assignment)
        # so _tokens_stored stays in sync and an overwrite refunds the old object's
        # bytes -- concurrent stores of the same key otherwise leak capacity and
        # leave _tokens_stored stuck at 0.
        self._index_insert(key, memory_obj)
        # allocate KVCache event here and return it
        e = KVCEvent(self.worker_id, hash(key), -1, self.tier_index, KVCEventType.INSERT)
        return e

    def batched_put_callback(self, future: Future, keys: List[OpalCacheEngineKey]):
        """
        Callback function for batched put tasks.
        """
        raise NotImplementedError

    def batched_submit_put_task(
        self,
        keys: Sequence[OpalCacheEngineKey],
        memory_objs: List[OpalMemoryObj],
        transfer_spec: Any = None,
    ) -> None:
        raise NotImplementedError

    def get_blocking(
        self,
        key: OpalCacheEngineKey,
    ) -> Generator[simpy.Event, None, Optional[OpalMemoryObj]]:
        """
        Blocking get function.
        """
        if (yield from self.contains(key)):
            memory_obj = self._index[key]
            r = OpalIORequest(memory_obj.size)
            self.backend_device.process_one_request(r)
            yield r.event
            return memory_obj
        else:
            return None

    def batched_get_blocking(
        self,
        keys: List[OpalCacheEngineKey],
    ) -> List[Optional[OpalMemoryObj]]:
        raise NotImplementedError

    async def support_batched_async_contains(self) -> bool:
        return False

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[OpalCacheEngineKey],
        pin: bool = False,
    ) -> int:
        raise NotImplementedError

    async def support_batched_get_non_blocking(self) -> bool:
        return False

    async def batched_get_non_blocking(
        self,
        lookup_id: str,
        keys: List[OpalCacheEngineKey],
        transfer_spec: Any = None,
    ) -> List[OpalMemoryObj]:
        raise NotImplementedError

    def pin(self, key: OpalCacheEngineKey) -> bool:
        raise NotImplementedError

    def unpin(self, key: OpalCacheEngineKey) -> bool:
        raise NotImplementedError

    def remove(self, key, force=True):
        raise NotImplementedError

    def get_allocator_backend(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
    
    def evict_sync(self, bytes_needed: int) -> int:
        """Synchronous drop: immediately remove unpinned LRU items to free
        ``bytes_needed`` bytes on this tier.  No I/O, no migration — dropped
        items are permanently lost.  Called from evict_and_cascade() and
        _migrate_to_next() to make room at the destination tier before writing.

        Returns the number of bytes actually freed."""
        need = bytes_needed - self.backend_device.capacity_remaining_bytes
        if need <= 0:
            return 0
        # Collect candidates in LRU order (front of OrderedDict), skipping pinned.
        to_drop: List[OpalCacheEngineKey] = []
        for key in self._index:
            if need <= 0:
                break
            if key not in self._pinned:
                to_drop.append(key)
                need -= self._index[key].size
        freed = 0
        for key in to_drop:
            obj = self._index_pop(key, None)
            if obj is not None:
                self.backend_device.capacity_remaining_bytes += obj.size
                freed += obj.size
        if to_drop:
            self.log.debug(
                f"[SYNC-DROP] {self.tier_name}: dropped {len(to_drop)} chunk(s) "
                f"({freed / 2**20:.2f} MB); "
                f"free now {self.backend_device.capacity_remaining_bytes / 2**20:.2f} MB"
            )
        return freed

    def evict_and_cascade(
        self, bytes_needed: int
    ) -> Generator[simpy.Event, None, int]:
        """Like evict_sync() but writes evicted items to next_backend before
        removing them from this tier.  Items on the last tier (no next_backend)
        are dropped as usual.

        This is a generator because writing to the next tier requires I/O yields.
        Called from batched_put() so that LRU items displaced by new writes are
        preserved in the tier below rather than silently lost."""
        need = bytes_needed - self.backend_device.capacity_remaining_bytes
        if need <= 0:
            return 0
        to_evict: List[OpalCacheEngineKey] = []
        for key in self._index:
            if need <= 0:
                break
            if key not in self._pinned:
                to_evict.append(key)
                need -= self._index[key].size
        freed = 0
        dropped = 0
        cascaded = 0
        for key in to_evict:
            obj = self._index_pop(key, None)
            if obj is None:
                continue
            self.backend_device.capacity_remaining_bytes += obj.size
            freed += obj.size
            nb = self.next_backend
            if nb is not None:
                if obj.size > nb.backend_device.capacity_remaining_bytes:
                    nb.evict_sync(obj.size)
                if obj.size <= nb.backend_device.capacity_remaining_bytes:
                    nb.backend_device.capacity_remaining_bytes -= obj.size
                    yield from nb.submit_put_task(key, obj)
                    cascaded += 1
                else:
                    dropped += 1
            else:
                dropped += 1
        if dropped > 0:
            self.log.debug(
                f"[KVC Mgr] [{self.tier_name}] [DROP] : "
                f"dropped {dropped} chunk(s) ({freed / 2**20:.2f} MB). "
                f"Free space {self.backend_device.capacity_remaining_bytes / 2**20:.2f} MB."
            )
        if cascaded > 0:
            self.log.debug(
                f"[KVC Mgr] [{self.tier_name}] [EVICT] : "
                f"evict cascaded {cascaded} chunk(s)"
            )
        return freed


# Generator[YieldType, SendType, ReturnType]
def allocate_and_copy_objects(
    simpy_env, keys: Sequence[OpalCacheEngineKey], src_memory_objs: list[OpalMemoryObj]
) -> Generator[Event, None, tuple[Sequence[OpalCacheEngineKey], list[OpalMemoryObj]]]:
    print(f"---- ALLOCATOR---- ")
    # FIXME:
    # calculate time for memory allocation
    time = 0.1
    # calculate time for I/O transfer
    time = 0.2
    yield simpy_env.timeout(time)
    print(f"returning appropriate values...")
    return keys, src_memory_objs


class OpalStorageManager:
    """
    This class should encapsulate the logic of storage tiering, placement, and replication.

    """

    def __init__(self, opal_env: "OpalSimulatorEnvironment", opal_config: "OpalConfig", worker_id: int):
        self.opal_env = opal_env
        self.opal_config = opal_config
        self.worker_id = worker_id
        # logger must be after this
        self.log = logging.getLogger(str(self))
        # there can be multiple such stores for GPU, CPU, local SSDs, or remote file system
        # this is where tiering information comes into play
        backends = self.opal_config["kvc"]["kvc_tiers"]
        # AbstractDevice gives performance estimation, while OpalStorageBackend has capacity, indexes etc.
        self.storage_backends: OrderedDict[str, OpalStorageBackend] = {}
        for i, b in enumerate(backends):
            # this instantiates the class based on the name
            store = globals()[b](self.opal_env)
            backend = OpalStorageBackend(opal_env, opal_config, store, self.worker_id, i, b)
            self.storage_backends[b] = backend
            self.log.debug(f'initialized "{b}" backend')

        # Wire singly-linked next_backend pointers so evict_and_cascade
        # can push items down the tier hierarchy without knowing about the manager.
        backend_list = list(self.storage_backends.values())
        for i in range(len(backend_list) - 1):
            backend_list[i].next_backend = backend_list[i + 1]


    def cannot_store(self) -> bool:
        return len(self.storage_backends) == 0

    def __str__(self):
        return f"{__class__.__name__}.{self.worker_id}"

    def batched_put(
        self,
        keys: Sequence[OpalCacheEngineKey],
        memory_objs: List[OpalMemoryObj],
        transfer_spec=None,
        location: Optional[str] = None,
    ) -> Generator[simpy.Event, None, Dict[str, Tuple[int, int]]]:
        """
        This is put operation, hence coming from the GPU -> *{CPU, SSD, Remote SSD, Node}
        """
        assert len(keys) == len(memory_objs), str(f"{len(keys)} != {len(memory_objs)} ")

        self.log.debug(f"Called batched_put, gradually checking and filling out tiers")
        chunk_per_tier = {}
        kvc_events = []
        for backend_name, backend in self.storage_backends.items():
            if location and backend_name != location:
                continue
            stored_so_far = 0
            new_stored = 0
            dev: AbstractDevice = backend.backend_device
            self.log.debug(
                f'storing {len(keys)} keys on tier: "{backend_name}" with available capacity: {dev.capacity_remaining_bytes/(1024*1024)} Mbytes'
            )
            # now we wait and sleep
            # And collect all the events
            kvc_events = []
            for k, m in zip(keys, memory_objs):
                if m.size <= dev.capacity_remaining_bytes:
                    if not (yield from backend.contains(k)):
                        # if key is not already stored
                        e: KVCEvent = yield from backend.submit_put_task(k, m)
                        kvc_events.append(e)
                        new_stored += 1
                    # we count it that we indeed stored.
                    stored_so_far += 1
                else:
                    break
            # we will be moving to the new tier, so adjust the arrays that needs storing
            keys = keys[stored_so_far:]
            memory_objs = memory_objs[stored_so_far:]
            chunk_per_tier[backend_name] = stored_so_far, new_stored
        # [KVC Store] write-side visibility: shows keys actually landing in each
        # tier (new vs already-present) and the tier's running index size, so it
        # is easy to confirm whether anything is ever written to CPU/DFS.
        for tier_name, (total, new) in chunk_per_tier.items():
            if total == 0:
                continue
            be = self.storage_backends[tier_name]
            self.log.info(
                f"[KVC Store] worker.{self.worker_id} | tier {tier_name} | "
                f"+{new} new keys (matched {total - new} already-present) | "
                f"tier now holds {len(be._index)} keys / {be._tokens_stored} tokens"
            )
        # FIXME: generate these KVCache events
        worker: "LLMWorkerSingleStage" = self.opal_env.registry.get_worker(self.worker_id)
        worker.append_kvc_events(kvc_events)
        self.log.debug(f"Generated {len(kvc_events)} KVCEvents for the worker")
        return chunk_per_tier

    def get(
        self,
        key: OpalCacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[OpalMemoryObj]:
        raise NotImplementedError

    def get_non_blocking(
        self,
        key: OpalCacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[asyncio.Future]:
        raise NotImplementedError

    def batched_get(
        self,
        keys: List[OpalCacheEngineKey],
        location: Optional[str] = None,
    ) -> Generator[simpy.Event, None, Optional[List[Optional[OpalMemoryObj]]]]:
        # for each backend, get the associate mem object
        # memory_obj = OpalMemoryObj(size, tokens_to_process, "gpu")
        backend = self.storage_backends[location]
        ret_val = []
        for k in keys:
            contains = yield from backend.contains(k)
            assert contains
            fetched_obj = yield from backend.get_blocking(k)
            ret_val.append(fetched_obj)
        return ret_val

    def prefetch_all_done_callback(
        self,
        task: asyncio.Future,
        lookup_id: str,
        cum_chunk_lengths_total: list[int],
        tier_expected_chunks: list[int],
    ) -> None:
        # atr - this is where the big diagram comment explanation is
        raise NotImplementedError

    def contains(
        self,
        key: OpalCacheEngineKey,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
    ) -> Optional[str]:
        raise NotImplementedError

    def batched_contains(
        self,
        keys: List[OpalCacheEngineKey],
        pin: bool = False,
    ) -> Generator[simpy.Event, None, tuple[int, dict]]:
        """
        Check whether the key exists in the storage backend.

        :param List[CacheEngineKey] keys: The keys to check.

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of ["LocalCPUBackend",
        "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param bool pin: Whether to pin the key.

        return: Return hit chunks and block mapping by prefix match.
        """
        total_keys = len(keys)
        total_hit_chunks = 0
        block_mapping = {}
        for backend_name, backend in self.storage_backends.items():
            # NOTE(Jiayi): We do not pin for PDBackend
            pin_in_backend = pin if backend_name != "PDBackend" else False

            hit_chunks = yield from backend.batched_contains(keys, pin_in_backend)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = keys[:hit_chunks]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]

        return total_hit_chunks, block_mapping

    def remove(
        self,
        key: OpalCacheEngineKey,
        locations: Optional[List[str]] = None,
    ) -> int:
        raise NotImplementedError

    def get_block_mapping(
        self, chunk_infos: List[Tuple[OpalCacheEngineKey, int, int]]
    ) -> Generator[simpy.Event, None, Dict[str, List[Tuple[OpalCacheEngineKey, int, int]]]]:
        """
        Get block mapping for the given chunk infos, works by prefix match.

        :param List[Tuple[CacheEngineKey, int, int]] chunk_infos:
        List of chunk infos, each tuple contains (key, begin, end)

        :return: Dict[str, List[Tuple[CacheEngineKey, int, int]]]:
        Block mapping for the given chunk infos, each key is the backend name,
        each value is a list of chunk infos in the backend.
        """
        keys = [chunk_info[0] for chunk_info in chunk_infos]
        total_keys = len(keys)
        block_mapping = {}
        total_hit_chunks = 0
        for backend_name, backend in self.storage_backends.items():
            hit_chunks = yield from backend.batched_contains(keys)
            if hit_chunks == 0:
                continue
            block_mapping[backend_name] = chunk_infos[total_hit_chunks : total_hit_chunks + hit_chunks]
            total_hit_chunks += hit_chunks
            if total_hit_chunks == total_keys:
                break
            keys = keys[hit_chunks:]
        return block_mapping


class OpalKVCacheEngine:
    """This class represent a worker-local kv cache manager. The logic and API of this class resembles
    of the LMCache/LMCacheEngine.
    There are three key functions here:
     1. store()  -- worker's side write
     2. retrieve() -- worker's side read (load on the GPU)
     3. lookup() -- called from the scheduler.
     all others can be implemented as the need arises.

    What I have to think how cleanly to integrate this logic with vLLM style integration API.
    Note: there is vLLM/CPU-offloading API, compare and contrast with that.

    Abstraction translation: vLLM integration (API, tokens) --> here (tokens -> keys via chunking) --> storage layer (works on keys)
    """

    def __init_metadata(self):
        # this is a fake in-place code taken from `vllm_v1_adapter.py` where such variable is initialized
        # and passed to the OpalKVCacheEngine.
        model = self._llm_model.model_name
        world_size = 1
        rank = 0
        kv_dtype = self._llm_model.torch_dtype_name
        use_mla = False
        num_layer = self._llm_model.num_hidden_layers
        chunk_size = self.opal_config["kvc"]["chunk_size"]
        head_size = self._llm_model.kv_head_size
        num_kv_head = self._llm_model.num_key_value_heads
        kv_shape = kv_shape = (num_layer, 1 if use_mla else 2, chunk_size, num_kv_head, head_size)
        role = "worker"
        metadata = OpalEngineMetadata(model, world_size, rank, "vllm", kv_dtype, kv_shape, use_mla, role)
        return metadata

    def __init__(self, opal_env: "OpalSimulatorEnvironment", opal_config: "OpalConfig", worker_id: int):
        self.opal_env = opal_env
        self.opal_config = opal_config
        self.worker_id = worker_id
        self._llm_model: OpalModelConfig = self.opal_env.llm_model
        self.token_database = OpalTokenDatabase(self.opal_config["kvc"]["chunk_size"], self.__init_metadata())
        self.storage_manager = OpalStorageManager(self.opal_env, self.opal_config, self.worker_id)
        self.log = logging.getLogger(str(self))
        self._counter_lookup = 0
        self._counter_lookup_tokens = 0

    def __str__(self):
        return f"{__class__.__name__}.{self.worker_id}"

    def store(
        self,
        tokens: Optional[list[int]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        **kwargs,
    ) -> Generator[simpy.Event, None, None]:
        """
        From OpalKVCacheEngine -> store()
        """
        if self.storage_manager.cannot_store():
            # check if we are full or empty, then no need to do useless work
            yield self.opal_env.simpy_env.timeout(0.0001)
            return

        start_time = self.opal_env.simpy_env.now
        self.log.debug(f"Storing the kvcaches for {len(tokens)} tokens")
        total_bytes = 0
        all_objects = []
        keys: List[OpalCacheEngineKey] = []
        # Let's generate chunk'ed hash keys
        for start, end, key in self.token_database.process_tokens(tokens, hashes, offsets):
            tokens_to_process = end - start
            size = self._llm_model.get_kvc_bytes(tokens_to_process)
            # tiering information will be attached later
            # for now just collecting all memory locations that can be
            memory_obj = OpalMemoryObj(size, tokens_to_process, "gpu")
            # non-contiguous to move
            all_objects.append(memory_obj)
            keys.append(key)
            total_bytes += size

        self.log.debug(f"storing KVCache from the GPU with {len(all_objects)} chunk'ed I/O objects")
        # now we have everything together, we now push it out to storage tiers
        chunk_stored: Dict[str, Tuple[int, int]] = yield from self.storage_manager.batched_put(keys, all_objects)

        total_new_bytes = 0
        for b, (all, new) in chunk_stored.items():
            self.log.debug(f"\t [{b}] : total = {all} (new: {new}, exist'ed (hence ignored): {all - new})")
            total_new_bytes += new
        onload_time_sec = self.opal_env.simpy_env.now - start_time
        self.log.debug(
            "Stored %d tokens." " size: %.4f GB," " cost %.4f ms, throughput: %.4f GB/s;",
            len(tokens),
            total_new_bytes / 1024**3,
            onload_time_sec * 1000,
            total_new_bytes / onload_time_sec / 1024**3 if onload_time_sec > 0 else 0,
        )

    def retrieve(
        self,
        tokens: list[int],
        max_fetch: Optional[int] = None,
        **kwargs,
    ) -> Generator[simpy.Event, None, np.ndarray]:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[max_fetch]: restricts the retrieve to certain number of
        tokens. This helps to resolve the issue when a lookup() returns one value
        and before retrieve() is called there are whole more new tokens generated
        that can be matched. In that case the GPU memory is over-committed badly.

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        """
        This is a coaleasing of the following code: 
         - retrieve() 
          - _process_tokens_internal() in the CacheEngine.py 
          - 
        """
        if not (max_fetch is None):
            # curtain the fetching to the max_fetch
            tokens = tokens[:max_fetch]

        ret_mask = np.zeros(len(tokens), dtype=np.bool_)
        if self.storage_manager.cannot_store():
            # check if we are full or empty, then no need to do useless work
            yield self.opal_env.simpy_env.timeout(0.0001)
            return ret_mask

        start_time = self.opal_env.simpy_env.now
        chunk_infos = []
        reordered_chunks: List[ProcessedChunk] = []
        tot_kv_size = 0
        num_required_tokens = len(tokens)
        for start, end, key in self.token_database.process_tokens(tokens=tokens):
            chunk_infos.append((key, start, end))

        # we have a list of all keys, chunk'ed
        # now we find out where they are stored
        block_mapping = yield from self.storage_manager.get_block_mapping(chunk_infos)
        # Now we bring them into the memory and then transfer to the GPU
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = yield from self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )
            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

        # back to the retrieve() function and, here in the code memory_objects
        # representing data in memory are transfer to the GPU. But we dont model
        # that for now
        """
        if len(reordered_chunks) > 0:
            _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
            self.gpu_connector.batched_to_gpu(
                list(memory_objs), list(starts), list(ends), **kwargs
            )
        """
        retrieved_tokens = int(np.sum(ret_mask))
        onload_time_sec = self.opal_env.simpy_env.now - start_time
        self.log.debug(
            "Retrieved %d out of %d required tokens (from %d total tokens)."
            " size: %.4f GB,"
            " cost %.4f ms, throughput: %.4f GB/s;",
            retrieved_tokens,
            num_required_tokens,
            len(tokens),
            tot_kv_size / 1024**3,
            onload_time_sec * 1000,
            tot_kv_size / onload_time_sec / 1024**3 if onload_time_sec > 0 else 0,
        )
        return ret_mask

    def lookup(
        self,
        tokens: Optional[List[int]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        pin: bool = False,
        num_computed_tokens: int = 0,
    ) -> Generator[simpy.Event, None, int]:

        self._counter_lookup += 1
        self._counter_lookup_tokens += len(tokens) if tokens else 0

        aligned_computed_tokens = num_computed_tokens
        res = aligned_computed_tokens

        if self.storage_manager.cannot_store():
            # check if we are full or empty, then no need to do useless work
            yield self.opal_env.simpy_env.timeout(0.0001)
            return res

        chunk_info_iterator = self.token_database.process_tokens(tokens=tokens, hashes=hashes, offsets=offsets)

        chunk_info_list = []
        keys = []
        for chunk_info in chunk_info_iterator:
            start, end, _ = chunk_info
            if end <= aligned_computed_tokens:
                continue
            chunk_info_list.append(chunk_info)
            # chunk_info contains (0 start, 1 end, 2 key)
            # chunk_info[2] is the key
            keys.append(chunk_info[2])

        # If no tokens to lookup, return immediately
        if not keys:
            return res

        # hit chunks by prefix matching
        hit_chunks, block_mapping = yield from self.storage_manager.batched_contains(keys, pin)

        for idx, (start, end, key) in enumerate(chunk_info_list):
            if idx < hit_chunks:
                res = end
                continue
            return res

        return res

    def __del__(self):
        self.log.warning(
            f"KVCManager is being destroyed"
            f"counter_lookup: {self._counter_lookup}, "
            f"counter_lookup_tokens: {self._counter_lookup_tokens}"
        )
