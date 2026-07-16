# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import logging

from opal.events import KVCEvent, KVCEventType, SystemEvent
from opal.kvc_manager import OpalKVCacheEngine
from opal.request import LLMRequest


class OpalWorkerState:
    def __init__(self, id: int):
        self.id = id
        self.name = f"KVBM-WState.{id}"
        # Setup logger
        self.log = logging.getLogger(self.name)
        # Maps chunk_hash to the set of tier names the chunk currently lives in
        self._chunk_tier: dict[int, set[str]] = {}

    def process_kvc_event(self, event: KVCEvent):
        chunk_hash = event.chunk_hash
        if event.eventType == KVCEventType.INSERT:
            self._chunk_tier.setdefault(chunk_hash, set()).add(event.tier_name)
        elif event.eventType == KVCEventType.DELETE:
            if chunk_hash in self._chunk_tier:
                self._chunk_tier[chunk_hash].discard(event.src_tier_name)
                if not self._chunk_tier[chunk_hash]:
                    del self._chunk_tier[chunk_hash]
        elif event.eventType == KVCEventType.MOVE:
            tiers = self._chunk_tier.setdefault(chunk_hash, set())
            tiers.discard(event.src_tier_name)
            tiers.add(event.tier_name)
            if not tiers:
                del self._chunk_tier[chunk_hash]
        elif event.eventType == KVCEventType.COPY:
            self._chunk_tier.setdefault(chunk_hash, set()).add(event.tier_name)
        else:
            raise Exception(f"Unknown KVCEventType: {event.eventType}")

    def process_system_event(self, sys_event: SystemEvent):
        # FIXME: we will implement the running avg. etc later, for now we are
        # only keeping track of the last update window
        self._last_sys_update = sys_event

    def match_prompt(self, hashes: list[int]) -> list[tuple[int, set[str]]]:
        """Return matched (hash, tier_set) pairs by prefix, stopping at first miss."""
        matched = []
        for h in hashes:
            if h in self._chunk_tier:
                matched.append((h, self._chunk_tier[h]))
            else:
                break
        return matched


class KVBM:
    """This class KVCache Block Manager (KVBM) represent a cluster-global view of the kv cache content.
    A copy of this information will be put at the router to make kv cache-aware
    routing decision.
    """

    def __init__(self, opal_env: "OpalSimulatorEnvironment"):
        self.opal_env = opal_env
        self.opal_config = self.opal_env.opalConfig
        self.simpy_env = self.opal_env.simpy_env
        self.name = "KVBM"
        # Setup logger
        self.log = logging.getLogger(self.name)

        self._tdb = self._get_chunker_function()
        # as we get new events we will allocate the state here!
        self._worker_state: dict[int, OpalWorkerState] = {}

        # Performance optimization: Reverse index from chunk_hash to worker_ids
        # This allows O(chunks) lookup instead of O(workers) for prefix matching
        # Key: chunk_hash (from KVCEvent), Value: set of worker_ids that have this chunk
        self._chunk_to_workers: dict[int, set[int]] = {}

    def _get_chunker_function(self):
        # -1 signifies that this is the global instance in printing
        engine = OpalKVCacheEngine(self.opal_env, self.opal_config, -1)
        tdb = engine.token_database
        return tdb

    def process_kvc_events(self, events: list[KVCEvent]):
        # self.log.debug(f"processing **KVC** event, len = {len(events)} events")
        for ex in events:
            if not (ex.worker_id in self._worker_state):
                # there is a new worker, we need to allocate a new entry
                self._worker_state[ex.worker_id] = OpalWorkerState(ex.worker_id)
                self.log.debug(f"Allocated a new worker state for worker.{ex.worker_id}")

            # Process the event in worker state
            self._worker_state[ex.worker_id].process_kvc_event(ex)

            # Update reverse index: chunk_hash -> worker_ids
            # This enables fast lookup of which workers have a specific chunk
            chunk_hash = ex.chunk_hash
            if ex.eventType == KVCEventType.INSERT:
                if chunk_hash not in self._chunk_to_workers:
                    self._chunk_to_workers[chunk_hash] = set()
                self._chunk_to_workers[chunk_hash].add(ex.worker_id)
            elif ex.eventType == KVCEventType.DELETE:
                if chunk_hash in self._chunk_to_workers:
                    self._chunk_to_workers[chunk_hash].discard(ex.worker_id)
                    if not self._chunk_to_workers[chunk_hash]:
                        del self._chunk_to_workers[chunk_hash]
            # MOVE and COPY: worker still holds the chunk, no change to reverse index

    def process_system_events(self, events: list[SystemEvent]):
        # self.log.debug(f"processing **systems** event, len = {len(events)} events")
        for ex in events:
            if not (ex.worker_id in self._worker_state):
                # there is a new worker, we need to allocate a new entry
                self._worker_state[ex.worker_id] = OpalWorkerState(ex.worker_id)
                self.log.debug(f"Allocated a new worker state for worker.{ex.worker_id}")
            # update the entires
            self._worker_state[ex.worker_id].process_system_event(ex)

    def scorer(self, req: LLMRequest):
        """Given a prompt, the function returns a list of pods, worker id to schedule it on
        It can maximize various systems aspect of KVCache management.

        Performance optimization: Instead of checking all workers (O(N) where N=10,000),
        we use the reverse index to find only workers that have relevant chunks (O(M) where M << N).

        Args:
            req (LLMRequest): LLM Request to schedule

        Returns:
            dict[int, float]: worker_id -> match_score (0.0 to 1.0)
        """
        # Convert request tokens to chunk hashes
        chunk_hashes = [c for _, _, c in self._tdb.process_tokens(req.hash_ids)]

        # Find candidate workers: those that have at least one of the required chunks
        # This is much faster than checking all workers when worker count is large
        candidate_workers = set()
        for chunk_hash in chunk_hashes:
            if chunk_hash in self._chunk_to_workers:
                # Add all workers that have this chunk to candidates
                candidate_workers.update(self._chunk_to_workers[chunk_hash])

        # If no workers have any chunks, return empty scores
        # The router will fall back to random/round-robin selection
        if not candidate_workers:
            return {}

        # Score only the candidate workers (typically much smaller than total workers)
        # Score = fraction of chunks that match (0.0 = no match, 1.0 = perfect match)
        score = {}
        for worker_id in candidate_workers:
            w = self._worker_state[worker_id]
            matched = w.match_prompt(chunk_hashes)
            score[worker_id] = len(matched) / len(chunk_hashes)

        return score
