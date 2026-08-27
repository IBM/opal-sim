# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging

import simpy

from opal.core.request import LLMRequest
from opal.utils.util import generate_time_with_rate_variation, safe_process
from opal.workloads.abstract_workload import AbstractWorkload


class RAGWorkload(AbstractWorkload):
    """Simulates RAG requests.

    Each request is composed of a shared system prompt followed by N randomly
    selected documents (without replacement within a request). Hash IDs for
    the system prompt and for each document are generated once and cached in
    an internal hash table so that repeated selection of the same document
    yields the same hash sequence (enabling prefix-cache reuse).

    Example workload stage configuration (JSON):

        {
          "workload": {
            "stages": [
              {
                "type": "RAGWorkload",
                "workload_params": {
                  "num_documents": 10,
                  "document_size": 16384,
                  "system_prompt_size": 1024,
                  "docs_per_request": 4,
                  "output_tokens": 128,
                  "request_rate": 2.0,
                  "jitter": 0.0,
                  "total_requests": 500,
                  "time_duration_sec": -1,
                  "max_concurrent_requests": 16
                }
              }
            ]
          }
        }
    """

    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        stage_id: int,
        workload_params: dict,
        req_router,
        name: str = "Workload(RAG)",
    ):
        super().__init__(opal_env, stage_id, workload_params, req_router, name)
        self.log = logging.getLogger(self.name)

        params = self.workload_params["workload_params"]
        self.num_documents = int(params.get("num_documents", 10))
        self.document_size = int(params.get("document_size", 16 * 1024))
        self.system_prompt_size = int(params.get("system_prompt_size", 1024))
        self.output_tokens = int(params.get("output_tokens", 128))
        self.docs_per_request = int(params.get("docs_per_request", 4))

        self.request_rate = float(params.get("request_rate", 1.0))
        self.request_interval = 1.0 / self.request_rate
        self.jitter = float(params.get("jitter", 0.0))
        self.total_requests = int(params.get("total_requests", -1))
        self.max_concurrent_requests = int(params.get("max_concurrent_requests", 16))
        self._inflight_requests = []

        if self.docs_per_request > self.num_documents:
            raise ValueError(
                f"docs_per_request ({self.docs_per_request}) cannot exceed " f"num_documents ({self.num_documents})"
            )

        # Monotonic hash-id allocator: every call returns a fresh unique int.
        self._next_hash_id = 0

        # Hash table: document index -> list[int] of hash IDs (length == document_size).
        # Populated lazily on first use; reused on subsequent selections.
        self._doc_hash_table: dict[int, list[int]] = {}

        # System prompt hashes are generated once and reused across all requests.
        self._system_prompt_hashes = self._allocate_hashes(self.system_prompt_size)

        self.is_finished = False
        self.request_id = -1

        self.log.info(
            f"Initialized {self.name}: num_documents={self.num_documents}, "
            f"document_size={self.document_size}, system_prompt_size={self.system_prompt_size}, "
            f"docs_per_request={self.docs_per_request}, output_tokens={self.output_tokens}, "
            f"request_rate={self.request_rate}, total_requests={self.total_requests}, "
            f"max_concurrent_requests={self.max_concurrent_requests}"
        )

    def __str__(self):
        return f"{self.name}.{self.stage_id}"

    def _allocate_hashes(self, n: int) -> list[int]:
        """Allocate `n` fresh, monotonically-increasing unique hash IDs."""
        start = self._next_hash_id
        self._next_hash_id += n
        return list(range(start, start + n))

    def _get_or_create_doc_hashes(self, doc_idx: int) -> list[int]:
        """Return cached hashes for `doc_idx`, generating them on first use."""
        cached = self._doc_hash_table.get(doc_idx)
        if cached is not None:
            return cached
        hashes = self._allocate_hashes(self.document_size)
        self._doc_hash_table[doc_idx] = hashes
        return hashes

    def _select_documents(self) -> list[int]:
        """Randomly pick `docs_per_request` distinct document indices."""
        # numpy Generator.choice with replace=False guarantees uniqueness.
        chosen = self.rng.choice(self.num_documents, size=self.docs_per_request, replace=False)
        chosen.sort()  # sort for deterministic ordering, typical RAG behavior
        self.log.info(f"{self.name} | Request {self.request_id} | Docs chosen: {chosen}")
        return [int(i) for i in chosen]

    def _build_request_hashes(self) -> list[int]:
        """Assemble the full hash-id sequence: system prompt followed by selected documents."""
        hash_ids: list[int] = list(self._system_prompt_hashes)
        for doc_idx in self._select_documents():
            hash_ids.extend(self._get_or_create_doc_hashes(doc_idx))
        return hash_ids

    def _intra_request_delay(self) -> float:
        if self.jitter > 0:
            return generate_time_with_rate_variation(self.request_rate, self.jitter)
        return self.request_interval

    def _request_generation_stops(self) -> bool:
        global_timeout = (
            self.simpy_env.now > self.opal_env.simulation_time if self.opal_env.simulation_time > 0 else False
        )
        local_timeout = self.local_timeout
        local_done = self.request_id >= self.total_requests if self.total_requests > 0 else False
        if global_timeout or local_timeout or local_done:
            self.log.info(
                f"Workload generation stops: global_timeout={global_timeout}, "
                f"local_timeout={local_timeout}, local_done={local_done}"
            )
            return True
        return False

    def generate_requests(self):
        self.request_id = 0
        outstanding: set = set()

        def _on_complete(request):
            outstanding.discard(request.id)

        while not self._request_generation_stops():
            # Block submission while at the concurrency cap; wake when any
            # in-flight request completes.
            while self.max_concurrent_requests > 0 and len(outstanding) >= self.max_concurrent_requests:
                events = [r.has_completed for r in self._inflight_requests if r.id in outstanding]
                if not events:
                    break
                yield simpy.AnyOf(self.simpy_env, events)
                # Drop any completed ids from the outstanding set.
                for r in list(self._inflight_requests):
                    if r.has_completed.triggered and r.id in outstanding:
                        outstanding.discard(r.id)

            hash_ids = self._build_request_hashes()
            input_length = len(hash_ids)
            request = LLMRequest(
                self.simpy_env,
                self.stage_id,
                input_length,
                hash_ids=hash_ids,
                output_length=self.output_tokens,
            )
            self.request_id += 1
            self.generated_requests += 1
            outstanding.add(request.id)
            self._inflight_requests.append(request)
            self.log.debug(
                f"{request} generated (input_length={input_length}, "
                f"unique_docs_seen={len(self._doc_hash_table)}, "
                f"outstanding={len(outstanding)}/{self.max_concurrent_requests})"
            )
            yield self.req_router.input_queue.put(request)
            yield self.simpy_env.timeout(self._intra_request_delay())

        self.is_finished = True
        self.log.info(
            f"RAG workload generation (stage_id={self.stage_id}) finished with "
            f"{self.request_id} requests, {len(self._doc_hash_table)} unique documents materialized"
        )
