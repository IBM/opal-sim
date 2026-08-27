# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import simpy

from opal.core.request import LLMRequest
from opal.workloads.rag_workload import RAGWorkload


@pytest.fixture
def fake_env():
    class MockEnv:
        def __init__(self):
            self.simpy_env = simpy.Environment()
            self.simulation_time = -1

        def get_fresh_random_variable(self):
            return np.random.default_rng(42)

        def are_we_done(self):
            return False

    return MockEnv()


def make_workload(fake_env, **params):
    workload_params = {
        "workload_params": {
            "num_documents": 10,
            "document_size": 128,
            "system_prompt_size": 32,
            "docs_per_request": 4,
            "output_tokens": 8,
            "request_rate": 1.0,
            "total_requests": 5,
            **params,
        }
    }
    return RAGWorkload(fake_env, stage_id=0, workload_params=workload_params, req_router=None)


class TestRAGWorkload:
    def test_rejects_docs_per_request_over_num_documents(self, fake_env):
        with pytest.raises(ValueError):
            make_workload(fake_env, num_documents=4, docs_per_request=5)

    def test_request_hashes_include_system_prompt_and_documents(self, fake_env):
        workload = make_workload(fake_env)
        hash_ids = workload._build_request_hashes()

        assert hash_ids[: workload.system_prompt_size] == workload._system_prompt_hashes
        assert len(hash_ids) == workload.system_prompt_size + workload.docs_per_request * workload.document_size

    def test_document_hashes_are_cached_and_reused(self, fake_env):
        workload = make_workload(fake_env)
        first = workload._get_or_create_doc_hashes(0)
        second = workload._get_or_create_doc_hashes(0)

        assert first == second
        assert len(workload._doc_hash_table) == 1

    def test_document_hashes_are_disjoint_across_documents(self, fake_env):
        workload = make_workload(fake_env)
        doc0 = set(workload._get_or_create_doc_hashes(0))
        doc1 = set(workload._get_or_create_doc_hashes(1))

        assert doc0.isdisjoint(doc1)

    def test_generate_requests_stops_after_total_requests(self, fake_env):
        workload = make_workload(fake_env, total_requests=3)
        submitted = []

        class FakeQueue:
            def put(self, request):
                submitted.append(request)
                return fake_env.simpy_env.event().succeed()

        workload.req_router = type("R", (), {"input_queue": FakeQueue()})()
        fake_env.simpy_env.process(workload.generate_requests())
        fake_env.simpy_env.run()

        assert workload.is_finished
        assert workload.request_id == 3
        assert len(submitted) == 3
        assert all(isinstance(r, LLMRequest) for r in submitted)
