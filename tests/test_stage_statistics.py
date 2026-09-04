# SPDX-License-Identifier: Apache-2.0
import io

import simpy

from opal.core.request import LLMRequest
from opal.stats.stage_statistics import StageStatistics


def make_finished_request(hit_tokens: int) -> LLMRequest:
    env = simpy.Environment()
    request = LLMRequest(env, stage_id=0, input_length=128, output_length=8)
    request.stats.add_scheduler_timestamp(env.now)
    request.stats.mark_prefill_done()
    request.stats.set_prefix_hit_tokens(hit_tokens)
    return request


class TestStageStatisticsKVCHitTokens:
    def test_add_finished_request_records_prefix_hit_tokens(self):
        stats = StageStatistics()
        stats.add_finished_request(make_finished_request(1234))
        stats.add_finished_request(make_finished_request(0))

        assert stats.raw_kvc_hit_tokens == [1234, 0]

    def test_round_trips_through_to_dict_and_from_dict(self):
        stats = StageStatistics()
        stats.add_finished_request(make_finished_request(4321))

        restored = StageStatistics.from_dict(stats.to_dict())

        assert restored.raw_kvc_hit_tokens == [4321]

    def test_from_dict_defaults_missing_field_to_empty_list(self):
        stats = StageStatistics()
        stats.add_finished_request(make_finished_request(999))
        data = stats.to_dict()
        del data["raw_kvc_hit_tokens"]

        restored = StageStatistics.from_dict(data)

        assert restored.raw_kvc_hit_tokens == []


class TestStageStatisticsSummaryLogFile:
    def test_print_summary_results_writes_to_log_file(self):
        stats = StageStatistics()
        stats.add_finished_request(make_finished_request(64))
        stats.queued_requests = 1
        stats.stage_time_start = 0
        stats.stage_time_end = 1

        log_file = io.StringIO()
        stats.print_summary_results(log_file=log_file)

        output = log_file.getvalue()
        assert "Serving Benchmark Result" in output
        assert "Successful requests" in output

    def test_print_summary_results_without_log_file_does_not_raise(self):
        stats = StageStatistics()
        stats.add_finished_request(make_finished_request(64))
        stats.queued_requests = 1
        stats.stage_time_start = 0
        stats.stage_time_end = 1

        stats.print_summary_results()
