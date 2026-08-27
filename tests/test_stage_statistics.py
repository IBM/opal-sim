# SPDX-License-Identifier: Apache-2.0
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
