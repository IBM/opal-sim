# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from opal.config.opal_config import OpalConfig
from opal.core.opal import OpalSimulator
from opal.cost.__main__ import main as cost_main
from opal.cost.cost_model import COST_FILE, gpu_instances, price_run, stage_usage

ROOT = Path(__file__).resolve().parent.parent

# Small hand-made profile so every expected number below can be checked by hand.
PROFILE = {
    "name": "test-cloud",
    "currency": "USD",
    "instances": {
        "big.8gpu": {
            "accelerator": "H100",
            "accelerator_count": 8,
            "prices": {"r1": {"on_demand": 80.0, "spot": 24.0}, "r2": {"on_demand": 64.0}},
        },
        "small.2gpu": {
            "accelerator": "H100",
            "accelerator_count": 2,
            "prices": {"r1": {"on_demand": 20.0, "spot": 8.0}},
        },
        "tiny.1gpu": {"accelerator": "H100", "accelerator_count": 1, "prices": {"r1": {"on_demand": 9.0}}},
        "other.1gpu": {"accelerator": "A100", "accelerator_count": 1, "prices": {"r1": {"on_demand": 3.0}}},
    },
}
# 3 workers x tp 2 = 6 GPUs for one hour, busy 75% of it (4.5 GPU-hours).
# Workers per instance: big.8gpu 4, small.2gpu 1, tiny.1gpu 0.
CONFIG = {"worker": {"hw": {"gpu": "H100", "tp": 2}}, "simulation": {"num_workers": 3}}
HW = CONFIG["worker"]["hw"]


def make_stage(duration=3600.0, workers=(3, 3), busy=(4050.0, 4050.0), tokens=((100, 10), (300, 90))):
    return {
        "stage_time_start": 0.0,
        "stage_time_end": duration,
        "per_unit_workers": list(workers),
        "per_unit_gpu_busy_time": list(busy),
        "finished_requests": len(tokens),
        "input_output_tokens_sz": [list(t) for t in tokens],
    }


# Session a: 2 turns, 40 output, 40 recomputed and 260 cached input tokens, open from 0 to 40 s.
# Session b: 1 turn, 60 output and 300 recomputed input tokens, open from 10 to 70 s.
SESSION_STAGE = {
    **make_stage(tokens=((100, 10), (300, 60), (200, 30))),
    "raw_session_ids": ["a", "b", "a"],
    "raw_kvc_hit_tokens": [80, 0, 180],
    "arrival_completion_times": [(0.0, 10.0), (10.0, 70.0), (20.0, 40.0)],
}


@pytest.fixture
def profile_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    return str(path)


def price(profile_file, workers=(3, 3), hw=HW, **cost_config):
    config = {"worker": {"hw": hw}, "simulation": {"num_workers": 3}}
    [report] = price_run([make_stage(workers=workers)], config, {"profile": profile_file, **cost_config})
    return report


def price_sessions(profile_file, **cost_config):
    [report] = price_run([SESSION_STAGE], CONFIG, {"profile": profile_file, "region": "r1", **cost_config})
    return report["sessions"]


def write_cost_config(tmp_path, cost):
    """configs/defaults.json (1 H100 worker) plus a `cost` section."""
    config = json.loads((ROOT / "configs/defaults.json").read_text())
    config["cost"] = cost
    path = tmp_path / "cost_config.json"
    path.write_text(json.dumps(config))
    return path


def load_config(config_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = OpalConfig()
    config.initialize(str(config_path))
    return config


def run_sim(config_path, tmp_path, monkeypatch):
    OpalSimulator.from_config(config=load_config(config_path, monkeypatch), output_dir=str(tmp_path)).run(10)
    [run_dir] = tmp_path.glob("sim-*")
    return run_dir


@pytest.fixture
def cost_run_dir(tmp_path, monkeypatch, profile_file):
    """A 10 s run priced with the hand-made profile: its one tp 1 worker fits a tiny.1gpu."""
    return run_sim(write_cost_config(tmp_path, {"profile": profile_file, "region": "r1"}), tmp_path, monkeypatch)


class TestGpuInstances:
    def test_lists_every_instance_type_and_region_offering_the_gpu(self):
        options = gpu_instances(PROFILE, "h100")

        assert sorted((o["region"], o["instance_type"], o["price_per_gpu_hour"]) for o in options) == [
            ("r1", "big.8gpu", 10.0),
            ("r1", "small.2gpu", 10.0),
            ("r1", "tiny.1gpu", 9.0),
            ("r2", "big.8gpu", 8.0),
        ]

    def test_spot_lists_only_options_with_a_spot_price(self):
        assert len(gpu_instances(PROFILE, "H100", "spot")) == 2

    def test_instance_type_filter(self):
        options = gpu_instances(PROFILE, "H100", instance_type="big.8gpu")

        assert {o["instance_type"] for o in options} == {"big.8gpu"}

    def test_unknown_gpu_lists_the_profile_gpus(self):
        with pytest.raises(ValueError, match="GPUs in this profile"):
            gpu_instances(PROFILE, "B200")

    def test_unknown_pricing_model(self):
        with pytest.raises(ValueError, match="Unknown pricing model"):
            gpu_instances(PROFILE, "H100", pricing="reserved")


class TestStageUsage:
    def test_provisioned_and_busy_gpu_hours(self):
        usage = stage_usage(make_stage(), HW, 3)

        assert usage["provisioned_gpu_hours"] == pytest.approx(6.0)
        assert usage["busy_gpu_hours"] == pytest.approx(4.5)
        assert usage["busy_share"] == pytest.approx(0.75)

    def test_token_and_request_totals(self):
        usage = stage_usage(make_stage(), HW, 3)

        assert usage["input_tokens"] == 400
        assert usage["output_tokens"] == 100
        assert usage["finished_requests"] == 2

    def test_busy_time_is_capped_at_provisioned(self):
        # A batch counts in the stage where it ends, so a stage can record more busy time than it provisioned.
        usage = stage_usage(make_stage(busy=(9000.0, 9000.0)), HW, 3)

        assert usage["busy_gpu_hours"] == pytest.approx(6.0)
        assert usage["busy_share"] == 1.0

    def test_mean_of_worker_samples_under_autoscaling(self):
        usage = stage_usage(make_stage(workers=(1, 1, 2, 4)), HW, 1)

        assert usage["provisioned_gpu_hours"] == pytest.approx(4.0)
        assert usage["workers_time_share"] == {1: 0.5, 2: 0.25, 4: 0.25}

    def test_no_worker_samples_falls_back_to_initial_workers(self):
        usage = stage_usage(make_stage(duration=0.5, workers=()), HW, 3)

        assert usage["mean_workers"] == 3.0

    def test_stage_that_never_ran_costs_nothing(self):
        # A stage that never starts keeps its times at -1.
        usage = stage_usage({**make_stage(), "stage_time_start": -1, "stage_time_end": -1}, HW, 3)

        assert usage["provisioned_gpu_hours"] == 0.0
        assert usage["busy_share"] == 0.0


class TestGpuHourRental:
    def test_cheapest_instance_type_in_the_region(self, profile_file):
        # r1: 3 x small.2gpu = $60 beats 1 x big.8gpu = $80; tiny.1gpu can't hold a tp 2 worker.
        report = price(profile_file, region="r1")

        assert report["pricing"]["instance_type"] == "small.2gpu"
        assert report["total"] == pytest.approx(60.0)

    def test_workers_gpus_split_into_busy_and_idle(self, profile_file):
        # The $60 of the workers' GPUs is 75% busy: $45 busy, $15 idle; every rented GPU holds a worker.
        report = price(profile_file, region="r1")

        assert report["breakdown"] == {
            "gpu_busy": pytest.approx(45.0),
            "gpu_idle": pytest.approx(15.0),
            "gpu_unused": pytest.approx(0.0),
        }
        assert report["pricing"]["gpus_used"] == 6.0
        assert report["pricing"]["gpus_rented"] == 6.0

    def test_whole_instances_are_billed(self, profile_file):
        # One 8-GPU instance for 6 GPUs: pay $80, of which 6 GPU-hours x $10 hold workers.
        report = price(profile_file, region="r1", instance_type="big.8gpu")

        assert report["total"] == pytest.approx(80.0)
        assert report["breakdown"] == {
            "gpu_busy": pytest.approx(45.0),
            "gpu_idle": pytest.approx(15.0),
            "gpu_unused": pytest.approx(20.0),
        }
        assert report["pricing"]["gpus_used"] == 6.0
        assert report["pricing"]["gpus_rented"] == 8.0

    def test_instances_counted_per_worker_sample(self, profile_file):
        # Autoscaling: ceil(5 / 4) = 2 instances in the last sample -> 1.25 instances on average.
        report = price(profile_file, workers=(1, 1, 2, 5), region="r1", instance_type="big.8gpu")

        assert report["total"] == pytest.approx(100.0)
        assert report["pricing"]["gpus_used"] == 4.5
        assert report["pricing"]["gpus_rented"] == 10.0

    def test_per_request_and_per_1M_tokens(self, profile_file):
        report = price(profile_file, region="r1")

        assert report["per_request"] == pytest.approx(30.0)
        assert report["per_1M_tokens"] == pytest.approx(60.0 / 500 * 1e6)

    def test_by_region_cheapest_first(self, profile_file):
        report = price(profile_file, region="r1")

        assert [(r["region"], r["cost"]) for r in report["by_region"]] == [("r1", 60.0), ("r2", 64.0)]

    def test_default_region_is_the_cheapest(self, profile_file):
        assert price(profile_file)["pricing"]["region"] == "r1"

    def test_cost_tie_goes_to_fewer_instances(self, profile_file):
        # Spot in r1: 1 x big.8gpu = 3 x small.2gpu = $24.
        report = price(profile_file, pricing="spot")

        assert report["pricing"]["instance_type"] == "big.8gpu"
        assert report["total"] == pytest.approx(24.0)
        assert report["breakdown"] == {
            "gpu_busy": pytest.approx(13.5),
            "gpu_idle": pytest.approx(4.5),
            "gpu_unused": pytest.approx(6.0),
        }

    def test_policy_and_pricing_names_are_case_insensitive(self, profile_file):
        report = price(profile_file, policy="GPU_Hour_Rental", pricing="Spot")

        assert report["policy"] == "gpu_hour_rental"
        assert report["pricing"]["pricing_model"] == "spot"

    def test_zero_length_stage_ranked_by_cost_per_hour(self, profile_file):
        # It costs 0 everywhere, but 3 x $20 per hour still beats 1 x $80.
        [report] = price_run([make_stage(duration=0.0)], CONFIG, {"profile": profile_file, "region": "r1"})

        assert report["total"] == 0.0
        assert report["pricing"]["instance_type"] == "small.2gpu"


class TestPriceRunErrors:
    def test_missing_profile(self):
        with pytest.raises(ValueError, match="needs a 'profile'"):
            price_run([make_stage()], CONFIG, {})

    def test_unknown_policy(self, profile_file):
        with pytest.raises(ValueError, match="Unknown cost policy"):
            price_run([make_stage()], CONFIG, {"profile": profile_file, "policy": "magic"})

    def test_region_without_a_price(self, profile_file):
        with pytest.raises(ValueError, match="Regions with a price"):
            price_run([make_stage()], CONFIG, {"profile": profile_file, "region": "r9"})

    def test_worker_larger_than_any_instance(self, profile_file):
        with pytest.raises(ValueError, match="GPUs of a worker"):
            price(profile_file, hw={"gpu": "H100", "tp": 16})


class TestSessionCosts:
    def test_output_only_weights(self, profile_file):
        # The $45 busy split 40 : 60 by output tokens, the $15 idle 40 s : 60 s by time open.
        sessions = price_sessions(profile_file, token_weights={"recomputed_input": 0, "cached_input": 0})

        assert sessions["per_session"] == {"a": pytest.approx(18.0 + 6.0), "b": pytest.approx(27.0 + 9.0)}

    def test_summary_over_sessions(self, profile_file):
        sessions = price_sessions(profile_file, token_weights={"recomputed_input": 0, "cached_input": 0})

        assert sessions["sessions"] == 2
        assert sessions["median"] == pytest.approx(30.0)
        assert sessions["max"] == pytest.approx(36.0)

    def test_token_weights_change_the_busy_split(self, profile_file):
        # A recomputed input token weighs half an output token: busy split 40 + 20 : 60 + 150.
        sessions = price_sessions(profile_file, token_weights={"recomputed_input": 0.5, "cached_input": 0})

        assert sessions["per_session"] == {"a": pytest.approx(10.0 + 6.0), "b": pytest.approx(35.0 + 9.0)}
        assert sessions["token_weights"] == {"output": 1.0, "recomputed_input": 0.5, "cached_input": 0}

    def test_default_token_weights(self, profile_file):
        sessions = price_sessions(profile_file)

        assert sessions["token_weights"] == {"output": 1.0, "recomputed_input": 0.3, "cached_input": 0.01}

    def test_unknown_weight_name_is_rejected(self, profile_file):
        with pytest.raises(ValueError, match="token_weights"):
            price_sessions(profile_file, token_weights={"input": 1.0})

    def test_no_session_ids_means_no_split(self, profile_file):
        # Every workload but the OTel trace replay.
        [report] = price_run([make_stage()], CONFIG, {"profile": profile_file, "region": "r1"})

        assert report["sessions"] is None


class TestEndToEnd:
    def test_report_written_per_stage(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())

        assert report["total"] > 0
        assert report["total"] == pytest.approx(sum(report["breakdown"].values()))
        assert report["pricing"]["instance_type"] == "tiny.1gpu"
        assert report["pricing"]["gpus_used"] == 1.0
        assert report["pricing"]["gpus_rented"] == 1.0
        assert report["sessions"] is None  # defaults.json replays no sessions

    def test_cost_lines_appended_to_simulation_log(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())
        log = (cost_run_dir / "simulation.log").read_text()

        busy = f"{report['usage']['busy_share']:.0%} busy"
        assert f"stage_0: {report['total']:.4f} USD | 1 x tiny.1gpu, r1, on_demand | 1 of 1 GPUs used, {busy}" in log
        # Stage 1 never starts in 10 s: it costs nothing but still names the 1-GPU instance.
        assert "stage_1: 0.0000 USD | 1 x tiny.1gpu, r1, on_demand | 1 of 1 GPUs used, 0% busy" in log

    def test_offline_pricing_matches_the_run(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())

        offline = cost_main([str(cost_run_dir), "--no-write"])

        assert offline[0]["total"] == pytest.approx(report["total"])

    def test_busy_seconds_add_up_to_gpu_utilization(self, cost_run_dir):
        # One worker in a stage that starts at t = 0: busy seconds = Opal's cumulative utilization x time.
        stats = json.loads((cost_run_dir / "stage_0" / "opal_stats.json").read_text())
        utilization = stats["per_unit_gpu_utilization"]

        assert sum(stats["per_unit_gpu_busy_time"]) == pytest.approx(utilization[-1] / 100 * len(utilization))

    def test_no_cost_section_means_no_cost_report(self, tmp_path, monkeypatch):
        run_dir = run_sim(ROOT / "configs/defaults.json", tmp_path, monkeypatch)

        assert not list(run_dir.glob(f"stage_*/{COST_FILE}"))

    def test_bad_cost_section_fails_before_the_run(self, tmp_path, monkeypatch, profile_file):
        config = load_config(write_cost_config(tmp_path, {"profile": profile_file, "policy": "magic"}), monkeypatch)

        with pytest.raises(ValueError, match="Unknown cost policy"):
            OpalSimulator.from_config(config=config, output_dir=str(tmp_path))
        assert not list(tmp_path.glob("sim-*"))
