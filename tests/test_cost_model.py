# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import pytest

from opal.config.opal_config import OpalConfig
from opal.core.opal import OpalSimulator
from opal.cost.__main__ import main as cost_main
from opal.cost.cost_model import (
    COST_FILE,
    format_report,
    gpu_instances,
    at_llm_seconds,
    price_run,
    stage_usage,
)

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


def make_stage(duration=3600.0, workers=(3, 3), busy=(4050.0, 4050.0), tokens=((100, 10), (300, 90)), cached=None):
    return {
        "stage_time_start": 0.0,
        "stage_time_end": duration,
        "per_unit_workers": list(workers),
        "per_unit_gpu_busy_time": list(busy),
        "finished_requests": len(tokens),
        "input_output_tokens_sz": [list(t) for t in tokens],
        "raw_kvc_hit_tokens": list(cached) if cached else [0] * len(tokens),
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


def session_cost(sessions):
    return {sid: s["cost_usd"] for sid, s in sessions["per_session"].items()}


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

        assert sorted((o["region"], o["instance_type"], o["usd_per_gpu_hour"]) for o in options) == [
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

    def test_errors_name_what_is_available(self):
        with pytest.raises(ValueError, match="GPUs in this profile"):
            gpu_instances(PROFILE, "B200")
        with pytest.raises(ValueError, match="Unknown pricing model"):
            gpu_instances(PROFILE, "H100", pricing="reserved")


class TestStageUsage:
    def test_provisioned_and_busy_gpu_hours(self):
        usage = stage_usage(make_stage(), HW, 3)

        assert usage["provisioned_gpu_hours"] == pytest.approx(6.0)
        assert usage["busy_gpu_hours"] == pytest.approx(4.5)
        assert usage["busy_pct"] == pytest.approx(75.0)

    def test_token_and_request_totals(self):
        usage = stage_usage(make_stage(), HW, 3)

        assert usage["input_tokens"] == 400
        assert usage["output_tokens"] == 100
        assert usage["finished_requests"] == 2

    def test_busy_time_is_capped_at_provisioned(self):
        # A batch counts in the stage where it ends, so a stage can record more busy time than it provisioned.
        usage = stage_usage(make_stage(busy=(9000.0, 9000.0)), HW, 3)

        assert usage["busy_gpu_hours"] == pytest.approx(6.0)
        assert usage["busy_pct"] == 100.0

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
        assert usage["busy_pct"] == 0.0


class TestGpuHourRental:
    def test_cheapest_instance_type_split_into_busy_and_idle(self, profile_file):
        # r1: 3 x small.2gpu = $60 beats 1 x big.8gpu = $80; tiny.1gpu can't hold a tp 2 worker.
        # The $60 of the workers' GPUs is 75% busy: $45 busy, $15 idle; every rented GPU holds a worker.
        report = price(profile_file, region="r1")

        assert report["pricing"]["instance_type"] == "small.2gpu"
        assert report["total_usd"] == pytest.approx(60.0)
        assert report["breakdown_usd"] == {
            "gpu_busy": pytest.approx(45.0),
            "gpu_idle": pytest.approx(15.0),
            "gpu_unused": pytest.approx(0.0),
            "billing_rounding": 0.0,
        }
        assert report["pricing"]["gpus_used"] == 6.0
        assert report["pricing"]["gpus_rented"] == 6.0

    def test_whole_instances_are_billed(self, profile_file):
        # One 8-GPU instance for 6 GPUs: pay $80, of which 6 GPU-hours x $10 hold workers.
        report = price(profile_file, region="r1", instance_type="big.8gpu")

        assert report["total_usd"] == pytest.approx(80.0)
        assert report["breakdown_usd"] == {
            "gpu_busy": pytest.approx(45.0),
            "gpu_idle": pytest.approx(15.0),
            "gpu_unused": pytest.approx(20.0),
            "billing_rounding": 0.0,
        }
        assert report["pricing"]["gpus_used"] == 6.0
        assert report["pricing"]["gpus_rented"] == 8.0

    def test_instances_counted_per_worker_sample(self, profile_file):
        # Autoscaling: ceil(5 / 4) = 2 instances in the last sample -> 1.25 instances on average.
        report = price(profile_file, workers=(1, 1, 2, 5), region="r1", instance_type="big.8gpu")

        assert report["total_usd"] == pytest.approx(100.0)
        assert report["pricing"]["gpus_used"] == 4.5
        assert report["pricing"]["gpus_rented"] == 10.0

    def test_per_request_and_per_1M_tokens(self, profile_file):
        report = price(profile_file, region="r1")

        assert report["per_request_usd"] == pytest.approx(30.0)
        assert report["per_1M_tokens_usd"] == pytest.approx(60.0 / 500 * 1e6)

    def test_by_region_cheapest_first_and_default_region(self, profile_file):
        report = price(profile_file)

        assert [(r["region"], r["cost_usd"]) for r in report["by_region"]] == [("r1", 60.0), ("r2", 64.0)]
        assert report["pricing"]["region"] == "r1"

    def test_cost_tie_goes_to_fewer_instances(self, profile_file):
        # Spot in r1: 1 x big.8gpu = 3 x small.2gpu = $24.
        report = price(profile_file, pricing="spot")

        assert report["pricing"]["instance_type"] == "big.8gpu"
        assert report["total_usd"] == pytest.approx(24.0)
        assert report["breakdown_usd"] == {
            "gpu_busy": pytest.approx(13.5),
            "gpu_idle": pytest.approx(4.5),
            "gpu_unused": pytest.approx(6.0),
            "billing_rounding": 0.0,
        }

    def test_policy_and_pricing_names_are_case_insensitive(self, profile_file):
        report = price(profile_file, policy="GPU_Hour_Rental", pricing="Spot")

        assert report["policy"] == "gpu_hour_rental"
        assert report["pricing"]["pricing_model"] == "spot"

    def test_zero_length_stage_ranked_by_cost_per_hour(self, profile_file):
        # It costs 0 everywhere, but 3 x $20 per hour still beats 1 x $80.
        [report] = price_run([make_stage(duration=0.0)], CONFIG, {"profile": profile_file, "region": "r1"})

        assert report["total_usd"] == 0.0
        assert report["pricing"]["instance_type"] == "small.2gpu"


class TestBilling:
    def test_exact_by_default(self, profile_file):
        report = price(profile_file, region="r1")

        assert report["billing"] == {"increment_s": 0.0, "minimum_s": 0.0, "exact": True}
        assert report["breakdown_usd"]["billing_rounding"] == 0.0

    def test_hourly_billing_rounds_each_instance_up(self, profile_file):
        # 30 min on 3 x small.2gpu ($20/h each), billed as 3 whole hours: $30 exact + $30 rounding.
        config = {"worker": {"hw": HW}, "simulation": {"num_workers": 3}}
        cost = {"profile": profile_file, "region": "r1", "billing_increment_s": 3600}
        [report] = price_run([make_stage(duration=1800.0, busy=(2025.0, 2025.0))], config, cost)

        assert report["total_usd"] == pytest.approx(60.0)
        assert report["breakdown_usd"]["billing_rounding"] == pytest.approx(30.0)
        assert report["gpu_hours"]["billing_rounding"] == pytest.approx(3.0)

    def test_minimum_billed_time(self, profile_file):
        # AWS-style (per second, 60 s minimum): a 30 s stage on 3 x small.2gpu is billed 60 s per instance.
        config = {"worker": {"hw": HW}, "simulation": {"num_workers": 3}}
        cost = {"profile": profile_file, "region": "r1", "billing_increment_s": 1, "billing_minimum_s": 60}
        [report] = price_run([make_stage(duration=30.0, busy=(0.0,))], config, cost)

        assert report["breakdown_usd"]["billing_rounding"] == pytest.approx(3 * 20.0 * 30 / 3600)

    def test_instance_added_late_is_rounded_from_its_start(self, profile_file):
        # 1, 1, 2, 5 workers on big.8gpu: the 2nd instance runs only the last 900 s and is billed a whole hour.
        report = price(
            profile_file, workers=(1, 1, 2, 5), region="r1", instance_type="big.8gpu", billing_increment_s=3600
        )

        assert report["breakdown_usd"]["billing_rounding"] == pytest.approx(0.75 * 80.0)
        assert report["total_usd"] == pytest.approx(100.0 + 60.0)
        # by_region is ranked with the rounding included: r2 (64 / h) is cheaper than r1 (80 / h).
        assert [(r["region"], r["cost_usd"]) for r in report["by_region"]] == [("r2", 128.0), ("r1", 160.0)]

    def test_console_shows_billing_and_zero_lines_only_when_they_cost(self, profile_file):
        exact = "\n".join(format_report(price(profile_file, region="r1")))
        hourly = price(
            profile_file, workers=(1, 1, 2, 5), region="r1", instance_type="big.8gpu", billing_increment_s=3600
        )
        hourly = "\n".join(format_report(hourly))

        assert "billing" not in exact and "GPU unused" not in exact  # 3 x small.2gpu: every GPU holds a worker
        assert "billing   in 3600 s steps, 0 s minimum per instance" in hourly
        assert "billing rounding" in hourly and "GPU unused" in hourly

    def test_negative_billing_is_rejected(self, profile_file):
        with pytest.raises(ValueError, match="billing_increment_s"):
            price(profile_file, billing_increment_s=-1)


class TestTokenPrices:
    # 1,100 input tokens of which 100 cached -> 1,000 recomputed; 100 output. On 3 x small.2gpu in r1 for $60.
    STAGE = make_stage(tokens=((600, 50), (500, 50)), cached=(100, 0))

    def price(self, profile_file, **cost_config):
        [report] = price_run([self.STAGE], CONFIG, {"profile": profile_file, "region": "r1", **cost_config})
        return report

    def test_bill_split_by_weighted_tokens(self, profile_file):
        # Weights 1 / 0.5 / 0: 100 + 0.5 * 1,000 = 600 weighted tokens share $60 -> $0.1 per output token.
        report = self.price(profile_file, token_weights={"recomputed_input": 0.5, "cached_input": 0})

        assert report["per_1M_tokens_by_kind_usd"] == {
            "output": pytest.approx(100_000.0),
            "recomputed_input": pytest.approx(50_000.0),
            "cached_input": 0.0,
        }
        assert report["usage"]["cached_input_tokens"] == 100

    def test_prices_add_back_up_to_the_bill(self, profile_file):
        report = self.price(profile_file)
        prices, u = report["per_1M_tokens_by_kind_usd"], report["usage"]
        billed = (
            prices["output"] * u["output_tokens"]
            + prices["recomputed_input"] * (u["input_tokens"] - u["cached_input_tokens"])
            + prices["cached_input"] * u["cached_input_tokens"]
        ) / 1e6

        assert billed == pytest.approx(report["total_usd"])

    def test_token_weights_reported_with_their_defaults(self, profile_file):
        assert self.price(profile_file)["token_weights"] == {
            "output": 1.0,
            "recomputed_input": 0.3,
            "cached_input": 0.01,
        }
        report = self.price(profile_file, token_weights={"recomputed_input": 0.5})

        assert report["token_weights"] == {"output": 1.0, "recomputed_input": 0.5, "cached_input": 0.01}

    def test_no_tokens_no_prices(self, profile_file):
        [report] = price_run([make_stage(tokens=())], CONFIG, {"profile": profile_file, "region": "r1"})

        assert report["per_1M_tokens_by_kind_usd"] is None
        assert "per 1M    no tokens" in format_report(report)


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

    def test_profile_not_in_usd(self, tmp_path):
        path = tmp_path / "eur.json"
        path.write_text(json.dumps({**PROFILE, "currency": "EUR"}))

        with pytest.raises(ValueError, match="reports USD"):
            price_run([make_stage()], CONFIG, {"profile": str(path)})


class TestSessionCosts:
    def test_output_only_weights(self, profile_file):
        # The $45 busy split 40 : 60 by output tokens, the $15 idle 40 s : 60 s by time open.
        sessions = price_sessions(profile_file, token_weights={"recomputed_input": 0, "cached_input": 0})

        assert session_cost(sessions) == {"a": pytest.approx(18.0 + 6.0), "b": pytest.approx(27.0 + 9.0)}
        assert sessions["sessions"] == 2
        assert sessions["median_usd"] == pytest.approx(30.0)
        assert sessions["max_usd"] == pytest.approx(36.0)

    def test_token_weights_change_the_busy_split(self, profile_file):
        # A recomputed input token weighs half an output token: busy split 40 + 20 : 60 + 150.
        sessions = price_sessions(profile_file, token_weights={"recomputed_input": 0.5, "cached_input": 0})

        assert session_cost(sessions) == {"a": pytest.approx(10.0 + 6.0), "b": pytest.approx(35.0 + 9.0)}

    def test_unknown_weight_name_is_rejected(self, profile_file):
        with pytest.raises(ValueError, match="token_weights"):
            price_sessions(profile_file, token_weights={"input": 1.0})

    def test_time_at_the_llm_and_in_gaps(self, profile_file):
        # a: open 0-40 s, requests 0-10 and 20-40 -> 30 s at the LLM, a 10 s gap; b: one request over its 60 s.
        sessions = price_sessions(profile_file)

        assert sessions["per_session"]["a"] | {"cost_usd": None} == {
            "open_s": 40.0,
            "at_llm_s": 30.0,
            "gap_s": 10.0,
            "cost_usd": None,
        }
        assert sessions["per_session"]["b"]["gap_s"] == 0.0
        assert sessions["gap_pct"] == pytest.approx(10.0)  # 10 s of 100 s open
        # Means add up: 50 s open = 45 s at the LLM + 5 s in gaps.
        assert (sessions["mean_open_s"], sessions["mean_at_llm_s"], sessions["mean_gap_s"]) == (50.0, 45.0, 5.0)

    def test_overlapping_requests_count_once_at_the_llm(self):
        assert at_llm_seconds([(5.0, 20.0), (0.0, 10.0), (30.0, 40.0)]) == 30.0

    def test_billing_rounding_is_split_by_time_open(self, profile_file):
        # Hourly billing of a 1 h stage adds nothing; billing per 7200 s adds a $60 hour, split 40 : 60 like idle.
        sessions = price_sessions(
            profile_file, billing_increment_s=7200, token_weights={"recomputed_input": 0, "cached_input": 0}
        )

        assert session_cost(sessions) == {"a": pytest.approx(18.0 + 6.0 + 24.0), "b": pytest.approx(27.0 + 9.0 + 36.0)}

    def test_no_session_ids_means_no_split(self, profile_file):
        # Every workload but the OTel trace replay.
        [report] = price_run([make_stage()], CONFIG, {"profile": profile_file, "region": "r1"})

        assert report["sessions"] is None


class TestEndToEnd:
    def test_report_written_per_stage(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())

        assert report["total_usd"] > 0
        assert report["total_usd"] == pytest.approx(sum(report["breakdown_usd"].values()))
        assert report["pricing"]["instance_type"] == "tiny.1gpu"
        assert report["pricing"]["gpus_used"] == 1.0
        assert report["pricing"]["gpus_rented"] == 1.0
        assert report["sessions"] is None  # defaults.json replays no sessions

    def test_cost_lines_appended_to_simulation_log(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())
        log = (cost_run_dir / "simulation.log").read_text()

        assert "===== cost: stage_0 =====" in log
        assert "setup     1 x tiny.1gpu (1 x H100 each), r1, on_demand: 9.00 USD/h per instance" in log
        assert f"cost      {report['total_usd']:.4f} USD (" in log
        # Exact billing prints no billing line, and lines that cost nothing (unused GPUs, rounding) get no row.
        assert "billing" not in log.split("===== cost: stage_0")[1]
        assert "GPU unused" not in log
        busy = report["breakdown_usd"]["gpu_busy"]
        assert f"{100 * busy / report['total_usd']:.1f}%{busy:>12.4f}" in log  # the GPU busy row: share and USD
        # Stage 1 never starts in 10 s: it costs nothing but still names the 1-GPU instance.
        assert "cost      0.0000 USD (no requests)\nper 1M    no tokens" in log
        assert f"total     {report['total_usd']:.4f} USD over 2 stage(s)" in log

    def test_no_plots_without_the_graphs_flag(self, cost_run_dir):
        assert not list(cost_run_dir.glob("stage_*/cost-*.pdf"))

    def test_plots_written_next_to_the_report_with_the_graphs_flag(self, cost_run_dir):
        cost_main([str(cost_run_dir), "-g"])

        assert (cost_run_dir / "stage_0" / "cost-breakdown.pdf").stat().st_size > 0
        assert (cost_run_dir / "stage_0" / "cost-gpu-busy.pdf").stat().st_size > 0
        assert not (cost_run_dir / "stage_0" / "cost-sessions.pdf").exists()  # no sessions in defaults.json

    def test_offline_pricing_matches_the_run(self, cost_run_dir):
        report = json.loads((cost_run_dir / "stage_0" / COST_FILE).read_text())

        offline = cost_main([str(cost_run_dir), "--no-write"])

        assert offline[0]["total_usd"] == pytest.approx(report["total_usd"])

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
