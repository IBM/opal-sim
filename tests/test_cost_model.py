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


@pytest.fixture
def profile_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(PROFILE))
    return str(path)


def price(profile_file, workers=(3, 3), hw=HW, **cost_config):
    config = {"worker": {"hw": hw}, "simulation": {"num_workers": 3}}
    [report] = price_run([make_stage(workers=workers)], config, {"profile": profile_file, **cost_config})
    return report


def test_gpu_instances():
    options = gpu_instances(PROFILE, "h100")
    assert sorted((o["region"], o["instance_type"], o["price_per_gpu_hour"]) for o in options) == [
        ("r1", "big.8gpu", 10.0),
        ("r1", "small.2gpu", 10.0),
        ("r1", "tiny.1gpu", 9.0),
        ("r2", "big.8gpu", 8.0),
    ]
    assert len(gpu_instances(PROFILE, "H100", "spot")) == 2
    assert {o["instance_type"] for o in gpu_instances(PROFILE, "H100", instance_type="big.8gpu")} == {"big.8gpu"}
    with pytest.raises(ValueError, match="GPUs in this profile"):
        gpu_instances(PROFILE, "B200")
    with pytest.raises(ValueError, match="Unknown pricing model"):
        gpu_instances(PROFILE, "H100", pricing="reserved")


def test_stage_usage():
    usage = stage_usage(make_stage(), HW, 3)
    assert usage["provisioned_gpu_hours"] == pytest.approx(6.0)
    assert (usage["busy_gpu_hours"], usage["busy_share"]) == (pytest.approx(4.5), pytest.approx(0.75))
    assert (usage["input_tokens"], usage["output_tokens"], usage["finished_requests"]) == (400, 100, 2)
    # Busy time is capped at what was provisioned (a batch counts in the stage where it ends).
    capped = stage_usage(make_stage(busy=(9000.0, 9000.0)), HW, 3)
    assert (capped["busy_gpu_hours"], capped["busy_share"]) == (pytest.approx(6.0), 1.0)
    # Mean of the per-second worker samples; no samples -> the initial worker count.
    scaling = stage_usage(make_stage(workers=(1, 1, 2, 4)), HW, 1)
    assert scaling["provisioned_gpu_hours"] == pytest.approx(4.0)
    assert scaling["workers_time_share"] == {1: 0.5, 2: 0.25, 4: 0.25}
    assert stage_usage(make_stage(duration=0.5, workers=()), HW, 3)["mean_workers"] == 3.0
    # A stage that never ran (times left at -1) costs nothing.
    never_ran = stage_usage({**make_stage(), "stage_time_start": -1, "stage_time_end": -1}, HW, 3)
    assert (never_ran["provisioned_gpu_hours"], never_ran["busy_share"]) == (0.0, 0.0)


def test_whole_instances_are_billed(profile_file):
    # r1: 3 x small.2gpu = $60 beats 1 x big.8gpu = $80; tiny.1gpu can't hold a tp 2 worker.
    # The $60 of the workers' GPUs is 75% busy: $45 busy, $15 idle.
    report = price(profile_file, region="r1")
    assert report["pricing"]["instance_type"] == "small.2gpu"
    assert report["total"] == pytest.approx(60.0)
    assert report["breakdown"] == {
        "gpu_busy": pytest.approx(45.0),
        "gpu_idle": pytest.approx(15.0),
        "gpu_unused": pytest.approx(0.0),
    }
    assert (report["pricing"]["gpus_used"], report["pricing"]["gpus_rented"]) == (6.0, 6.0)
    assert report["per_request"] == pytest.approx(30.0)
    assert report["per_1M_tokens"] == pytest.approx(60.0 / 500 * 1e6)
    assert [(r["region"], r["cost"]) for r in report["by_region"]] == [("r1", 60.0), ("r2", 64.0)]
    # One 8-GPU instance for 6 GPUs: pay $80, of which 6 GPU-hours x $10 hold workers.
    big = price(profile_file, region="r1", instance_type="big.8gpu")
    assert big["total"] == pytest.approx(80.0)
    assert big["breakdown"] == {
        "gpu_busy": pytest.approx(45.0),
        "gpu_idle": pytest.approx(15.0),
        "gpu_unused": pytest.approx(20.0),
    }
    assert (big["pricing"]["gpus_used"], big["pricing"]["gpus_rented"]) == (6.0, 8.0)
    # Autoscaling: instances are counted per sample, ceil(5 / 4) = 2 -> 1.25 instances on average.
    scaled = price(profile_file, workers=(1, 1, 2, 5), region="r1", instance_type="big.8gpu")
    assert scaled["total"] == pytest.approx(100.0)
    assert (scaled["pricing"]["gpus_used"], scaled["pricing"]["gpus_rented"]) == (4.5, 10.0)


def test_region_tie_and_names(profile_file):
    # No region given: the cheapest one (r1) is used.
    assert price(profile_file)["pricing"]["region"] == "r1"
    # Spot in r1: 1 x big.8gpu = 3 x small.2gpu = $24; the tie goes to fewer instances.
    # Policy and pricing names are case-insensitive.
    spot = price(profile_file, policy="GPU_Hour_Rental", pricing="Spot")
    assert spot["pricing"]["pricing_model"] == "spot" and spot["pricing"]["instance_type"] == "big.8gpu"
    assert spot["total"] == pytest.approx(24.0)
    assert spot["breakdown"] == {
        "gpu_busy": pytest.approx(13.5),
        "gpu_idle": pytest.approx(4.5),
        "gpu_unused": pytest.approx(6.0),
    }
    # A zero-length stage costs 0 everywhere but is still ranked by cost per hour: 3 x $20 < 1 x $80.
    [empty] = price_run([make_stage(duration=0.0)], CONFIG, {"profile": profile_file, "region": "r1"})
    assert empty["total"] == 0.0 and empty["pricing"]["instance_type"] == "small.2gpu"


def test_price_run_errors(profile_file):
    with pytest.raises(ValueError, match="needs a 'profile'"):
        price_run([make_stage()], CONFIG, {})
    with pytest.raises(ValueError, match="Unknown cost policy"):
        price_run([make_stage()], CONFIG, {"profile": profile_file, "policy": "magic"})
    with pytest.raises(ValueError, match="Regions with a price"):
        price_run([make_stage()], CONFIG, {"profile": profile_file, "region": "r9"})
    with pytest.raises(ValueError, match="GPUs of a worker"):
        price(profile_file, hw={"gpu": "H100", "tp": 16})


def run_sim(config_path, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    config = OpalConfig()
    config.initialize(str(config_path))
    OpalSimulator.from_config(config=config, output_dir=str(tmp_path)).run(10)
    [run_dir] = tmp_path.glob("sim-*")
    return run_dir


def test_cost_written_at_end_of_run_and_offline_matches(tmp_path, monkeypatch, profile_file):
    # configs/defaults.json (1 H100 worker) + a cost section with the hand-made profile.
    config = json.loads((ROOT / "configs/defaults.json").read_text())
    config["cost"] = {"profile": profile_file, "region": "r1"}
    config_path = tmp_path / "cost_config.json"
    config_path.write_text(json.dumps(config))
    run_dir = run_sim(config_path, tmp_path, monkeypatch)
    report = json.loads((run_dir / "stage_0" / COST_FILE).read_text())
    assert report["total"] > 0
    assert report["total"] == pytest.approx(sum(report["breakdown"].values()))
    assert (report["pricing"]["gpus_used"], report["pricing"]["gpus_rented"]) == (1.0, 1.0)  # tiny.1gpu
    log = (run_dir / "simulation.log").read_text()
    busy = f"{report['usage']['busy_share']:.0%} busy"
    assert f"stage_0: {report['total']:.4f} USD | 1 x tiny.1gpu, r1, on_demand | 1 of 1 GPUs used, {busy}" in log
    # Stage 1 never starts in 10 s: it costs nothing but still names the 1-GPU instance.
    assert "stage_1: 0.0000 USD | 1 x tiny.1gpu, r1, on_demand | 1 of 1 GPUs used, 0% busy" in log
    assert cost_main([str(run_dir), "--no-write"])[0]["total"] == pytest.approx(report["total"])
    # One worker in a stage that starts at t = 0: the busy seconds add up to Opal's cumulative utilization x time.
    stats = json.loads((run_dir / "stage_0" / "opal_stats.json").read_text())
    utilization = stats["per_unit_gpu_utilization"]
    assert sum(stats["per_unit_gpu_busy_time"]) == pytest.approx(utilization[-1] / 100 * len(utilization))


def test_no_cost_section_means_no_cost_report(tmp_path, monkeypatch):
    run_dir = run_sim(ROOT / "configs/defaults.json", tmp_path, monkeypatch)
    assert not list(run_dir.glob(f"stage_*/{COST_FILE}"))
