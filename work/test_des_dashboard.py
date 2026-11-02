"""Check that the web adapter preserves the original simulation and policies."""
import csv
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

import run_des_dashboard as web
from conwip.baseline_algorithms import FrozenCoordinatedPI, evaluate_episode


@pytest.fixture(scope="module")
def dashboard():
    return web.Dashboard()


def test_manual_actions_and_machine_snapshots_match_original(dashboard):
    dashboard.reset({"seed": 30101, "controller": "manual"})
    dashboard.configure({"controller": "manual", "manual_caps": [4, 2, 1]})
    actual = dashboard.step({"count": 3})
    original = web.PairV2Env(web.new_config())
    original.reset(seed=30101)
    expected_rewards = []
    for _ in range(3):
        actions = [1 + int(np.sign(goal - cap)) for goal, cap in zip([4, 2, 1], original.caps)]
        _, rewards, _, _, _ = original.step(dict(zip(original.possible_agents, actions)))
        expected_rewards.append(rewards["cell_0"])
    assert actual["minute"] == 30
    assert actual["metrics"] == original.metrics()
    assert dashboard.rewards == expected_rewards
    assert len(actual["cells"]) == 3
    assert [len(cell["stages"]) for cell in actual["cells"]] == [5, 5, 3]
    assert sum(len(stage["machines"]) for cell in actual["cells"] for stage in cell["stages"]) == 16
    assert actual["events"][0]["kind"] == "decision"
    for cell in actual["cells"]:
        assert sum(stage["queue_count"] + sum(machine["job"] is not None for machine in stage["machines"])
                   for stage in cell["stages"]) == cell["wip"]


@pytest.mark.parametrize("controller", ["pi", "m4", "madqn"])
def test_full_episode_matches_existing_evaluation(dashboard, controller):
    dashboard.reset({"seed": 30101, "controller": controller})
    for _ in range(36):
        snapshot = dashboard.step({"count": 10})
    assert snapshot["done"]
    assert snapshot["minute"] == 3600
    assert snapshot["summary"]["steps"] == 360
    if controller == "pi":
        result, _ = evaluate_episode(FrozenCoordinatedPI(), 30101)
        expected = {"reward_x1000": result.reward_x1000, "nrmse": result.nrmse,
                    "mean_wip": result.mean_wip, "cap_adjustment_points": result.cap_adjustment_points}
    else:
        expected = web.algorithms.run_episode(dashboard.bank.specs[controller], 30101,
                                              model=dashboard.bank.get(controller))
    assert snapshot["summary"]["reward_x1000"] == pytest.approx(expected["reward_x1000"], abs=1e-10)
    assert snapshot["summary"]["nrmse"] == pytest.approx(expected["nrmse"], abs=1e-12)
    assert snapshot["metrics"]["mean_wip_component_equiv"] == pytest.approx(expected["mean_wip"], abs=1e-10)
    assert snapshot["summary"]["adjustment_points"] == expected["cap_adjustment_points"]
    assert dashboard.step({"count": 1})["summary"]["steps"] == 360
    rows = list(csv.DictReader(io.StringIO(dashboard.export_csv().decode("utf-8-sig"))))
    assert len(rows) == 361
    assert float(rows[-1]["mean_reward_x1000"]) == pytest.approx(expected["reward_x1000"])


def test_invalid_parameters_do_not_change_simulation(dashboard):
    dashboard.reset({"seed": 30101, "controller": "manual"})
    before = dashboard.snapshot()
    for payload in [{"target": True}, {"target": 0}, {"target": 2.5},
                    {"gamma": 0}, {"manual_caps": [3, 9, 2]},
                    {"controller": "unknown"}, {"b_slowdown_percent": -1}]:
        with pytest.raises(ValueError):
            dashboard.configure(payload)
        assert dashboard.snapshot() == before


def test_http_api_serves_assets_actions_and_csv():
    server = web.create_server(0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for asset, mime in [("/", "text/html"), ("/app.js", "text/javascript"), ("/style.css", "text/css")]:
            with urlopen(base + asset) as response:
                assert response.status == 200
                assert response.headers["Content-Type"].startswith(mime)
                assert len(response.read()) > 100
        request = Request(base + "/api/reset", data=json.dumps({"seed": 12, "controller": "manual", "schedule": False}).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert json.load(response)["future_schedule"] == []
        request = Request(base + "/api/step", data=b'{"count": 1}', headers={"Content-Type": "application/json"})
        with urlopen(request) as response:
            assert json.load(response)["minute"] == 10
        with urlopen(base + "/api/export.csv") as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert b"mean_reward_x1000" in response.read()
        request = Request(base + "/api/configure", data=b'{"target": -1}', headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 400
        with urlopen(base + "/api/state") as response:
            assert json.load(response)["conditions"]["target"] == 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
