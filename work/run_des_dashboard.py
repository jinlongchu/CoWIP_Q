"""Local, read-only-model web interface for the original PairV2 DES.

The adapter records events without changing the simulator's event scheduling,
reward, decision interval, or trained policies. No model is trained here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import threading
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
sys.path.insert(0, str(WORK))
import run_controlled_m4_ablation as algorithms
from conwip.baseline_algorithms import FrozenCoordinatedPI, new_config
from conwip.pair_v2 import BASE_MEANS, PairV2Env

ASSETS = WORK / "des_dashboard"
MODEL_ROOT = ROOT / "outputs" / "controlled_m4_ablation" / "formal"
SEEDS = tuple(range(13101, 13106))
NAMES = ('A processing workshop', 'B processing workshop', 'Assembly workshop')
CONTROLLERS = {
    "m4": 'M4 independent Q ensemble · PI off',
    "madqn": 'Original MADQN · five-model Q ensemble',
    "pi": 'Coordinated PI',
    "manual": 'Manual cap control',
}


def integer(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"""{name} must be an integer from {lo} to {hi}""")
    if not math.isfinite(value) or int(value) != value or not lo <= value <= hi:
        raise ValueError(f"""{name} must be an integer from {lo} to {hi}""")
    return int(value)


class ObservedEnv(PairV2Env):
    """Record actual machine transitions; leave all DES behavior inherited."""

    def __init__(self, config):
        super().__init__(config)
        self.events = deque(maxlen=300)
        self.record_events = False
        self.event_number = 0

    def note(self, kind, message):
        if self.record_events:
            self.event_number += 1
            self.events.append({"id": self.event_number,
                                "minute": float(self.sim.now - self.stats_start),
                                "kind": kind, "message": message})

    def _slot(self, i, j, k, status, job=None):
        previous = self.status[i][j][k]
        old_job = self.active[i][j][k]
        super()._slot(i, j, k, status, job)
        if previous == 1 and status != 1 and old_job is not None:
            label = f"{('A', 'B', 'K')[i]}-{old_job['uid']:04d}"
            if i == 2 and j == 2:
                self.note("finished", f"""Finished kit {label} completed assembly and left the line""")
            else:
                self.note("transfer", f"""{label} completed {NAMES[i]} / stage {j + 1}""")
        if previous == 2 and status == 0 and old_job is not None:
            self.note("buffer", f"""{('A', 'B')[i]}-{old_job['uid']:04d} entered the matching buffer""")
        if status == 2:
            self.note("blocked", f"""{NAMES[i]} / stage {j + 1} blocked: downstream buffer is full""")

    def _start(self, i, j, k, job):
        super()._start(i, j, k, job)
        self.note("processing", f"""{('A', 'B', 'K')[i]}-{job['uid']:04d} started stage {j + 1} / machine {k + 1}""")


class ModelBank:
    def __init__(self):
        torch.set_num_threads(1)
        self.models = {}
        self.specs = {
            "m4": algorithms.Spec("no_pi", "M4 PI OFF", True, True, True, 3, True, False),
            "madqn": algorithms.Spec("original_madqn", "Original MADQN", False, False, False, 1, False, False, True),
        }

    def paths(self, controller):
        key = self.specs[controller].key
        return [MODEL_ROOT / key / f"run_{seed}" / "best_model.pt" for seed in SEEDS]

    def available(self, controller):
        return controller not in self.specs or all(p.is_file() for p in self.paths(controller))

    def get(self, controller):
        if controller not in self.models:
            if not self.available(controller):
                raise ValueError('The five best model files are incomplete. Select coordinated PI or manual control.')
            members = [algorithms.load_model(self.specs[controller], p) for p in self.paths(controller)]
            self.models[controller] = algorithms.QEnsemble(members).eval()
        return self.models[controller]


class Dashboard:
    def __init__(self):
        self.lock = threading.RLock()
        self.bank = ModelBank()
        self.reset({"seed": 30101, "controller": "m4" if self.bank.available("m4") else "pi"})

    def reset(self, payload):
        seed = integer(payload.get("seed", 30101), 'Random seed', 0, 2_000_000_000)
        controller = payload.get("controller", "m4")
        self.validate_controller(controller)
        schedule = payload.get("schedule", True)
        if not isinstance(schedule, bool):
            raise ValueError('The preset-disturbance flag must be Boolean')
        config = new_config()
        if not schedule:
            config = replace(config, schedule=())
        candidate = ObservedEnv(config)
        candidate.reset(seed=seed)
        self.env = candidate
        self.env.record_events = True
        self.controller = controller
        self.seed = seed
        self.pi = FrozenCoordinatedPI()
        self.manual_caps = list(config.initial_caps)
        self.history = []
        self.rewards = []
        self.tracking_costs = []
        self.adjustment_points = 0
        self.last_actions = [1, 1, 1]
        self.last_q = None
        self.intervened = False
        local = self.local()
        frame = algorithms.make_frame(local, np.ones(3, dtype=int), self.bank.specs["m4"])
        self.frames = deque([frame.copy() for _ in range(4)], maxlen=4)
        self.env.note("control", f"""Seed {seed} loaded; 720-minute warm-up complete; controller: {CONTROLLERS[controller]}""")
        self.record_point(initial=True)
        return self.snapshot()

    def validate_controller(self, controller):
        if controller not in CONTROLLERS:
            raise ValueError('Unknown control policy')
        if controller in self.bank.specs:
            self.bank.get(controller)

    def local(self):
        obs = self.env._observations()
        return np.stack([obs[a] for a in self.env.possible_agents]).astype(np.float32)

    def action(self):
        if self.controller == "manual":
            return np.asarray([1 + int(np.sign(goal - cap))
                               for goal, cap in zip(self.manual_caps, self.env.caps)], dtype=int)
        if self.controller == "pi":
            decisions = self.pi.act(self.env)
            return np.asarray([decisions[a] for a in self.env.possible_agents], dtype=int)
        local = self.local()
        frame = algorithms.make_frame(local, np.ones(3, dtype=int), self.bank.specs["m4"])
        self.frames[-1] = frame
        rep = np.stack(self.frames, axis=1) if self.controller == "m4" else local[:, None, :]
        with torch.inference_mode():
            x = torch.as_tensor(rep[None], dtype=torch.float32)
            mask = algorithms.mask_for(x[:, :, -1], self.bank.specs[self.controller])
            values = self.bank.get(self.controller).values(x, mask)[0]
            self.last_q = [[float(q) if legal else None for q, legal in zip(row, valid)]
                           for row, valid in zip(values.tolist(), mask[0].tolist())]
            return values.argmax(-1).cpu().numpy()

    def step(self, payload):
        count = integer(payload.get("count", 1), 'Step count', 1, 10)
        for _ in range(count):
            if not self.env.agents:
                break
            self.last_q = None
            actions = self.action()
            before = self.env.caps.copy()
            _, rewards, _, _, _ = self.env.step(dict(zip(self.env.possible_agents, actions.tolist())))
            self.last_actions = list(map(int, actions))
            self.rewards.append(float(rewards["cell_0"]))
            self.tracking_costs.append(float(self.env.last_reward_parts["tracking_cost"]))
            self.adjustment_points += int(before != self.env.caps)
            local = self.local()
            self.frames.append(algorithms.make_frame(local, np.ones(3, dtype=int), self.bank.specs["m4"]))
            changes = "/".join(('Decrease 1', 'Hold', 'Increase 1')[a] for a in self.last_actions)
            self.env.note("decision", f"""Joint action {changes}; caps {before} → {self.env.caps}""")
            self.record_point()
        return self.snapshot()

    def configure(self, payload):
        controller = payload.get("controller", self.controller)
        self.validate_controller(controller)
        caps = payload.get("manual_caps", self.manual_caps)
        if not isinstance(caps, list) or len(caps) != 3:
            raise ValueError('Provide three target caps')
        caps = [integer(c, 'Target cap', 0, 8) for c in caps]
        patch = {}
        if "target" in payload:
            patch["target"] = integer(payload["target"], 'Target throughput', 1, 20)
        if "gamma" in payload:
            shape = float(payload["gamma"])
            if shape not in (1., 2., 4., 9.):
                raise ValueError('Select a valid processing-variability level')
            patch["gamma"] = (shape, shape, self.env.current.gamma[2])
        if "b_slowdown_percent" in payload:
            slowdown = integer(payload["b_slowdown_percent"], 'B3 processing-time increase', 0, 100)
            means = [list(row) for row in self.env.current.means]
            means[1][2] = BASE_MEANS[1][2] * (1 + slowdown / 100)
            patch["means"] = tuple(tuple(row) for row in means)
        if patch:
            self.env.current.updated(patch)
        changed_controller = controller != self.controller
        if changed_controller:
            self.controller = controller
            self.pi = FrozenCoordinatedPI()
            self.last_q = None
            self.env.note("control", f"""Controller changed to {CONTROLLERS[controller]}""")
        self.manual_caps = caps
        if patch:
            self.env.set_conditions(source="dashboard", **patch)
            self.env.note("control", 'Production conditions changed manually; future preset disturbances retain their scheduled times')
        if patch or changed_controller or (self.controller == "manual" and caps != self.env.caps):
            self.intervened = True
        return self.snapshot()

    def record_point(self, initial=False):
        m = self.env.metrics()
        self.history.append({
            "minute": m["control_minutes"], "throughput": m["throughput_moving_kits_per_hour"],
            "target": m["target_kits_per_hour"], "reference": m["tracking_target_kits_per_hour"],
            "wip": m["total_wip_component_equiv"], "mean_wip": m["mean_wip_component_equiv"],
            "cap_a": self.env.caps[0], "cap_b": self.env.caps[1], "cap_assembly": self.env.caps[2],
            "buffer_a": len(self.env.buffers[0]), "buffer_b": len(self.env.buffers[1]),
            "reward_x1000": None if initial else self.rewards[-1] * 1000,
            "mean_reward_x1000": float(np.mean(self.rewards) * 1000) if self.rewards else None,
            "nrmse": float(np.sqrt(np.mean(self.tracking_costs))) if self.tracking_costs else None,
            "adjustment_points": self.adjustment_points, "controller": self.controller,
        })

    def snapshot(self):
        env = self.env
        metrics = env.metrics()
        cells = []
        for i in range(3):
            stages = []
            for j, machine_statuses in enumerate(env.status[i]):
                machines = []
                for k, status in enumerate(machine_statuses):
                    job = env.active[i][j][k]
                    progress = 0.
                    remaining = 0.
                    if job is not None:
                        started, duration, _, _, _ = job["service"][j]
                        progress = min(1., max(0., (env.sim.now - started) / duration))
                        remaining = max(0., started + duration - env.sim.now)
                    machines.append({"machine": k + 1, "status": ("idle", "busy", "blocked")[status],
                                     "job": None if job is None else f"{('A', 'B', 'K')[i]}-{job['uid']:04d}",
                                     "progress": progress, "remaining": remaining})
                stages.append({"number": j + 1, "mean_minutes": env.current.means[i][j],
                               "queue_count": len(env.queues[i][j]),
                               "queue_jobs": [f"{('A', 'B', 'K')[i]}-{job['uid']:04d}" for job in env.queues[i][j]],
                               "machines": machines})
            cells.append({"id": i, "name": NAMES[i], "wip": env.wip[i], "cap": env.caps[i],
                          "busy": env.recent["busy"][i], "blocked": env.recent["blocked"][i],
                          "idle": env.recent["idle"][i], "stages": stages})
        future = [{"minute": t, "target": patch.get("target"),
                   "kinds": [k for k in ("target", "gamma", "means") if k in patch]}
                  for t, patch in env.schedule[env.next_event:]]
        return {
            "application": "conwip-des-dashboard", "seed": self.seed,
            "controller": self.controller, "controller_label": CONTROLLERS[self.controller],
            "controllers": [{"id": k, "label": v, "available": self.bank.available(k)} for k, v in CONTROLLERS.items()],
            "minute": metrics["control_minutes"], "horizon": env.cfg.horizon_minutes,
            "decision_minutes": env.cfg.decision_minutes, "window_minutes": env.cfg.window_minutes,
            "warmup_minutes": env.cfg.warmup_minutes, "done": not bool(env.agents),
            "metrics": metrics, "cells": cells,
            "buffers": [{"name": f"""{('A', 'B')[i]} matching buffer""", "count": len(buffer),
                         "capacity": env.cfg.buffer_capacities[i],
                         "jobs": [f"{('A', 'B')[i]}-{job['uid']:04d}" for job in buffer]}
                        for i, buffer in enumerate(env.buffers)],
            "conditions": {"target": env.current.target, "gamma": list(env.current.gamma),
                           "b_slowdown_percent": round((env.current.means[1][2] / BASE_MEANS[1][2] - 1) * 100),
                           "schedule_enabled": bool(env.cfg.schedule)},
            "summary": {"reward_x1000": float(np.mean(self.rewards) * 1000) if self.rewards else None,
                        "nrmse": float(np.sqrt(np.mean(self.tracking_costs))) if self.tracking_costs else None,
                        "adjustment_points": self.adjustment_points, "steps": len(self.rewards),
                        "intervened": self.intervened},
            "last_actions": self.last_actions, "last_q": self.last_q,
            "reward_parts": env.last_reward_parts if self.rewards else None,
            "history": self.history, "events": list(reversed(env.events))[:45], "future_schedule": future,
        }

    def export_csv(self):
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(self.history[0]))
        writer.writeheader()
        writer.writerows(self.history)
        return ("\ufeff" + output.getvalue()).encode("utf-8")


def create_server(port=8767):
    dashboard = Dashboard()

    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, data, content_type, status=200, attachment=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{attachment}"')
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload, status=200):
            self.send_bytes(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
                            "application/json; charset=utf-8", status)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/api/state":
                with dashboard.lock:
                    self.send_json(dashboard.snapshot())
            elif route == "/api/export.csv":
                with dashboard.lock:
                    self.send_bytes(dashboard.export_csv(), "text/csv; charset=utf-8", attachment=f"des_seed_{dashboard.seed}.csv")
            elif route in ("/", "/style.css", "/app.js"):
                name = {"/": "index.html", "/style.css": "style.css", "/app.js": "app.js"}[route]
                mime = {"/": "text/html", "/style.css": "text/css", "/app.js": "text/javascript"}[route]
                self.send_bytes((ASSETS / name).read_bytes(), mime + "; charset=utf-8")
            elif route == "/favicon.ico":
                self.send_bytes(b"", "image/x-icon", 204)
            else:
                self.send_json({"error": 'Page not found'}, 404)

        def do_POST(self):
            route = urlparse(self.path).path
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError('Use a JSON request')
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 8192:
                    raise ValueError('Invalid request length')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Request must be a JSON object')
                handlers = {"/api/reset": dashboard.reset, "/api/step": dashboard.step,
                            "/api/configure": dashboard.configure}
                if route not in handlers:
                    self.send_json({"error": 'Endpoint not found'}, 404)
                    return
                with dashboard.lock:
                    self.send_json(handlers[route](payload))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.send_json({"error": f"""Simulation failed: {exc}"""}, 500)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.dashboard = dashboard
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"DES dashboard ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
