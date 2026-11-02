"""Train a learned-dynamics MPC controller for the PairV2 environment.

The controller is deliberately separate from MADQN:
  1. collect transitions from M4, PI and a valid random policy;
  2. fit an ensemble of one-step predictors for state delta and reward;
  3. use beam-search MPC over the 27 joint cap actions;
  4. evaluate on the same test/holdout seeds and write a comparison table.

This is a compact, reproducible implementation rather than a claim that MPC
must beat M4. The predictive model is the trained component; the controller
is re-optimized online at every 10-minute decision.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "conwip_four_strategy_v3_madqn"
sys.path.insert(0, str(REPO))
import run_madqn_ablation as runner
from conwip.pair_v2 import PairV2Env
from conwip.baseline_algorithms import FrozenCoordinatedPI, new_config


OUT = (Path(__file__).resolve().parents[1] / 'outputs/mpc_model_predictive').resolve()
STATE_DIM = 3 * runner.LOCAL_DIM
ACTION_DIM = 9
INPUT_DIM = STATE_DIM + ACTION_DIM
TARGET_DIM = STATE_DIM + 1
TRAIN_SEEDS = list(range(17101, 17117))
PI_SEEDS = list(range(18101, 18117))
RANDOM_SEEDS = list(range(19101, 19117))
TEST_SEEDS = list(range(14101, 14113))
HOLDOUT_SEEDS = list(range(15101, 15121))
ACTION_VECTORS = np.asarray(
    [[a // 9, (a // 3) % 3, a % 3] for a in range(27)], dtype=np.int64
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def load_m4_ensemble() -> nn.Module:
    """Load the five converged M4 checkpoints already produced in this task."""
    source_paths = [
        OUT.parent / "madqn_m4_converged_extend" / "run_13101" / "best_model.pt",
        OUT.parent / "madqn_m4_converged" / "run_13102" / "best_model.pt",
        OUT.parent / "madqn_m4_converged" / "run_13103" / "best_model.pt",
        OUT.parent / "madqn_m4_converged_finish_13104" / "run_13104" / "best_model.pt",
        OUT.parent / "madqn_m4_converged" / "run_13105" / "best_model.pt",
    ]
    missing = [str(p) for p in source_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("M4 checkpoints missing: " + "; ".join(missing))
    models = []
    for path in source_paths:
        model = runner.CTDEModel(72)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        models.append(model.eval())
    class Ensemble(nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = nn.ModuleList(members)
        def agent_values(self, sequence, mask):
            return torch.stack([m.agent_values(sequence, mask) for m in self.members]).mean(0)
    return Ensemble(models).eval()


def pi_action_vector(pi: FrozenCoordinatedPI, env: PairV2Env) -> np.ndarray:
    actions = pi.act(env)
    return np.asarray([actions[a] for a in env.possible_agents], dtype=np.int64)


def m4_action(model: nn.Module, history: deque, pi_actions: np.ndarray) -> np.ndarray:
    sequence = np.stack(history, axis=1).astype(np.float32)
    with torch.no_grad():
        tensor = torch.as_tensor(sequence[None], dtype=torch.float32)
        mask = runner.action_mask(tensor, "m4")
        residual = model.agent_values(tensor, mask)[0].argmax(-1).cpu().numpy().astype(np.int64)
    return np.clip(pi_actions + residual - 1, 0, 2).astype(np.int64)


def valid_random_action(caps: list[int], rng: np.random.Generator) -> np.ndarray:
    result = []
    for cap in caps:
        valid = [a for a in (0, 1, 2) if (a != 0 or cap > 0) and (a != 2 or cap < 8)]
        result.append(int(rng.choice(valid)))
    return np.asarray(result, dtype=np.int64)


def collect_episode(policy: str, seed: int, m4: nn.Module | None, rows: list[dict]) -> None:
    env = PairV2Env(new_config())
    obs, _ = env.reset(seed=seed)
    local = np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
    pi = FrozenCoordinatedPI() if policy == "m4" else None
    pi_actions = pi_action_vector(pi, env) if pi is not None else None
    history = None
    if policy == "m4":
        features, _ = runner.feature_frame(local, pi_actions)
        history = deque([features.copy() for _ in range(runner.SEQ_LEN)], maxlen=runner.SEQ_LEN)
    rng = np.random.default_rng(seed + 900_000)
    while env.agents:
        state = local.reshape(-1).copy()
        if policy == "m4":
            actions = m4_action(m4, history, pi_actions)
        elif policy == "pi":
            actions = pi_action_vector(FrozenCoordinatedPI(), env)
        else:
            actions = valid_random_action(env.caps, rng)
        action_onehot = np.eye(3, dtype=np.float32)[actions].reshape(-1)
        next_obs, reward, _, _, _ = env.step(dict(zip(env.possible_agents, actions.tolist())))
        next_local = np.stack([next_obs[a] for a in env.possible_agents]).astype(np.float32)
        next_state = next_local.reshape(-1).copy()
        rows.append({
            "state": state,
            "action": action_onehot,
            "next_state": next_state,
            "reward": float(reward["cell_0"]),
            "policy": policy,
            "seed": seed,
        })
        if policy == "m4" and env.agents:
            next_pi = pi_action_vector(pi, env)
            next_features, _ = runner.feature_frame(next_local, next_pi)
            history.append(next_features.copy())
            pi_actions = next_pi
        local = next_local


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 192), nn.LayerNorm(192), nn.SiLU(),
            nn.Linear(192, 192), nn.SiLU(),
            nn.Linear(192, 128), nn.SiLU(),
            nn.Linear(128, TARGET_DIM),
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.net[-1].weight, 0.02)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        return self.net(x)


@dataclass
class ModelBundle:
    models: list[Predictor]
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def predict(self, states: np.ndarray, actions: np.ndarray):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        x = np.concatenate([states, actions], axis=1)
        xn = (x - self.x_mean) / self.x_std
        xt = torch.as_tensor(xn, dtype=torch.float32)
        outputs = []
        with torch.no_grad():
            for model in self.models:
                outputs.append(model(xt).cpu().numpy() * self.y_std + self.y_mean)
        values = np.stack(outputs, axis=0)
        deltas = values[:, :, :STATE_DIM]
        rewards = values[:, :, STATE_DIM]
        next_states = states[None, :, :] + deltas
        # Cap dynamics are exact and should not be left to a learned regressor.
        for agent in range(3):
            idx = agent * runner.LOCAL_DIM + 1
            next_states[:, :, idx] = np.clip(
                states[None, :, idx] + (2.0 * actions[:, agent * 3 + 2] + actions[:, agent * 3 + 1] - 1.0) / 8.0,
                0.0, 1.0,
            )
        next_states = np.nan_to_num(next_states, nan=0.0, posinf=2.0, neginf=-2.0)
        return next_states, rewards


def train_predictors(samples: list[dict], ensemble_size: int = 3, epochs: int = 35) -> tuple[ModelBundle, list[dict]]:
    states = np.asarray([x["state"] for x in samples], dtype=np.float32)
    actions = np.asarray([x["action"] for x in samples], dtype=np.float32)
    next_states = np.asarray([x["next_state"] for x in samples], dtype=np.float32)
    rewards = np.asarray([x["reward"] for x in samples], dtype=np.float32)[:, None]
    x = np.concatenate([states, actions], axis=1)
    y = np.concatenate([next_states - states, rewards], axis=1)
    # A seed-based split prevents adjacent transitions from the same episode
    # from filling both train and validation sets.
    seeds = np.asarray([x["seed"] for x in samples])
    valid = (seeds % 5) == 0
    train_idx, valid_idx = np.where(~valid)[0], np.where(valid)[0]
    x_mean, x_std = x[train_idx].mean(0), x[train_idx].std(0) + 1e-5
    y_mean, y_std = y[train_idx].mean(0), y[train_idx].std(0) + 1e-5
    xn = (x - x_mean) / x_std
    yn = (y - y_mean) / y_std
    xt = torch.as_tensor(xn, dtype=torch.float32)
    yt = torch.as_tensor(yn, dtype=torch.float32)
    records = []
    models = []
    for member in range(ensemble_size):
        seed_all(27000 + member)
        rng = np.random.default_rng(27000 + member)
        model = Predictor()
        optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        batch_size = 512
        best_state, best_loss = None, float("inf")
        for epoch in range(1, epochs + 1):
            order = rng.permutation(train_idx)
            model.train()
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                pred = model(xt[idx])
                loss = nn.functional.smooth_l1_loss(pred, yt[idx])
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(nn.functional.mse_loss(model(xt[valid_idx]), yt[valid_idx]).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                records.append({"member": member, "epoch": epoch, "validation_mse_normalized": val_loss})
        model.load_state_dict(best_state)
        models.append(model.eval())
    bundle = ModelBundle(models, x_mean, x_std, y_mean, y_std)
    return bundle, records


def action_is_legal(state: np.ndarray, action: np.ndarray) -> bool:
    caps = [int(np.rint(state[i * runner.LOCAL_DIM + 1] * 8)) for i in range(3)]
    return all((a != 0 or c > 0) and (a != 2 or c < 8) for c, a in zip(caps, action))


def mpc_action(bundle: ModelBundle, state: np.ndarray, anchor_actions: np.ndarray | None = None,
               horizon: int = 4, beam_width: int = 8, risk: float = 0.08,
               safety_margin: float = 0.02) -> np.ndarray:
    states = state[None].astype(np.float32)
    costs = np.zeros(1, dtype=np.float64)
    plans = np.zeros((1, 0, 3), dtype=np.int64)
    for step in range(horizon):
        parent_ids, action_rows = [], []
        for parent, parent_state in enumerate(states):
            for action in ACTION_VECTORS:
                if action_is_legal(parent_state, action) and (anchor_actions is None or np.all(np.abs(action - anchor_actions) <= 1)):
                    parent_ids.append(parent)
                    action_rows.append(action)
        if not action_rows:
            return np.ones(3, dtype=np.int64)
        parent_ids = np.asarray(parent_ids, dtype=np.int64)
        action_rows = np.asarray(action_rows, dtype=np.int64)
        expanded_states = states[parent_ids]
        action_onehot = np.eye(3, dtype=np.float32)[action_rows].reshape(-1, ACTION_DIM)
        predicted_states, predicted_rewards = bundle.predict(expanded_states, action_onehot)
        mean_reward = predicted_rewards.mean(0)
        reward_uncertainty = predicted_rewards.std(0)
        # Reward is already the environment objective. The small uncertainty
        # term makes the planner avoid model-extrapolation corner cases.
        step_cost = -mean_reward + risk * reward_uncertainty
        if anchor_actions is not None:
            step_cost += 0.04 * np.mean(np.abs(action_rows - anchor_actions), axis=1)
        step_cost += 0.015 * np.mean(np.abs(action_rows - 1), axis=1)
        total_cost = costs[parent_ids] + step_cost
        order = np.argsort(total_cost)[:beam_width]
        states = predicted_states.mean(0)[order]
        costs = total_cost[order]
        plans = np.concatenate([plans[parent_ids][order], action_rows[order, None, :]], axis=1)
    best_index = int(np.argmin(costs))
    if anchor_actions is not None:
        # Compare with a short-horizon anchor rollout. The learned planner is
        # allowed to deviate only when it predicts a meaningful improvement;
        # otherwise the known-safe PI action is retained.
        anchor_state = state[None].astype(np.float32)
        anchor_onehot = np.eye(3, dtype=np.float32)[anchor_actions][None].reshape(1, ACTION_DIM)
        anchor_cost = 0.0
        for _ in range(horizon):
            anchor_next, anchor_rewards = bundle.predict(anchor_state, anchor_onehot)
            anchor_cost += float(-anchor_rewards.mean(0)[0] + risk * anchor_rewards.std(0)[0])
            anchor_cost += 0.015 * float(np.mean(np.abs(anchor_actions - 1)))
            anchor_state = anchor_next.mean(0)
        if float(costs[best_index]) >= anchor_cost - safety_margin:
            return anchor_actions.astype(np.int64)
    return plans[best_index, 0]


def run_mpc_episode(bundle: ModelBundle, seed: int, horizon: int = 4) -> dict:
    env = PairV2Env(new_config())
    obs, _ = env.reset(seed=seed)
    local = np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
    pi = FrozenCoordinatedPI()
    rewards, tracking, changes = [], [], 0
    while env.agents:
        state = local.reshape(-1)
        anchor = pi_action_vector(pi, env)
        action = mpc_action(bundle, state, anchor_actions=anchor, horizon=horizon)
        before = env.caps.copy()
        next_obs, reward, _, _, _ = env.step(dict(zip(env.possible_agents, action.tolist())))
        local = np.stack([next_obs[a] for a in env.possible_agents]).astype(np.float32)
        rewards.append(float(reward["cell_0"]))
        tracking.append(float(env.last_reward_parts["tracking_cost"]))
        changes += int(before != env.caps)
    return {
        "reward_x1000": float(np.mean(rewards) * 1000.0),
        "nrmse": float(np.sqrt(np.mean(tracking))),
        "mean_wip": float(env.metrics()["mean_wip_component_equiv"]),
        "cap_adjustment_points": int(changes),
    }


def summarize(rows: list[dict]) -> dict:
    result = {"n": len(rows)}
    for key in ("reward_x1000", "nrmse", "mean_wip", "cap_adjustment_points"):
        values = np.asarray([x[key] for x in rows], dtype=float)
        result[key + "_mean"] = float(values.mean())
        result[key + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def write_csv(path: Path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def load_predictor_bundle(path: Path) -> ModelBundle:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    models = []
    for state_dict in payload["models"]:
        model = Predictor()
        model.load_state_dict(state_dict)
        models.append(model.eval())
    return ModelBundle(models, payload["x_mean"], payload["x_std"], payload["y_mean"], payload["y_std"])


def write_comparison(summary: dict):
    source = OUT.parent / "madqn_m4_final_converged" / "comparison_four_madqn_with_m4_final.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    rows.append({
        "method": "mpc_learned_ensemble_beam",
        "display": 'Learned-model MPC (3-model ensemble + PI safety region + beam search)',
        "n": str(summary["test"]["n"]), "training_runs": "3",
        "reward_mean": str(summary["test"]["reward_x1000_mean"]),
        "reward_sd": str(summary["test"]["reward_x1000_sd"]),
        "nrmse_mean": str(summary["test"]["nrmse_mean"]),
        "nrmse_sd": str(summary["test"]["nrmse_sd"]),
        "mean_wip": str(summary["test"]["mean_wip_mean"]),
        "mean_wip_sd": str(summary["test"]["mean_wip_sd"] if "mean_wip_sd" in summary["test"] else 0.0),
        "cap_adjustment_mean": str(summary["test"]["cap_adjustment_points_mean"]),
        "cap_adjustment_sd": str(summary["test"]["cap_adjustment_points_sd"]),
        "environment": 'PairV2 environment (3,3,2), test',
    })
    out = OUT / "comparison_four_madqn_with_mpc.csv"
    fields = list(rows[0])
    write_csv(out, fields, rows)
    md = ['# Four-Strategy Comparison (Learned-Model MPC Added)', "", '| Method | Test samples | Trained models | reward×1000 | NRMSE | Mean WIP | Cap-change count |', "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        md.append(f"| {row['display']} | {row['n']} | {row['training_runs']} | {float(row['reward_mean']):.4f} ± {float(row['reward_sd']):.4f} | {float(row['nrmse_mean']):.5f} ± {float(row['nrmse_sd']):.5f} | {float(row['mean_wip']):.4f} | {float(row['cap_adjustment_mean']):.2f} |")
    md += ["", 'Reward is higher-is-better (closer to 0); NRMSE is lower-is-better. MPC trains prediction models rather than a direct policy network. At each decision, it replans over 4 steps using beam width 8 within a safety region around the current PI action.']
    (OUT / "comparison_four_madqn_with_mpc.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main():
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    seed_all(26001)
    m4 = load_m4_ensemble()
    checkpoint = OUT / "mpc_predictor_ensemble.pt"
    if checkpoint.exists():
        bundle = load_predictor_bundle(checkpoint)
        meta = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
        samples = [None] * int(meta["samples"])
    else:
        samples = []
        for policy, seeds in (("m4", TRAIN_SEEDS), ("pi", PI_SEEDS), ("random", RANDOM_SEEDS)):
            for seed in seeds:
                collect_episode(policy, seed, m4, samples)
        meta = {"samples": len(samples), "policies": {p: sum(x["policy"] == p for x in samples) for p in ("m4", "pi", "random")}}
        (OUT / "dataset_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        bundle, curve = train_predictors(samples)
        with (OUT / "predictor_training.csv").open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["member", "epoch", "validation_mse_normalized"])
            writer.writeheader(); writer.writerows(curve)
        torch.save({"models": [m.state_dict() for m in bundle.models], "x_mean": bundle.x_mean, "x_std": bundle.x_std, "y_mean": bundle.y_mean, "y_std": bundle.y_std}, checkpoint)
    rows = []
    summary = {}
    for split, seeds in (("test", TEST_SEEDS), ("holdout", HOLDOUT_SEEDS)):
        split_rows = []
        for seed in seeds:
            item = run_mpc_episode(bundle, seed)
            item.update(split=split, seed=seed, method="mpc_learned_ensemble_beam")
            rows.append(item); split_rows.append(item)
        values = summarize(split_rows)
        for key in ("reward_x1000", "nrmse", "mean_wip", "cap_adjustment_points"):
            values[key + "_sd"] = float(np.std([x[key] for x in split_rows], ddof=1)) if len(split_rows) > 1 else 0.0
        summary[split] = values
    write_csv(OUT / "evaluation_per_run.csv", ["method", "split", "seed", "reward_x1000", "nrmse", "mean_wip", "cap_adjustment_points"], rows)
    (OUT / "comparison_mpc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_comparison(summary)
    report = [
        '# Learned-Model MPC Training and Control Results', "",
        '## Design', "",
        '- Collect transitions from M4, PI and legal random policies so the predictor sees multiple policies.',
        '- Train 3 independent one-step networks to predict the next 135-dimensional observation increment and immediate reward.',
        '- Use a 4-step receding horizon and beam width 8 online. Anchor candidates to current PI actions, allow only adjacent action offsets at each station, and replan next step.',
        '- Enforce deterministic cap changes in predicted states and use ensemble disagreement as a light risk penalty.', "",
        '## Results', "",
        '| split | reward×1000 | NRMSE | Mean WIP | Cap-change count |',
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("test", "holdout"):
        x = summary[split]
        report.append(f"| {split} | {x['reward_x1000_mean']:.4f} ± {x['reward_x1000_sd']:.4f} | {x['nrmse_mean']:.5f} ± {x['nrmse_sd']:.5f} | {x['mean_wip_mean']:.4f} ± {x['mean_wip_sd']:.4f} | {x['cap_adjustment_points_mean']:.2f} ± {x['cap_adjustment_points_sd']:.2f} |")
    report += ["", 'MPC is not required to beat M4. This experiment checks whether it can serve as an independent, interpretable predictive-control baseline. See `evaluation_per_run.csv`, `mpc_predictor_ensemble.pt` and `comparison_four_madqn_with_mpc.csv`.', "", f"""Elapsed time: {time.monotonic() - start:.1f} seconds."""]
    (OUT / 'MPC_Training_and_Control_Report.md').write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "samples": len(samples), "seconds": time.monotonic() - start}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
