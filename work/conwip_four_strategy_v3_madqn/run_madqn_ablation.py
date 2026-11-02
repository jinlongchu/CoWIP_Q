"""Ablation study for the PairV2 MADQN controller.

M0 is the packaged baseline result.  This runner trains and evaluates:
  M1: global coordination features + recurrent local Q networks
  M2: M1 + monotonic QMIX-style centralized training
  M3: M2 + n-step returns + prioritized replay
  M4: M3 + current-PI-guided residual action prior
  M5: M4 + a Q-advantage trust region for PI residuals

The environment, reward, test seeds, and reporting metrics match the packaged
four-strategy experiment.  M4 is intentionally reported as a hybrid method.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from conwip.baseline_algorithms import FrozenCoordinatedPI, new_config
from conwip.pair_v2 import PairV2Env


N_AGENTS = 3
N_ACTIONS = 3
LOCAL_DIM = 45
GLOBAL_DIM = 24
SEQ_LEN = 4
BASE_GAMMA = 0.99
TEST_SEEDS = list(range(14101, 14113))
TRAIN_SEEDS = list(range(13101, 13106))
MODULES = ('m1', 'm2', 'm3', 'm4', 'm5')
PI_RESIDUAL_MODULES = ('m4', 'm5')
RESIDUAL_MARGIN = 0.01


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def global_context(local_obs: np.ndarray) -> np.ndarray:
    """Build a compact broadcast signal from observable local frames.

    It contains all three local WIP/caps, shared buffer/throughput context, and
    the recent utilization/blocking/idle/pair-shortage summaries.
    """
    local_obs = np.asarray(local_obs, dtype=np.float32)
    values = np.concatenate((
        local_obs[:, :2].reshape(-1),       # local WIP and cap
        local_obs[0, 2:8],                   # buffers and shared throughput context
        local_obs[:, 16:20].reshape(-1),    # recent utilization and shortage
    ))
    assert values.shape == (GLOBAL_DIM,)
    return values.astype(np.float32)


def feature_frame(local_obs: np.ndarray, pi_actions: np.ndarray | None = None):
    """Return per-agent M1 features and the centralized mixer state."""
    local_obs = np.asarray(local_obs, dtype=np.float32)
    state = global_context(local_obs)
    broadcast = np.repeat(state[None, :], N_AGENTS, axis=0)
    features = np.concatenate((local_obs, broadcast), axis=1)
    if pi_actions is not None:
        one_hot = np.eye(N_ACTIONS, dtype=np.float32)[np.asarray(pi_actions, dtype=int)]
        features = np.concatenate((features, one_hot), axis=1)
    return features.astype(np.float32), state


def legal_mask(local_obs: torch.Tensor) -> torch.Tensor:
    """Legal decrement/hold/increment actions for each cell."""
    caps = torch.round(local_obs[..., 1] * 8).long()
    return torch.stack((caps > 0, torch.ones_like(caps, dtype=torch.bool), caps < 8), -1)


def action_mask(sequence: torch.Tensor, module: str) -> torch.Tensor:
    """Mask direct actions, or residual actions around PI for M4."""
    local = sequence[..., :LOCAL_DIM]
    if module not in PI_RESIDUAL_MODULES:
        return legal_mask(local[:, :, -1])
    caps = torch.round(local[:, :, -1, 1] * 8).long()
    pi_action = sequence[:, :, -1, LOCAL_DIM + GLOBAL_DIM:LOCAL_DIM + GLOBAL_DIM + 3].argmax(-1)
    residual = torch.arange(3, device=sequence.device).view(1, 1, 3) - 1
    direct = pi_action.unsqueeze(-1) + residual
    return ((direct >= 0) & (direct <= 2) &
            ((direct != 0) | (caps.unsqueeze(-1) > 0)) &
            ((direct != 2) | (caps.unsqueeze(-1) < 8)))


def mlp(inputs: int, outputs: int, hidden: int = 64) -> nn.Sequential:
    net = nn.Sequential(
        nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs),
    )
    for layer in net:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(net[-1].weight, 0.01)
    return net


class RecurrentAgentQ(nn.Module):
    """Three cell-specific recurrent local Q heads."""

    def __init__(self, input_dim: int, hidden: int = 64, prefer_hold: bool = False):
        super().__init__()
        # Run all three agent histories in one batched GRU call.  The readout
        # heads remain cell-specific, preserving different A/B/assembly roles
        # without tripling recurrent-kernel overhead.
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.readouts = nn.ModuleList([mlp(hidden, N_ACTIONS, hidden) for _ in range(N_AGENTS)])
        if prefer_hold:
            for readout in self.readouts:
                with torch.no_grad():
                    # Zero residual is the PI fallback; start with a clear
                    # prior for it while allowing TD updates to override it.
                    readout[-1].bias[1] = 0.50

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # sequence: batch x agents x time x features
        batch, agents, steps, features = sequence.shape
        _, h = self.gru(sequence.reshape(batch * agents, steps, features))
        h = h[-1].reshape(batch, agents, -1)
        outputs = [self.readouts[i](h[:, i]) for i in range(N_AGENTS)]
        values = torch.stack(outputs, dim=1)
        if mask is not None:
            values = values.masked_fill(~mask, -1e9)
        return values


class QMix(nn.Module):
    """Small monotonic QMIX mixer conditioned on the global observable state."""

    def __init__(self, state_dim: int, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.w1 = mlp(state_dim, N_AGENTS * embed_dim, 64)
        self.b1 = mlp(state_dim, embed_dim, 64)
        self.w2 = mlp(state_dim, embed_dim, 64)
        self.b2 = mlp(state_dim, 1, 64)

    def forward(self, agent_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        w1 = self.w1(state).abs().view(-1, N_AGENTS, self.embed_dim)
        b1 = self.b1(state).view(-1, 1, self.embed_dim)
        hidden = torch.bmm(agent_q.unsqueeze(1), w1) + b1
        hidden = torch.nn.functional.elu(hidden)
        w2 = self.w2(state).abs().view(-1, self.embed_dim, 1)
        b2 = self.b2(state).view(-1, 1, 1)
        return (torch.bmm(hidden, w2) + b2).squeeze(-1).squeeze(-1)


class IndependentModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.agents = RecurrentAgentQ(input_dim, prefer_hold=input_dim == 72)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.agents(sequence, mask)


class CTDEModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.agents = RecurrentAgentQ(input_dim, prefer_hold=input_dim == 72)
        self.mixer = QMix(GLOBAL_DIM)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.agents(sequence, mask)

    def total_value(self, agent_q: torch.Tensor, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        selected = agent_q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        return self.mixer(selected, state)


class Replay:
    """Uniform or prioritized replay with a simple array-backed storage."""

    def __init__(self, capacity: int, prioritized: bool):
        self.capacity = capacity
        self.prioritized = prioritized
        self.data = []
        self.priorities = []
        self.position = 0

    def __len__(self):
        return len(self.data)

    def add(self, item, priority: float = 1.0):
        if len(self.data) < self.capacity:
            self.data.append(item)
            self.priorities.append(float(priority))
        else:
            self.data[self.position] = item
            self.priorities[self.position] = float(priority)
            self.position = (self.position + 1) % self.capacity

    def sample(self, rng: np.random.Generator, batch_size: int, beta: float):
        n = len(self.data)
        if self.prioritized:
            priorities = np.asarray(self.priorities, dtype=np.float64)
            probs = np.power(np.maximum(priorities, 1e-6), 0.6)
            probs /= probs.sum()
            indices = rng.choice(n, size=batch_size, replace=True, p=probs)
            weights = np.power(n * probs[indices], -beta)
            weights /= weights.max()
        else:
            indices = rng.integers(n, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
        return indices, [self.data[int(i)] for i in indices], weights.astype(np.float32)

    def update(self, indices, priorities):
        if not self.prioritized:
            return
        for index, priority in zip(indices, priorities):
            self.priorities[int(index)] = float(max(priority, 1e-6))


class Trainer:
    def __init__(self, module: str, seed: int, total_steps: int):
        self.module = module
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.total_steps = total_steps
        self.count = 0
        self.eps = 1.0
        self.nstep = 3 if module in ('m3', 'm4', 'm5') else 1
        self.prioritized = module in ('m3', 'm4', 'm5')
        self.learn_every = 4
        self.learn_updates = 0
        self.input_dim = 72 if module in PI_RESIDUAL_MODULES else 69
        self.model = IndependentModel(self.input_dim) if module == 'm1' else CTDEModel(self.input_dim)
        self.target = copy.deepcopy(self.model).eval()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay = Replay(100_000, self.prioritized)
        self.pending = deque()

    def _valid_actions(self, features: np.ndarray):
        caps = np.rint(features[:, 1] * 8).astype(int)
        if self.module not in PI_RESIDUAL_MODULES:
            return [[a for a, ok in enumerate((cap > 0, True, cap < 8)) if ok] for cap in caps]
        pi_actions = np.argmax(features[:, LOCAL_DIM + GLOBAL_DIM:LOCAL_DIM + GLOBAL_DIM + 3], axis=-1)
        valid = []
        for cap, pi_action in zip(caps, pi_actions):
            valid.append([r for r in range(3)
                          if 0 <= pi_action + r - 1 <= 2 and
                          (pi_action + r - 1 != 0 or cap > 0) and
                          (pi_action + r - 1 != 2 or cap < 8)])
        return valid

    def action(self, sequence: np.ndarray, state: np.ndarray, pi_actions=None, explore=True):
        valid = self._valid_actions(sequence[:, -1])
        if explore:
            self.eps = max(0.05, 1.0 - 0.95 * self.count / max(self.total_steps, 1))
            if self.module in PI_RESIDUAL_MODULES and pi_actions is not None:
                guidance = max(0.05, 0.50 * (1.0 - self.count / max(self.total_steps, 1)))
                if self.rng.random() < guidance:
                    # Residual code 1 means zero residual: follow PI exactly.
                    return np.ones(N_AGENTS, dtype=np.int64)
            if self.rng.random() < self.eps:
                return np.asarray([self.rng.choice(v) for v in valid], dtype=np.int64)
        with torch.no_grad():
            seq = torch.as_tensor(sequence[None], dtype=torch.float32)
            mask = action_mask(seq, self.module)
            values = self.model.agent_values(seq, mask)[0]
            chosen = values.argmax(-1)
            if self.module == 'm5':
                best_values, best_actions = values.max(-1)
                hold_values = values[:, 1]
                low_advantage = ((best_actions != 1) &
                                 ((best_values - hold_values) < RESIDUAL_MARGIN))
                chosen = torch.where(low_advantage, torch.ones_like(chosen), best_actions)
            return chosen.cpu().numpy().astype(np.int64)

    def _append_nstep(self, sequence, state, actions, reward, next_sequence, next_state, done):
        self.pending.append((sequence.copy(), state.copy(), actions.copy(), float(reward),
                             next_sequence.copy(), next_state.copy(), bool(done)))
        if not done and len(self.pending) < self.nstep:
            return
        self._commit_pending()
        if done:
            while self.pending:
                self._commit_pending()

    def _commit_pending(self):
        first = self.pending[0]
        total = 0.0
        discount = 1.0
        terminal = False
        next_sequence = first[4]
        next_state = first[5]
        steps = 0
        for item in self.pending:
            total += discount * item[3]
            steps += 1
            next_sequence, next_state, terminal = item[4], item[5], item[6]
            if terminal or steps >= self.nstep:
                break
            discount *= BASE_GAMMA
        item = (first[0], first[1], first[2], total, next_sequence, next_state,
                terminal, BASE_GAMMA ** steps)
        self.replay.add(item, priority=max(self.replay.priorities, default=1.0))
        self.pending.popleft()

    def add(self, sequence, state, actions, reward, next_sequence, next_state, done):
        self.count += 1
        self._append_nstep(sequence, state, actions, reward, next_sequence, next_state, done)
        if len(self.replay) < 1000 or self.count % self.learn_every:
            return
        self._learn()

    def _learn(self):
        beta = min(1.0, 0.4 + 0.6 * self.count / max(self.total_steps, 1))
        indices, batch, weights = self.replay.sample(self.rng, 64, beta)
        seq, state, actions, rewards, next_seq, next_state, done, discounts = zip(*batch)
        seq = torch.as_tensor(np.asarray(seq), dtype=torch.float32)
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        next_seq = torch.as_tensor(np.asarray(next_seq), dtype=torch.float32)
        next_state = torch.as_tensor(np.asarray(next_state), dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)
        discounts = torch.as_tensor(discounts, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)

        mask = action_mask(seq, self.module)
        next_mask = action_mask(next_seq, self.module)
        values = self.model.agent_values(seq, mask)
        selected = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        current_total = None
        if self.module != 'm1':
            current_total = self.model.mixer(selected, state)
        with torch.no_grad():
            next_online = self.model.agent_values(next_seq, next_mask)
            next_actions = next_online.argmax(-1)
            next_target = self.target.agent_values(next_seq, next_mask)
            next_selected = next_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            if self.module == 'm1':
                target = rewards[:, None] + discounts[:, None] * (1.0 - done[:, None]) * next_selected
            else:
                next_total = self.target.mixer(next_selected, next_state)
                target = rewards + discounts * (1.0 - done) * next_total
        if self.module == 'm1':
            td = selected - target
            loss_values = torch.nn.functional.smooth_l1_loss(selected, target, reduction='none').mean(-1)
        else:
            td = current_total - target
            loss_values = torch.nn.functional.smooth_l1_loss(current_total, target, reduction='none')
        loss = (loss_values * weights).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optim.step()
        if self.prioritized:
            self.replay.update(indices, np.abs(td.detach().cpu().numpy()) + 1e-5)
        self.learn_updates += 1
        if self.learn_updates % 250 == 0:
            self.target.load_state_dict(self.model.state_dict())


def pi_action_vector(pi: FrozenCoordinatedPI, env: PairV2Env) -> np.ndarray:
    actions = pi.act(env)
    return np.asarray([actions[a] for a in env.possible_agents], dtype=np.int64)


def run_episode(module: str, seed: int, trainer: Trainer | None = None, learn: bool = False):
    env = PairV2Env(new_config())
    obs, _ = env.reset(seed=seed)
    pi = FrozenCoordinatedPI() if module in PI_RESIDUAL_MODULES else None
    local = np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
    pi_actions = pi_action_vector(pi, env) if pi is not None else None
    features, state = feature_frame(local, pi_actions)
    history = deque([features.copy() for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
    rewards, tracking_costs = [], []
    changes = 0
    while env.agents:
        sequence = np.stack(history, axis=1).astype(np.float32)
        if trainer is None:
            with torch.no_grad():
                seq_tensor = torch.as_tensor(sequence[None], dtype=torch.float32)
                mask = action_mask(seq_tensor, module)
                chosen_actions = trainer_model.agent_values(seq_tensor, mask)[0].argmax(-1).cpu().numpy()
        else:
            chosen_actions = trainer.action(sequence, state, pi_actions, explore=learn)
        actions = chosen_actions.copy()
        if module in PI_RESIDUAL_MODULES:
            actions = np.clip(pi_actions + chosen_actions - 1, 0, 2).astype(np.int64)
        action_dict = dict(zip(env.agents, actions.tolist()))
        before = env.caps.copy()
        next_obs, reward, _, _, _ = env.step(action_dict)
        done = not bool(env.agents)
        next_local = np.stack([next_obs[a] for a in env.possible_agents]).astype(np.float32)
        next_pi_actions = pi_action_vector(pi, env) if (pi is not None and not done) else pi_actions
        next_features, next_state = feature_frame(next_local, next_pi_actions)
        next_history = deque(history, maxlen=SEQ_LEN)
        next_history.append(next_features.copy())
        next_sequence = np.stack(next_history, axis=1).astype(np.float32)
        if trainer is not None and learn:
            replay_actions = chosen_actions if module in PI_RESIDUAL_MODULES else actions
            trainer.add(sequence, state, replay_actions, float(reward['cell_0']),
                        next_sequence, next_state, done)
        history = next_history
        features, state, pi_actions = next_features, next_state, next_pi_actions
        rewards.append(float(reward['cell_0']))
        tracking_costs.append(float(env.last_reward_parts['tracking_cost']))
        changes += int(before != env.caps)
    metrics = {
        'reward_x1000': float(np.mean(rewards) * 1000.0),
        'nrmse': float(np.sqrt(np.mean(tracking_costs))),
        'mean_wip': float(env.metrics()['mean_wip_component_equiv']),
        'cap_adjustment_points': int(changes),
    }
    return metrics


# Set only during evaluation, keeping the episode code shared with training.
trainer_model: nn.Module | None = None


def train_job(job):
    module, run_seed, episodes, output = job
    seed_all(run_seed)
    root = Path(output) / module / f'run_{run_seed}'
    root.mkdir(parents=True, exist_ok=False)
    trainer = Trainer(module, run_seed, episodes * 360)
    rows = []
    start = time.monotonic()
    for episode in range(episodes):
        episode_seed = 31_000_000 + (run_seed - 13101) * 1000 + episode
        result = run_episode(module, episode_seed, trainer=trainer, learn=True)
        rows.append(result)
        if (episode + 1) % 8 == 0 or episode + 1 == episodes:
            record = {
                'episode': episode + 1,
                'train_nrmse': float(np.mean([x['nrmse'] for x in rows[-8:]])),
                'train_reward_x1000': float(np.mean([x['reward_x1000'] for x in rows[-8:]])),
                'epsilon': trainer.eps,
                'seconds': time.monotonic() - start,
            }
            with (root / 'training.csv').open('a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(record))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(record)
    torch.save(trainer.model.state_dict(), root / 'model.pt')
    return {'module': module, 'run_seed': run_seed, 'episodes': episodes,
            'checkpoint': str((root / 'model.pt').resolve()), 'training_seconds': time.monotonic() - start}


def evaluate_checkpoint(module: str, checkpoint: str, run_seed: int):
    global trainer_model
    input_dim = 72 if module in PI_RESIDUAL_MODULES else 69
    trainer_model = IndependentModel(input_dim) if module == 'm1' else CTDEModel(input_dim)
    trainer_model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    trainer_model.eval()
    rows = []
    for seed in TEST_SEEDS:
        result = run_episode(module, seed, trainer=None, learn=False)
        result.update(module=module, run_seed=run_seed, seed=seed)
        rows.append(result)
    return rows


def summarize(rows):
    result = []
    for module in sorted(set(r['module'] for r in rows)):
        group = [r for r in rows if r['module'] == module]
        entry = {
            'module': module,
            'n': len(group),
            'training_runs': len(set(r['run_seed'] for r in group)),
            'test_seeds': len(set(r['seed'] for r in group)),
        }
        for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
            values = np.asarray([r[key] for r in group], dtype=float)
            entry[key + '_mean'] = float(values.mean())
            entry[key + '_sd'] = float(values.std(ddof=1))
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--modules', nargs='+', choices=MODULES, default=list(MODULES))
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', default='outputs/madqn_ablation')
    args = parser.parse_args()
    if min(args.episodes, args.runs, args.workers) < 1:
        parser.error('episodes, runs, and workers must be positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(module, seed, args.episodes, str(output))
            for module in args.modules for seed in TRAIN_SEEDS[:args.runs]]
    manifests = []
    if args.workers == 1:
        for job in jobs:
            manifests.append(train_job(job))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            manifests.extend(pool.map(train_job, jobs))
    (output / 'training_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    all_rows = []
    for manifest in manifests:
        all_rows.extend(evaluate_checkpoint(manifest['module'], manifest['checkpoint'], manifest['run_seed']))
    with (output / 'evaluation_per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        fields = ['module', 'run_seed', 'seed', 'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = summarize(all_rows)
    (output / 'comparison_madqn_ablation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'comparison_madqn_ablation.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
