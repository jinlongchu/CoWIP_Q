"""Single-factor MADQN ablations.

S1: local MADQN + global coordination features + GRU
S2: local MADQN + 3-step returns + prioritized replay
S3: local MADQN + current-PI residual action prior
S13: S1 + S3, with no QMIX, n-step, or PER

No variant includes QMIX or another ablation module.  The same five training
seeds, 300 episodes, and twelve held-out test seeds are used for all variants.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from conwip.baseline_algorithms import FrozenCoordinatedPI, new_config
from conwip.pair_v2 import PairV2Env
from run_madqn_ablation import (
    BASE_GAMMA, GLOBAL_DIM, LOCAL_DIM, N_ACTIONS, N_AGENTS, Replay,
    RecurrentAgentQ, TEST_SEEDS, TRAIN_SEEDS, feature_frame, legal_mask,
    mlp, seed_all,
)


SEQ_LEN = 4
VARIANTS = ('s1', 's2', 's3', 's13')


def uses_recurrent(variant: str) -> bool:
    return variant in ('s1', 's13')


def uses_pi_residual(variant: str) -> bool:
    return variant in ('s3', 's13')


def pi_feature_offset(variant: str) -> int:
    return LOCAL_DIM + (GLOBAL_DIM if uses_recurrent(variant) else 0)


class FeedForwardAgentQ(nn.Module):
    def __init__(self, input_dim: int, prefer_hold: bool = False):
        super().__init__()
        self.heads = nn.ModuleList([mlp(input_dim, N_ACTIONS, 64) for _ in range(N_AGENTS)])
        if prefer_hold:
            for head in self.heads:
                with torch.no_grad():
                    head[-1].bias[1] = 0.50

    def forward(self, frame: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        values = torch.stack([head(frame[:, i]) for i, head in enumerate(self.heads)], dim=1)
        if mask is not None:
            values = values.masked_fill(~mask, -1e9)
        return values


def standalone_frame(local_obs: np.ndarray, pi_actions=None) -> np.ndarray:
    local_obs = np.asarray(local_obs, dtype=np.float32)
    if pi_actions is None:
        return local_obs
    one_hot = np.eye(N_ACTIONS, dtype=np.float32)[np.asarray(pi_actions, dtype=int)]
    return np.concatenate((local_obs, one_hot), axis=1).astype(np.float32)


def standalone_mask(frame: torch.Tensor, variant: str) -> torch.Tensor:
    local = frame[..., :LOCAL_DIM]
    if not uses_pi_residual(variant):
        return legal_mask(local)
    caps = torch.round(local[..., 1] * 8).long()
    offset = pi_feature_offset(variant)
    pi_action = frame[..., offset:offset + 3].argmax(-1)
    residual = torch.arange(3, device=frame.device).view(1, 1, 3) - 1
    direct = pi_action.unsqueeze(-1) + residual
    return ((direct >= 0) & (direct <= 2) &
            ((direct != 0) | (caps.unsqueeze(-1) > 0)) &
            ((direct != 2) | (caps.unsqueeze(-1) < 8)))


class StandaloneTrainer:
    def __init__(self, variant: str, seed: int, total_steps: int):
        self.variant = variant
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.total_steps = total_steps
        self.count = 0
        self.learn_updates = 0
        self.learn_every = 4
        self.nstep = 3 if variant == 's2' else 1
        self.prioritized = variant == 's2'
        self.recurrent = uses_recurrent(variant)
        self.pi_residual = uses_pi_residual(variant)
        input_dim = (LOCAL_DIM + (GLOBAL_DIM if self.recurrent else 0) +
                     (3 if self.pi_residual else 0))
        self.model = (RecurrentAgentQ(input_dim) if self.recurrent
                      else FeedForwardAgentQ(input_dim, prefer_hold=self.pi_residual))
        self.target = copy.deepcopy(self.model).eval()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay = Replay(100_000, self.prioritized)
        self.pending = deque()
        self.eps = 1.0

    def _valid_actions(self, frame: np.ndarray):
        caps = np.rint(frame[:, 1] * 8).astype(int)
        if not self.pi_residual:
            return [[a for a, ok in enumerate((cap > 0, True, cap < 8)) if ok] for cap in caps]
        offset = pi_feature_offset(self.variant)
        pi_actions = np.argmax(frame[:, offset:offset + 3], axis=-1)
        valid = []
        for cap, base in zip(caps, pi_actions):
            valid.append([r for r in range(3)
                          if 0 <= base + r - 1 <= 2 and
                          (base + r - 1 != 0 or cap > 0) and
                          (base + r - 1 != 2 or cap < 8)])
        return valid

    def _model_values(self, representation: np.ndarray):
        if self.recurrent:
            tensor = torch.as_tensor(representation[None], dtype=torch.float32)
            mask = legal_mask(tensor[:, :, -1])
        else:
            tensor = torch.as_tensor(representation[None], dtype=torch.float32)
            mask = standalone_mask(tensor, self.variant)
        return self.model(tensor, mask)[0]

    def action(self, representation: np.ndarray, pi_actions=None, explore=True):
        frame = representation[:, -1] if self.recurrent else representation
        valid = self._valid_actions(frame)
        if explore:
            self.eps = max(0.05, 1.0 - 0.95 * self.count / max(self.total_steps, 1))
            if self.pi_residual and pi_actions is not None:
                guidance = max(0.05, 0.50 * (1.0 - self.count / max(self.total_steps, 1)))
                if self.rng.random() < guidance:
                    return np.ones(N_AGENTS, dtype=np.int64)
            if self.rng.random() < self.eps:
                return np.asarray([self.rng.choice(v) for v in valid], dtype=np.int64)
        with torch.no_grad():
            return self._model_values(representation).argmax(-1).cpu().numpy().astype(np.int64)

    def _append_nstep(self, representation, actions, reward, next_representation, done):
        self.pending.append((representation.copy(), np.asarray(actions, dtype=np.int64).copy(),
                             float(reward), next_representation.copy(), bool(done)))
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
        next_representation = first[3]
        steps = 0
        for item in self.pending:
            total += discount * item[2]
            steps += 1
            next_representation, terminal = item[3], item[4]
            if terminal or steps >= self.nstep:
                break
            discount *= BASE_GAMMA
        item = (first[0], first[1], total, next_representation, terminal, BASE_GAMMA ** steps)
        self.replay.add(item, priority=max(self.replay.priorities, default=1.0))
        self.pending.popleft()

    def add(self, representation, actions, reward, next_representation, done):
        self.count += 1
        self._append_nstep(representation, actions, reward, next_representation, done)
        if len(self.replay) < 1000 or self.count % self.learn_every:
            return
        self._learn()

    def _learn(self):
        beta = min(1.0, 0.4 + 0.6 * self.count / max(self.total_steps, 1))
        indices, batch, weights = self.replay.sample(self.rng, 64, beta)
        representations, actions, rewards, next_representations, done, discounts = zip(*batch)
        reps = torch.as_tensor(np.asarray(representations), dtype=torch.float32)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)
        next_reps = torch.as_tensor(np.asarray(next_representations), dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        done = torch.as_tensor(done, dtype=torch.float32)
        discounts = torch.as_tensor(discounts, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)

        if self.recurrent:
            mask = legal_mask(reps[:, :, -1])
            next_mask = legal_mask(next_reps[:, :, -1])
        else:
            mask = standalone_mask(reps, self.variant)
            next_mask = standalone_mask(next_reps, self.variant)
        # In S3, action 1 is the zero-residual action and remains the legal
        # default whenever PI itself proposes a legal action.
        values = self.model(reps, mask)
        next_online = self.model(next_reps, next_mask)
        next_actions = next_online.argmax(-1)
        selected = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_target = self.target(next_reps, next_mask)
            next_selected = next_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target = rewards[:, None] + discounts[:, None] * (1.0 - done[:, None]) * next_selected
        td = selected - target
        loss_values = torch.nn.functional.smooth_l1_loss(selected, target, reduction='none').mean(-1)
        loss = (loss_values * weights).mean()
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optim.step()
        if self.prioritized:
            self.replay.update(indices, np.abs(td.detach().cpu().numpy()).mean(-1) + 1e-5)
        self.learn_updates += 1
        if self.learn_updates % 250 == 0:
            self.target.load_state_dict(self.model.state_dict())

def run_episode(variant: str, seed: int, trainer: StandaloneTrainer | None = None,
                learn: bool = False, model: nn.Module | None = None):
    env = PairV2Env(new_config())
    obs, _ = env.reset(seed=seed)
    pi = FrozenCoordinatedPI() if uses_pi_residual(variant) else None
    local = np.stack([obs[a] for a in env.possible_agents]).astype(np.float32)
    pi_actions = np.asarray([1, 1, 1], dtype=np.int64)
    if pi is not None:
        actions_dict = pi.act(env)
        pi_actions = np.asarray([actions_dict[a] for a in env.possible_agents], dtype=np.int64)
    if uses_recurrent(variant):
        features, _ = feature_frame(local)
        features = standalone_frame(features, pi_actions if uses_pi_residual(variant) else None)
        history = deque([features.copy() for _ in range(SEQ_LEN)], maxlen=SEQ_LEN)
        representation = np.stack(history, axis=1).astype(np.float32)
    else:
        features = standalone_frame(local, pi_actions if uses_pi_residual(variant) else None)
        representation = features.copy()
    rewards, costs = [], []
    changes = 0
    while env.agents:
        if trainer is not None:
            chosen = trainer.action(representation, pi_actions, explore=learn)
        else:
            with torch.no_grad():
                x = torch.as_tensor(representation[None], dtype=torch.float32)
                if uses_recurrent(variant):
                    mask = legal_mask(x[:, :, -1])
                else:
                    mask = standalone_mask(x, variant)
                chosen = model(x, mask)[0].argmax(-1).cpu().numpy().astype(np.int64)
        direct = chosen.copy()
        if uses_pi_residual(variant):
            direct = np.clip(pi_actions + chosen - 1, 0, 2).astype(np.int64)
        before = env.caps.copy()
        next_obs, reward, _, _, _ = env.step(dict(zip(env.agents, direct.tolist())))
        done = not bool(env.agents)
        next_local = np.stack([next_obs[a] for a in env.possible_agents]).astype(np.float32)
        next_pi_actions = pi_actions
        if pi is not None and not done:
            next_dict = pi.act(env)
            next_pi_actions = np.asarray([next_dict[a] for a in env.possible_agents], dtype=np.int64)
        if uses_recurrent(variant):
            next_features, _ = feature_frame(next_local)
            next_features = standalone_frame(next_features,
                                              next_pi_actions if uses_pi_residual(variant) else None)
            next_history = deque(history, maxlen=SEQ_LEN)
            next_history.append(next_features.copy())
            next_representation = np.stack(next_history, axis=1).astype(np.float32)
        else:
            next_representation = standalone_frame(next_local,
                                                   next_pi_actions if uses_pi_residual(variant) else None)
        if trainer is not None and learn:
            # Store the action selected in the model's action space, i.e. the
            # residual code for S3 and the direct action otherwise.
            trainer.add(representation, chosen, float(reward['cell_0']), next_representation, done)
        representation = next_representation
        if uses_recurrent(variant):
            history = next_history
        pi_actions = next_pi_actions
        rewards.append(float(reward['cell_0']))
        costs.append(float(env.last_reward_parts['tracking_cost']))
        changes += int(before != env.caps)
    return {
        'reward_x1000': float(np.mean(rewards) * 1000.0),
        'nrmse': float(np.sqrt(np.mean(costs))),
        'mean_wip': float(env.metrics()['mean_wip_component_equiv']),
        'cap_adjustment_points': int(changes),
    }


def train_job(job):
    variant, run_seed, episodes, output = job
    seed_all(run_seed)
    root = Path(output) / variant / f'run_{run_seed}'
    root.mkdir(parents=True, exist_ok=False)
    trainer = StandaloneTrainer(variant, run_seed, episodes * 360)
    rows = []
    start = time.monotonic()
    for episode in range(episodes):
        episode_seed = 31_000_000 + (run_seed - 13101) * 1000 + episode
        result = run_episode(variant, episode_seed, trainer=trainer, learn=True)
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
    return {'variant': variant, 'run_seed': run_seed, 'episodes': episodes,
            'checkpoint': str((root / 'model.pt').resolve())}


def evaluate_checkpoint(variant: str, checkpoint: str, run_seed: int):
    input_dim = (LOCAL_DIM + (GLOBAL_DIM if uses_recurrent(variant) else 0) +
                 (3 if uses_pi_residual(variant) else 0))
    model = (RecurrentAgentQ(input_dim) if uses_recurrent(variant)
             else FeedForwardAgentQ(input_dim, prefer_hold=uses_pi_residual(variant)))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    rows = []
    for seed in TEST_SEEDS:
        result = run_episode(variant, seed, model=model, learn=False)
        result.update(variant=variant, run_seed=run_seed, seed=seed)
        rows.append(result)
    return rows


def summarize(rows, variants=VARIANTS):
    summary = []
    for variant in variants:
        group = [row for row in rows if row['variant'] == variant]
        item = {'variant': variant, 'n': len(group),
                'training_runs': len(set(row['run_seed'] for row in group)),
                'test_seeds': len(set(row['seed'] for row in group))}
        for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
            values = np.asarray([row[key] for row in group], dtype=float)
            item[key + '_mean'] = float(values.mean())
            item[key + '_sd'] = float(values.std(ddof=1))
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--output', default='outputs/madqn_standalone_ablation')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    variants = tuple(args.variants)
    jobs = [(variant, seed, args.episodes, str(output))
            for variant in variants for seed in TRAIN_SEEDS[:args.runs]]
    if args.workers == 1:
        manifests = [train_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            manifests = list(pool.map(train_job, jobs))
    (output / 'training_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    rows = []
    for manifest in manifests:
        rows.extend(evaluate_checkpoint(manifest['variant'], manifest['checkpoint'], manifest['run_seed']))
    fields = ['variant', 'run_seed', 'seed', 'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points']
    with (output / 'evaluation_per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, variants)
    (output / 'comparison_standalone.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output / 'comparison_standalone.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
