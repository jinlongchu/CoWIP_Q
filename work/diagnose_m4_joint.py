"""Evaluate centralized joint-action selection over the M4 Q ensemble."""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = (Path(__file__).resolve().parents[1] / 'work/conwip_four_strategy_v3_madqn')
sys.path.insert(0, str(REPO))
import run_madqn_ablation as runner
from conwip.baseline_algorithms import FrozenCoordinatedPI, evaluate_episode, new_config


class IndependentEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.stack([model.agent_values(sequence, mask) for model in self.models]).mean(0)


class JointEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        combos = torch.as_tensor(list(itertools.product(range(3), repeat=3)), dtype=torch.long)
        self.register_buffer('combos', combos)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # The M4 mixer was trained on the global state, which is broadcast in
        # every agent feature frame. Recover that state from the latest frame.
        state = sequence[:, :, -1, runner.LOCAL_DIM:runner.LOCAL_DIM + runner.GLOBAL_DIM].mean(1)
        combos = self.combos.to(sequence.device)
        batch, agents, actions = mask.shape
        num_combos = combos.shape[0]
        legal = torch.stack([
            mask[:, i, combos[:, i]] for i in range(runner.N_AGENTS)
        ], dim=-1).all(-1)
        totals = []
        action_batch = combos.unsqueeze(0).expand(batch, -1, -1)
        for model in self.models:
            q = model.agent_values(sequence, mask)
            q_selected = torch.gather(
                q.unsqueeze(1).expand(-1, num_combos, -1, -1),
                -1,
                action_batch.unsqueeze(-1),
            ).squeeze(-1)
            mix_state = state.unsqueeze(1).expand(-1, num_combos, -1)
            total = model.mixer(
                q_selected.reshape(batch * num_combos, agents),
                mix_state.reshape(batch * num_combos, runner.GLOBAL_DIM),
            ).reshape(batch, num_combos)
            totals.append(total)
        totals = torch.stack(totals).mean(0)
        totals = totals.masked_fill(~legal, -1e9)
        selected = combos[totals.argmax(-1)]
        output = torch.full_like(q, -1e9)
        output.scatter_(2, selected.unsqueeze(-1), 0.0)
        return output


def load_models(source: Path):
    models = []
    for checkpoint in sorted(source.glob('run_*/model.pt')):
        model = runner.CTDEModel(72)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        models.append(model)
    return models


def evaluate_policy(policy, seeds, method, split):
    runner.TEST_SEEDS = list(seeds)
    runner.trainer_model = policy.eval()
    rows = []
    for seed in seeds:
        result = runner.run_episode('m4', seed, trainer=None, learn=False)
        result.update(method=method, split=split, seed=seed)
        rows.append(result)
    return rows


def main():
    source = (Path(__file__).resolve().parents[1] / 'outputs/madqn_ablation/m4')
    output = (Path(__file__).resolve().parents[1] / 'outputs/m4_joint_ensemble_diagnostic')
    output.mkdir(parents=True, exist_ok=True)
    models = load_models(source)
    independent = IndependentEnsemble(models)
    joint = JointEnsemble(models)
    same_test = list(range(14101, 14113))
    unseen = list(range(15101, 15121))
    rows = []
    for seeds, split in ((same_test, 'same_test'), (unseen, 'unseen')):
        rows.extend(evaluate_policy(independent, seeds, 'q_ensemble_independent', split))
        rows.extend(evaluate_policy(joint, seeds, 'q_ensemble_joint', split))
        for row in rows:
            if 'run_seed' not in row:
                row['run_seed'] = -1
        if split == 'unseen':
            for seed in seeds:
                metrics, _ = evaluate_episode(FrozenCoordinatedPI(), seed, new_config())
                row = metrics.__dict__.copy()
                row.update(method='pi', split=split, seed=seed, run_seed=-1)
                rows.append(row)
    fields = ['method', 'run_seed', 'split', 'seed', 'reward_x1000', 'nrmse', 'mean_wip',
              'cap_adjustment_points']
    with (output / 'per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for split in ('same_test', 'unseen'):
        for method in ('q_ensemble_independent', 'q_ensemble_joint', 'pi'):
            group = [x for x in rows if x['split'] == split and x['method'] == method]
            if not group:
                continue
            item = {'split': split, 'method': method, 'n': len(group)}
            for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
                values = np.asarray([x[key] for x in group], dtype=float)
                item[key + '_mean'] = float(values.mean())
                item[key + '_sd'] = float(values.std(ddof=1))
            summary.append(item)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
