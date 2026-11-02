"""Evaluate an ensemble of the five M4 checkpoints."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import diagnose_m4_margin as diagnostic
import run_madqn_ablation as runner
from conwip.baseline_algorithms import FrozenCoordinatedPI, evaluate_episode, new_config


class EnsembleGate(nn.Module):
    def __init__(self, models, margin: float):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.margin = float(margin)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = torch.stack([model.agent_values(sequence, mask) for model in self.models]).mean(0)
        if self.margin > 0:
            values = values.clone()
            values[..., 0] -= self.margin
            values[..., 2] -= self.margin
        return values


def load_models(source: Path):
    models = []
    for checkpoint in sorted(source.glob('run_*/model.pt')):
        model = runner.CTDEModel(72)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        model.eval()
        models.append(model)
    return models


def eval_policy(models, margin, seeds, label):
    runner.TEST_SEEDS = list(seeds)
    runner.trainer_model = EnsembleGate(models, margin).eval()
    rows = []
    for seed in seeds:
        result = runner.run_episode('m4', seed, trainer=None, learn=False)
        result.update(method=label, seed=seed, margin=margin)
        rows.append(result)
    return rows


def main():
    source = (Path(__file__).resolve().parents[1] / 'outputs/madqn_ablation/m4')
    output = (Path(__file__).resolve().parents[1] / 'outputs/m4_ensemble_diagnostic')
    output.mkdir(parents=True, exist_ok=True)
    models = load_models(source)
    test_seeds = list(range(14101, 14113))
    unseen_seeds = list(range(15101, 15121))
    rows = []
    for seeds, split in ((test_seeds, 'same_test'), (unseen_seeds, 'unseen')):
        for margin, label in ((0.0, 'ensemble_raw'), (0.01, 'ensemble_margin_0.01')):
            for row in eval_policy(models, margin, seeds, label):
                row['split'] = split
                row['run_seed'] = -1
                rows.append(row)
        if split == 'unseen':
            for seed in seeds:
                metrics, _ = evaluate_episode(FrozenCoordinatedPI(), seed, new_config())
                row = metrics.__dict__.copy()
                row.update(method='pi', seed=seed, margin=None, split=split, run_seed=-1)
                rows.append(row)
    fields = ['method', 'run_seed', 'split', 'seed', 'margin', 'reward_x1000', 'nrmse',
              'mean_wip', 'cap_adjustment_points']
    with (output / 'per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for split in ('same_test', 'unseen'):
        for method in ('ensemble_raw', 'ensemble_margin_0.01', 'pi'):
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
