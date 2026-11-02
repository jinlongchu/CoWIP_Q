"""Evaluate the five M4 checkpoints after convergence and assemble artifacts."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

import train_m4_until_converged as base


OUT = (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_final_converged').resolve()
CHECKPOINTS = {
    13101: (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged_extend/run_13101'),
    13102: (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged/run_13102'),
    13103: (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged/run_13103'),
    13104: (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged_finish_13104/run_13104'),
    13105: (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged/run_13105'),
}

# Correct the path above if the date folder is the current one.
CHECKPOINTS[13104] = (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged_finish_13104/run_13104')


class IndependentQEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.stack([model.agent_values(sequence, mask) for model in self.models]).mean(0)


def write_rows(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path):
    model = base.runner.CTDEModel(72)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def summarize(rows):
    result = {}
    for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key + '_mean'] = float(values.mean())
        result[key + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def evaluate(model, seeds):
    base.runner.trainer_model = model.eval()
    rows = []
    for seed in seeds:
        row = base.runner.run_episode('m4', seed, trainer=None, learn=False)
        row['seed'] = seed
        rows.append(row)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = []
    models = []
    for seed, source in CHECKPOINTS.items():
        manifest = json.loads((source.parent / 'training_manifest.json').read_text(encoding='utf-8')) if (source.parent / 'training_manifest.json').exists() else None
        if manifest:
            item = next(x for x in manifest if x['run_seed'] == seed)
        else:
            item = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
        item = dict(item)
        item['final_checkpoint_source'] = str((source / 'best_model.pt').resolve())
        manifests.append(item)
        target = OUT / f'run_{seed}'
        target.mkdir(parents=True, exist_ok=True)
        for name in ('best_model.pt', 'training.csv', 'learning_curve.csv', 'validation.csv'):
            src = source / name
            if src.exists():
                shutil.copy2(src, target / name)
        models.append(load_checkpoint(source / 'best_model.pt'))
    (OUT / 'training_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')

    all_rows = []
    for seed, model in zip(CHECKPOINTS, models):
        for split, seeds in [('test', base.FINAL_TEST_SEEDS), ('holdout', base.FINAL_HOLDOUT_SEEDS)]:
            for row in evaluate(model, seeds):
                row.update(method='m4_final_converged_individual', run_seed=seed, split=split)
                all_rows.append(row)
    ensemble = IndependentQEnsemble(models)
    for split, seeds in [('test', base.FINAL_TEST_SEEDS), ('holdout', base.FINAL_HOLDOUT_SEEDS)]:
        for row in evaluate(ensemble, seeds):
            row.update(method='m4_final_converged_q_ensemble', run_seed=-1, split=split)
            all_rows.append(row)
    fields = ['method', 'run_seed', 'split', 'seed', 'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points']
    write_rows(OUT / 'evaluation_per_run.csv', fields, all_rows)
    summary = []
    for method in ('m4_final_converged_individual', 'm4_final_converged_q_ensemble'):
        for split in ('test', 'holdout'):
            group = [row for row in all_rows if row['method'] == method and row['split'] == split]
            item = {'method': method, 'split': split, 'n': len(group)}
            item.update(summarize(group))
            summary.append(item)
    (OUT / 'comparison_final.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
