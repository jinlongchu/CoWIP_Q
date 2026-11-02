"""Train M4 until validation reward stops improving, then evaluate its Q ensemble."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_DIR = Path(__file__).resolve().parent / 'conwip_four_strategy_v3_madqn'
sys.path.insert(0, str(REPO_DIR))
import run_madqn_ablation as runner


MIN_EPISODES = 300
MAX_EPISODES = 800
EVAL_INTERVAL = 50
PATIENCE = 4
MIN_IMPROVEMENT = 0.05
VALIDATION_SEEDS = list(range(16101, 16113))
FINAL_TEST_SEEDS = list(range(14101, 14113))
FINAL_HOLDOUT_SEEDS = list(range(15101, 15121))


class IndependentQEnsemble(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.stack([model.agent_values(sequence, mask) for model in self.models]).mean(0)


def write_rows(path: Path, fields, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(model: nn.Module, seeds):
    runner.trainer_model = model.eval()
    rows = []
    for seed in seeds:
        result = runner.run_episode('m4', seed, trainer=None, learn=False)
        result['seed'] = seed
        rows.append(result)
    return rows


def summarize(rows):
    result = {}
    for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key + '_mean'] = float(values.mean())
        result[key + '_sd'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def train_job(job):
    run_seed, output, max_episodes = job
    runner.seed_all(run_seed)
    root = Path(output) / f'run_{run_seed}'
    root.mkdir(parents=True, exist_ok=False)
    # Keep the original M4 exploration schedule: epsilon reaches its floor by
    # the original 300-episode budget even when convergence needs longer.
    trainer = runner.Trainer('m4', run_seed, MIN_EPISODES * 360)
    train_rows = []
    curve_rows = []
    validation_rows = []
    best_reward = -float('inf')
    best_nrmse = float('inf')
    best_episode = 0
    no_improvement = 0
    converged = False
    start = time.monotonic()
    for episode in range(max_episodes):
        episode_seed = 31_000_000 + (run_seed - 13101) * 1000 + episode
        result = runner.run_episode('m4', episode_seed, trainer=trainer, learn=True)
        result['episode'] = episode + 1
        train_rows.append(result)
        if (episode + 1) % 8 == 0 or episode + 1 == max_episodes:
            train_tail = train_rows[-8:]
            curve_rows.append({
                'episode': episode + 1,
                'train_nrmse': float(np.mean([x['nrmse'] for x in train_tail])),
                'train_reward_x1000': float(np.mean([x['reward_x1000'] for x in train_tail])),
                'epsilon': trainer.eps,
                'seconds': time.monotonic() - start,
            })
        if (episode + 1) % EVAL_INTERVAL != 0 and episode + 1 != max_episodes:
            continue
        train_tail = train_rows[-8:]
        train_record = {
            'episode': episode + 1,
            'train_nrmse': float(np.mean([x['nrmse'] for x in train_tail])),
            'train_reward_x1000': float(np.mean([x['reward_x1000'] for x in train_tail])),
            'epsilon': trainer.eps,
            'seconds': time.monotonic() - start,
        }
        val_model = trainer.model.eval()
        validation = evaluate_model(val_model, VALIDATION_SEEDS)
        trainer.model.train()
        val_summary = summarize(validation)
        val_record = {
            'episode': episode + 1,
            'validation_reward_x1000': val_summary['reward_x1000_mean'],
            'validation_reward_sd': val_summary['reward_x1000_sd'],
            'validation_nrmse': val_summary['nrmse_mean'],
            'validation_nrmse_sd': val_summary['nrmse_sd'],
            'validation_mean_wip': val_summary['mean_wip_mean'],
            'validation_cap_adjustment_points': val_summary['cap_adjustment_points_mean'],
            'epsilon': trainer.eps,
            'seconds': time.monotonic() - start,
        }
        validation_rows.append(val_record)
        improved = val_record['validation_reward_x1000'] > best_reward + MIN_IMPROVEMENT
        tie_improved = (abs(val_record['validation_reward_x1000'] - best_reward) <= MIN_IMPROVEMENT and
                        val_record['validation_nrmse'] < best_nrmse - 0.0002)
        if improved or tie_improved:
            best_reward = val_record['validation_reward_x1000']
            best_nrmse = val_record['validation_nrmse']
            best_episode = episode + 1
            no_improvement = 0
            torch.save(trainer.model.state_dict(), root / 'best_model.pt')
        else:
            no_improvement += 1
        if episode + 1 >= MIN_EPISODES and no_improvement >= PATIENCE:
            converged = True
            break
    torch.save(trainer.model.state_dict(), root / 'last_model.pt')
    write_rows(root / 'training.csv',
               ['episode', 'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'],
               train_rows)
    write_rows(root / 'learning_curve.csv',
               ['episode', 'train_nrmse', 'train_reward_x1000', 'epsilon', 'seconds'],
               curve_rows)
    write_rows(root / 'validation.csv', list(validation_rows[0]) if validation_rows else [], validation_rows)
    return {
        'run_seed': run_seed,
        'episodes_completed': len(train_rows),
        'converged': converged,
        'best_episode': best_episode,
        'best_validation_reward_x1000': best_reward,
        'best_validation_nrmse': best_nrmse,
        'checkpoint': str((root / 'best_model.pt').resolve()),
    }


def load_checkpoint(path):
    model = runner.CTDEModel(72)
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-episodes', type=int, default=MAX_EPISODES)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--output', default='outputs/madqn_m4_converged')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, str(output), args.max_episodes) for seed in runner.TRAIN_SEEDS[:args.runs]]
    if args.workers == 1:
        manifests = [train_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            manifests = list(pool.map(train_job, jobs))
    (output / 'training_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')

    models = [load_checkpoint(item['checkpoint']) for item in manifests]
    individual_rows = []
    for item, model in zip(manifests, models):
        for split, seeds in [('test', FINAL_TEST_SEEDS), ('holdout', FINAL_HOLDOUT_SEEDS)]:
            for row in evaluate_model(model, seeds):
                row.update(method='m4_converged_individual', run_seed=item['run_seed'], split=split)
                individual_rows.append(row)
    ensemble = IndependentQEnsemble(models)
    ensemble_rows = []
    for split, seeds in [('test', FINAL_TEST_SEEDS), ('holdout', FINAL_HOLDOUT_SEEDS)]:
        for row in evaluate_model(ensemble, seeds):
            row.update(method='m4_converged_q_ensemble', run_seed=-1, split=split)
            ensemble_rows.append(row)
    eval_rows = individual_rows + ensemble_rows
    fields = ['method', 'run_seed', 'split', 'seed', 'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points']
    write_rows(output / 'evaluation_per_run.csv', fields, eval_rows)
    summary = []
    for method in ('m4_converged_individual', 'm4_converged_q_ensemble'):
        for split in ('test', 'holdout'):
            group = [x for x in eval_rows if x['method'] == method and x['split'] == split]
            if group:
                item = {'method': method, 'split': split, 'n': len(group)}
                item.update(summarize(group))
                summary.append(item)
    (output / 'comparison_converged.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'manifests': manifests, 'summary': summary}, indent=2), flush=True)


if __name__ == '__main__':
    main()
