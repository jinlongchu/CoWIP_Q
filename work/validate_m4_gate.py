"""Validate the M4 Q-margin gate on unseen environment seeds."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import diagnose_m4_margin as diagnostic
import run_madqn_ablation as runner
from conwip.baseline_algorithms import FrozenCoordinatedPI, evaluate_episode, new_config


def main():
    source = (Path(__file__).resolve().parents[1] / 'outputs/madqn_ablation/m4')
    output = (Path(__file__).resolve().parents[1] / 'outputs/m4_gate_validation')
    output.mkdir(parents=True, exist_ok=True)
    runner.TEST_SEEDS = list(range(15101, 15121))
    rows = []
    for checkpoint in sorted(source.glob('run_*/model.pt')):
        for margin, label in ((0.0, 'm4_raw'), (0.01, 'm4_margin_0.01')):
            for row in diagnostic.evaluate(str(checkpoint), margin, 0.0):
                row['method'] = label
                row['run_seed'] = int(Path(checkpoint).parent.name.split('_')[1])
                rows.append(row)
    for seed in runner.TEST_SEEDS:
        metrics, _ = evaluate_episode(FrozenCoordinatedPI(), seed, new_config())
        row = metrics.__dict__.copy()
        row.update(method='pi', run_seed=-1, checkpoint='', margin=None,
                   error_threshold=None)
        rows.append(row)
    fields = ['method', 'run_seed', 'checkpoint', 'seed', 'margin', 'error_threshold',
              'reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points']
    with (output / 'per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for method in ('m4_raw', 'm4_margin_0.01', 'pi'):
        group = [x for x in rows if x['method'] == method]
        item = {'method': method, 'n': len(group)}
        for key in ('reward_x1000', 'nrmse', 'mean_wip', 'cap_adjustment_points'):
            values = np.asarray([x[key] for x in group], dtype=float)
            item[key + '_mean'] = float(values.mean())
            item[key + '_sd'] = float(values.std(ddof=1))
        summary.append(item)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
