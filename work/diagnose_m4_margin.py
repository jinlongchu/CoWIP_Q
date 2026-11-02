"""Evaluate existing M4 checkpoints with a PI-residual Q-margin gate."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO = (Path(__file__).resolve().parents[1] / 'work/conwip_four_strategy_v3_madqn')
sys.path.insert(0, str(REPO))
import run_madqn_ablation as runner


class MarginGate(nn.Module):
    def __init__(self, base: nn.Module, margin: float, error_threshold: float = 0.0):
        super().__init__()
        self.base = base
        self.margin = float(margin)
        self.error_threshold = float(error_threshold)

    def agent_values(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.base.agent_values(sequence, mask)
        if self.margin <= 0:
            return values
        gated = values.clone()
        # Residual codes 0 and 2 change PI's action.  Subtracting the margin
        # means a residual is selected only when its Q advantage over code 1
        # exceeds the trust-region threshold.
        gated[..., 0] -= self.margin
        gated[..., 2] -= self.margin
        if self.error_threshold > 0:
            # Local observation index 5 is the normalized signed tracking gap.
            # Near the target, force the zero-residual PI fallback.
            close_to_target = sequence[:, :, -1, 5].abs() < self.error_threshold
            gated[..., 0] = torch.where(
                close_to_target, torch.full_like(gated[..., 0], -1e9), gated[..., 0])
            gated[..., 2] = torch.where(
                close_to_target, torch.full_like(gated[..., 2], -1e9), gated[..., 2])
        return gated


def evaluate(checkpoint: str, margin: float, error_threshold: float):
    model = runner.CTDEModel(72)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()
    runner.trainer_model = MarginGate(model, margin, error_threshold).eval()
    rows = []
    for seed in runner.TEST_SEEDS:
        result = runner.run_episode('m4', seed, trainer=None, learn=False)
        result.update(margin=margin, error_threshold=error_threshold,
                      checkpoint=checkpoint, seed=seed)
        rows.append(result)
    return rows


def main():
    source = (Path(__file__).resolve().parents[1] / 'outputs/madqn_ablation/m4')
    output = (Path(__file__).resolve().parents[1] / 'outputs/m4_gate_grid')
    output.mkdir(parents=True, exist_ok=True)
    modes = (
        ('margin_0.01', 0.01, 0.0),
        ('margin_0.01_err_0.02', 0.01, 0.02),
        ('margin_0.01_err_0.04', 0.01, 0.04),
        ('margin_0.01_err_0.06', 0.01, 0.06),
        ('margin_0.01_err_0.08', 0.01, 0.08),
    )
    rows = []
    for checkpoint in sorted(source.glob('run_*/model.pt')):
        for _, margin, error_threshold in modes:
            rows.extend(evaluate(str(checkpoint), margin, error_threshold))
    fields = ['margin', 'error_threshold', 'checkpoint', 'seed', 'reward_x1000',
              'nrmse', 'mean_wip', 'cap_adjustment_points']
    with (output / 'per_run.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name, margin, error_threshold in modes:
        group = [x for x in rows if x['margin'] == margin and
                 x['error_threshold'] == error_threshold]
        summary.append({
            'mode': name,
            'margin': margin,
            'error_threshold': error_threshold,
            'n': len(group),
            'reward_x1000_mean': float(np.mean([x['reward_x1000'] for x in group])),
            'reward_x1000_sd': float(np.std([x['reward_x1000'] for x in group], ddof=1)),
            'nrmse_mean': float(np.mean([x['nrmse'] for x in group])),
            'mean_wip_mean': float(np.mean([x['mean_wip'] for x in group])),
            'cap_adjustment_points_mean': float(np.mean([x['cap_adjustment_points'] for x in group])),
        })
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
