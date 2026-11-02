"""Create a dependency-free SVG learning-curve figure and Markdown report."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path


OUT = (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_final_converged').resolve()
RUNS = [13101, 13102, 13103, 13104, 13105]
COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def f(row, key):
    return float(row[key])


def esc(value):
    return html.escape(str(value), quote=True)


def polyline(points, color):
    return f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" />'


def panel(x0, y0, width, height, title, xlabel, ylabel, series, x_max):
    left, right, top, bottom = x0 + 70, x0 + width - 20, y0 + 38, y0 + height - 48
    values = [value for _, rows in series for _, value in rows]
    ymin, ymax = min(values), max(values)
    pad = max((ymax - ymin) * 0.08, 0.001)
    ymin -= pad
    ymax += pad

    def sx(value):
        return left + (value / x_max) * (right - left)

    def sy(value):
        return bottom - ((value - ymin) / (ymax - ymin)) * (bottom - top)

    parts = [f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="white" stroke="#d1d5db"/>']
    parts.append(f'<text x="{x0 + 14}" y="{y0 + 24}" font-size="16" font-family="Arial" font-weight="bold">{esc(title)}</text>')
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = sy(value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{value:.3f}</text>')
    for tick in range(5):
        value = x_max * tick / 4
        x = sx(value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" font-size="11" font-family="Arial">{value:.0f}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>')
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="{y0 + height - 10}" text-anchor="middle" font-size="12" font-family="Arial">{esc(xlabel)}</text>')
    parts.append(f'<text x="{x0 + 17}" y="{(top + bottom) / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 {x0 + 17} {(top + bottom) / 2:.1f})">{esc(ylabel)}</text>')
    for index, (label, rows) in enumerate(series):
        points = [(sx(x), sy(value)) for x, value in rows]
        if len(points) > 1:
            parts.append(polyline(points, COLORS[index]))
        lx = right - 112
        ly = top + 16 + index * 18
        parts.append(f'<line x1="{lx}" y1="{ly - 4}" x2="{lx + 16}" y2="{ly - 4}" stroke="{COLORS[index]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx + 22}" y="{ly}" font-size="11" font-family="Arial">{esc(label)}</text>')
    return '\n'.join(parts)


def make_svg(curves):
    width, height = 1440, 940
    panels = []
    configs = [
        ('Training reward', 'Episode', 'Reward ×1000', 'learning_curve', 'train_reward_x1000'),
        ('Training NRMSE', 'Episode', 'NRMSE', 'learning_curve', 'train_nrmse'),
        ('Validation reward', 'Validation episode', 'Reward ×1000', 'validation', 'validation_reward_x1000'),
        ('Validation NRMSE', 'Validation episode', 'NRMSE', 'validation', 'validation_nrmse'),
    ]
    positions = [(20, 20), (740, 20), (20, 485), (740, 485)]
    for (title, xlabel, ylabel, kind, key), (x, y) in zip(configs, positions):
        series = []
        x_max = 0
        for index, run in enumerate(RUNS):
            rows = curves[run][kind]
            if kind == 'learning_curve':
                pairs = [(float(row['episode']), f(row, key)) for row in rows]
            else:
                pairs = [(float(row['episode']), f(row, key)) for row in rows]
            x_max = max(x_max, max(x for x, _ in pairs))
            series.append((str(run), pairs))
        panels.append(panel(x, y, 680, 420, title, xlabel, ylabel, series, x_max))
    return '\n'.join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="940" viewBox="0 0 1440 940">',
        '<rect width="1440" height="940" fill="#f9fafb"/>',
        '<text x="720" y="18" text-anchor="middle" font-size="18" font-family="Arial" font-weight="bold">M4 Independent Q Ensemble: Extended-Training Learning Curves</text>',
        *panels,
        '</svg>',
    ])


def main():
    curves = {}
    manifests = json.loads((OUT / 'training_manifest.json').read_text(encoding='utf-8'))
    for run in RUNS:
        root = OUT / f'run_{run}'
        curves[run] = {
            'learning_curve': read_csv(root / 'learning_curve.csv'),
            'validation': read_csv(root / 'validation.csv'),
        }
    (OUT / 'learning_curves.svg').write_text(make_svg(curves), encoding='utf-8')

    summary = json.loads((OUT / 'comparison_final.json').read_text(encoding='utf-8'))
    lookup = {(row['method'], row['split']): row for row in summary}
    lines = [
        '# M4 Independent Q Ensemble: Extended-Training Results',
        '',
        '## Conclusions',
        '',
        'All 5 random seeds completed at least 300 training episodes, with the best checkpoint retained after validation curves reached a plateau. The validation criterion is no reward improvement for 4 consecutive validation cycles. Seeds 13102, 13103 and 13105 stopped at 750 episodes, 13101 at 900, and 13104 at 1250.',
        '',
        "See [learning_curves.svg](learning_curves.svg). Raw curves are in each `run_*` directory's `learning_curve.csv` and `validation.csv`.",
        '',
        '## Convergence Overview',
        '',
        '| seed | episodes | Best validation episode | Best validation reward ×1000 | Best validation NRMSE |',
        '|---:|---:|---:|---:|---:|',
    ]
    for item in manifests:
        lines.append(f"| {item['run_seed']} | {item['episodes_completed']} | {item['best_episode']} | {item['best_validation_reward_x1000']:.4f} | {item['best_validation_nrmse']:.4f} |")
    lines += [
        '',
        '## Final Performance',
        '',
        '| Method | split | n | reward×1000 | NRMSE | mean WIP | cap adjustment points |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for method, label in [('m4_final_converged_q_ensemble', 'Final M4 independent Q ensemble'), ('m4_final_converged_individual', 'Mean of 5 individual checkpoints')]:
        for split in ('test', 'holdout'):
            row = lookup[(method, split)]
            lines.append(f"| {label} | {split} | {row['n']} | {row['reward_x1000_mean']:.4f} ± {row['reward_x1000_sd']:.4f} | {row['nrmse_mean']:.5f} ± {row['nrmse_sd']:.5f} | {row['mean_wip_mean']:.4f} | {row['cap_adjustment_points_mean']:.2f} |")
    lines += [
        '',
        '## Comparison with Earlier Results',
        '',
        'On the same test seeds, the previous M4 independent Q ensemble obtained reward×1000 = -39.2008 and NRMSE = 0.13442. This extended-training version obtains -36.8359 and 0.12642: reward improves 2.3650 and NRMSE decreases 0.00800.',
        '',
        'On 20 holdout seeds, previous M4 obtained reward×1000 = -38.3855 and NRMSE = 0.13098. The extended-training version obtains -36.3854 and 0.12451: reward improves 2.0001 and NRMSE decreases 0.00646.',
        '',
        'The current PI baseline obtains reward×1000 = -47.9778 and NRMSE = 0.16769. This M4 independent Q ensemble performs better on test reward and NRMSE.',
        '',
        '## Files',
        '',
        '- [Learning curves](learning_curves.svg)',
        '- [Final summary](comparison_final.json)',
        '- [Evaluation by seed](evaluation_per_run.csv)',
        '- [Training manifest](training_manifest.json)',
    ]
    (OUT / 'M4_Training_Plateau_Report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
