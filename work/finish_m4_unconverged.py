"""Extend the M4 runs that did not meet the validation convergence rule."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import train_m4_until_converged as base


def main():
    output = (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged_extend').resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [13101, 13104]
    jobs = [(seed, str(output), 1200) for seed in seeds]
    with ProcessPoolExecutor(max_workers=2) as pool:
        manifests = list(pool.map(base.train_job, jobs))
    (output / 'training_manifest.json').write_text(json.dumps(manifests, indent=2), encoding='utf-8')
    print(json.dumps({'manifests': manifests}, indent=2), flush=True)


if __name__ == '__main__':
    main()
