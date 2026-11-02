"""Finish the last M4 seed through one more validation window."""
from __future__ import annotations

import json
from pathlib import Path

import train_m4_until_converged as base


def main():
    output = (Path(__file__).resolve().parents[1] / 'outputs/madqn_m4_converged_finish_13104').resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = base.train_job((13104, str(output), 1300))
    (output / 'training_manifest.json').write_text(json.dumps([manifest], indent=2), encoding='utf-8')
    print(json.dumps({'manifest': manifest}, indent=2), flush=True)


if __name__ == '__main__':
    main()
