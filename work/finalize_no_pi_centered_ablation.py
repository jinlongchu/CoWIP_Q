"""Summarize the PI-off M4 ablation against its fixed five-module baseline."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "outputs" / "m4_no_pi_ablation"
TRAIN_SEEDS = list(range(13101, 13106))
TEST_SEEDS = list(range(30101, 30131))
BASELINE = "no_pi"
ABLATIONS = {
    "no_global": 'Global features',
    "no_gru": "GRU",
    "no_qmix": "QMIX",
    "one_step": "3-step n-step",
    "uniform_replay": "PER",
}
METRICS = ("reward_x1000", "nrmse", "mean_wip", "cap_adjustment_points")

def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def bootstrap(values, rng, n=8000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]

def main():
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    expected = {"no_pi", "no_global", "no_gru", "no_qmix", "one_step", "uniform_replay"}
    if {c["key"] for c in protocol["configs"]} != expected or any(c["pi"] for c in protocol["configs"]):
        raise ValueError("The six-condition protocol must keep PI residual guidance OFF in every condition.")
    if protocol["train_seeds"] != TRAIN_SEEDS or protocol["test_seeds"] != TEST_SEEDS:
        raise ValueError("Training/test seeds differ from the previously approved seed sets.")
    rows = read_csv(OUT / "evaluation_per_episode.csv")
    test_rows = [r for r in rows if r["split"] == "test"]
    for config in sorted(expected):
        subset = [r for r in test_rows if r["config"] == config]
        for method, expected_n in (("single", 150), ("q_ensemble_5", 30)):
            measured = [r for r in subset if r["method"] == method]
            if len(measured) != expected_n or {int(r["seed"]) for r in measured} != set(TEST_SEEDS):
                raise ValueError(f"Incomplete fixed-seed test coverage for {config} / {method}.")
    rng = np.random.default_rng(93017)
    contrasts = []
    for method in ("single", "q_ensemble_5"):
        base = {(int(r["run_seed"]), int(r["seed"])): r for r in test_rows
                if r["config"] == BASELINE and r["method"] == method}
        for key, module in ABLATIONS.items():
            target = {(int(r["run_seed"]), int(r["seed"])): r for r in test_rows
                      if r["config"] == key and r["method"] == method}
            for metric in ("reward_x1000", "nrmse"):
                common = sorted(set(base) & set(target))
                diffs = [float(target[k][metric]) - float(base[k][metric]) for k in common]
                if method == "single":
                    # Two-way resampling respects the crossed training-run and test-seed design.
                    by_key = {k: float(target[k][metric]) - float(base[k][metric]) for k in common}
                    draws = []
                    for _ in range(8000):
                        run_ids = rng.choice(TRAIN_SEEDS, size=len(TRAIN_SEEDS), replace=True)
                        test_ids = rng.choice(TEST_SEEDS, size=len(TEST_SEEDS), replace=True)
                        draws.append(np.mean([by_key[(int(rs), int(ts))]
                                              for rs in run_ids for ts in test_ids]))
                    lo, hi = np.quantile(draws, [.025, .975])
                else:
                    lo, hi = bootstrap(diffs, rng)
                contrasts.append({
                    "method": method, "ablation": key, "module_removed": module,
                    "metric": metric, "baseline_mean": float(np.mean([float(base[k][metric]) for k in common])),
                    "ablation_mean": float(np.mean([float(target[k][metric]) for k in common])),
                    "delta_ablation_minus_baseline": float(np.mean(diffs)),
                    "ci95_low": float(lo), "ci95_high": float(hi), "paired_n": len(common),
                })
    (OUT / "paired_contrasts.json").write_text(
        json.dumps(contrasts, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = json.loads((OUT / "training_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 30 or {r["config"] for r in manifest} != expected:
        raise ValueError("Expected five completed training runs for each of the six configurations.")
    labels = {"no_pi": 'New M4 baseline (PI off)', "no_global": 'New M4 without global features',
              "no_gru": 'New M4 without GRU', "no_qmix": 'New M4 without QMIX',
              "one_step": 'New M4 without n-step (1-step)',
              "uniform_replay": 'New M4 without PER (uniform replay)'}
    diagnostic_rows = []
    for run in manifest:
        run["label"] = labels[run["config"]]
        diagnostic_rows.append({
            "config": run["config"], "label": run["label"], "run_seed": run["run_seed"],
            "episodes_completed": run["episodes_completed"], "converged": run["converged"],
            "best_episode": run["best_episode"],
            "best_validation_reward_x1000": run["best_validation_reward_x1000"],
            "best_validation_nrmse": run["best_validation_nrmse"],
            "training_minutes": run["training_seconds"] / 60,
        })
    (OUT / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUT / "training_diagnostics.csv", list(diagnostic_rows[0]), diagnostic_rows)
    print(json.dumps({"test_rows": len(test_rows), "contrasts": len(contrasts),
                      "training_runs": len(diagnostic_rows),
                      "converged_runs": sum(bool(r["converged"]) for r in manifest)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
