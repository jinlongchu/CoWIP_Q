# MADQN M1–M4 Ablation Experiment Report

## Experiment Definition

- M0: existing independent local MADQN results supplied in the archive.
- M1: global coordination features + GRU local Q network.
- M2: M1 + QMIX-style centralized training and decentralized execution.
- M3: M2 + 3-step returns + prioritized experience replay.
- M4: M3 + residual actions guided by the current PI controller; residual 0 follows the PI action.

M1–M4 each use 5 training seeds and 300 episodes per seed, evaluated on the same 12 test seeds. Each configuration has 60 test episodes. This historical M4 is a hybrid controller, not pure MADQN.

## Matched Test Results

| Method | Reward ×1000 | NRMSE | Mean total WIP | Cap-change points |
|---|---:|---:|---:|---:|
| M0 current MADQN | -65.659 ± 12.260 | 0.197 ± 0.035 | 16.851 ± 1.841 | 351.0 ± 7.9 |
| M1 global features + GRU | -50.098 ± 6.954 | 0.153 ± 0.025 | 17.049 ± 1.885 | 351.8 ± 9.1 |
| M2 + QMIX | -51.644 ± 5.339 | 0.163 ± 0.016 | 16.628 ± 1.612 | 330.4 ± 28.4 |
| M3 + n-step+PER | -49.720 ± 6.073 | 0.155 ± 0.024 | 16.897 ± 2.760 | 325.5 ± 29.8 |
| M4 + PI residual guidance | **-43.066 ± 2.970** | **0.145 ± 0.011** | **15.226 ± 0.638** | **265.4 ± 17.6** |
| Current PI (reconstructed) | -47.978 ± 3.906 | 0.168 ± 0.012 | 14.452 ± 0.217 | 129.6 ± 11.9 |

## Conclusions

1. M1 is the strongest observation/memory-only improvement: NRMSE decreases approximately 22.1% from M0, but WIP increases slightly and actions remain frequent.
2. M2 does not further reduce NRMSE, but lowers WIP and cap-change points. QMIX mainly improves coordination and action stability rather than tracking accuracy directly.
3. M3 further improves mean NRMSE and reward over M2, suggesting that n-step returns and prioritized replay help with delayed production feedback, although WIP variance increases.
4. M4 performs best: versus M0, NRMSE falls approximately 26.2%, reward cost 34.4%, WIP 9.6%, and cap-change points 24.4%.
5. Versus current PI, M4 reduces NRMSE approximately 13.4% and improves aggregate reward, but WIP is approximately 5.4% higher and cap-change points are approximately 2.05 times PI. It improves tracking and aggregate reward but does not dominate PI on every metric.
6. The lowest NRMSE is 0.145, still above the 0.10 target. These ablations support further M4 optimization, not a claim that the target has been achieved.

## Interpretation Limits

M0 reuses the original archived MADQN results; M1–M4 use the matched ablation trainer. Changes from M0 to M1 therefore include both the training implementation and listed modules. Incremental comparisons among M1–M4 are better evidence of module contributions. See `evaluation_per_run.csv` and `comparison_madqn_ablation.csv`.
