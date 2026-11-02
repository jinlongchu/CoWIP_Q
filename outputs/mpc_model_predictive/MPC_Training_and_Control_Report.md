# Learned-Model MPC Training and Control Results

## Design

- Collect transitions from M4, PI and legal random policies so the predictor sees multiple policies.
- Train 3 independent one-step networks to predict the next 135-dimensional observation increment and immediate reward.
- Use a 4-step receding horizon and beam width 8 online. Anchor candidates to current PI actions, allow only adjacent action offsets at each station, and replan next step.
- Enforce deterministic cap changes in predicted states and use ensemble disagreement as a light risk penalty.

## Results

| split | reward×1000 | NRMSE | Mean WIP | Cap-change count |
|---|---:|---:|---:|---:|
| test | -54.0383 ± 6.6339 | 0.18339 ± 0.01743 | 14.6259 ± 0.2583 | 157.42 ± 13.47 |
| holdout | -54.0082 ± 8.9423 | 0.18244 ± 0.02383 | 14.6441 ± 0.3093 | 159.45 ± 12.59 |

MPC is not required to beat M4. This experiment checks whether it can serve as an independent, interpretable predictive-control baseline. See `evaluation_per_run.csv`, `mpc_predictor_ensemble.pt` and `comparison_four_madqn_with_mpc.csv`.

Elapsed time: 194.9 seconds.
