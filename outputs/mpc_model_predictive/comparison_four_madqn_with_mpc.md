# Four-Strategy Comparison (Learned-Model MPC Added)

| Method | Test samples | Trained models | reward×1000 | NRMSE | Mean WIP | Cap-change count |
|---|---:|---:|---:|---:|---:|---:|
| Fixed CONWIP (training-seed scan) | 12 | 0 | -116.5701 ± 5.9655 | 0.31025 ± 0.00966 | 15.3696 | 1.00 |
| Coordinated PI (reconstructed) | 12 | 0 | -47.9778 ± 3.9057 | 0.16769 ± 0.01186 | 14.4520 | 129.58 |
| MADQN (original 300 episodes) | 60 | 5 | -65.6588 ± 12.2601 | 0.19686 ± 0.03543 | 16.8510 | 351.00 |
| Centralized DQN (existing matched-seed training) | 60 | 5 | -62.9914 ± 10.4461 | 0.17949 ± 0.03337 | 20.0859 | 351.82 |
| M4 independent Q ensemble (extended training) | 12 | 5 | -36.8359 ± 1.7177 | 0.12642 ± 0.00677 | 14.6918 | 231.92 |
| Learned-model MPC (3-model ensemble + PI safety region + beam search) | 12 | 3 | -54.0383 ± 6.6339 | 0.18339 ± 0.01743 | 14.6259 | 157.42 |

Reward is higher-is-better (closer to 0); NRMSE is lower-is-better. MPC trains prediction models rather than a direct policy network. At each decision, it replans over 4 steps using beam width 8 within a safety region around the current PI action.
